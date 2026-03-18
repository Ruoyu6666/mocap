import argparse
import numpy as np
import pdb
from tqdm import tqdm
from itertools import islice
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from myfolder.code.mocap.trainers.utils import *
from models.VQ.VQVAE import VQVAE
from dataset.mabe_mice import MABeMouseDataset
from dataset.latent import LatentRepresentationDataset



def get_args_parser():

    parser = argparse.ArgumentParser("VQ-VAE Training & Compute Representation", add_help=False)
    
    """Model Hyperparameters"""
    parser.add_argument("--in_dim", type=int, default=128)
    parser.add_argument("--n_hiddens", type=int, default=128)         # h_dim
    parser.add_argument("--n_residual_hiddens", type=int, default=64) # res_h_dim
    parser.add_argument("--n_residual_layers", type=int, default=1)   # n_res_layers
    parser.add_argument("--embedding_dim", type=int, default=64)      # e_dim
    parser.add_argument("--n_embeddings", type=int, default=100)      # K: n_e 512
    parser.add_argument("--beta", type=float, default=.25)

    """Dataset and DataLoader parameters"""
    parser.add_argument("--dataset",  type=str, default='mabe_mouse')
    #parser.add_argument("--path_to_data_dir", type=str, default='/home/rguo_hpc/myfolder/data/MaBe/mouse/mouse_triplet_train.npy')
    parser.add_argument("--path_to_data_dir", type=str, default='/home/rguo_hpc/myfolder/code/pipeline/pretrain/models/BehaveMAE/outputs/mice/experiment/submission_1.npy')

    
    parser.add_argument("--num_frames", default=300, type=int)
    parser.add_argument("--sliding_window", default=1, type=int)
    parser.add_argument("--sampling_rate", default=1, type=int)
    parser.add_argument("--if_fill_holes", default=False, type=str2bool)
    parser.add_argument("--patch_size", default=(3, 1, 24), type = int )
    parser.add_argument("--compression_factor", type=int, default=4)
    
    """Dataset augmentation and preprocessing"""
    parser.add_argument("--data_augment", default=True, type=str2bool)
    parser.add_argument("--centeralign", action="store_true")
    parser.add_argument("--include_testdata", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",)
    
    """Training parameters"""
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_updates", type=int, default=5200)
    parser.add_argument("--learning_rate", type=float, default=2e-3) # do not reduce learning rate for now, as training is not that long. Can add lr scheduler later if needed.
    
    """Saving and logging"""
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="./outputs/") #  models, results, checkpoints
    parser.add_argument("--ckpt_path", type=str, default="./outputs/models/vqvae_model.pth")
    
    """Type of job"""
    parser.add_argument("--job", type=str, choices=["train", "compute_representations"])

    return parser.parse_args()




def train(model, loader_train, optimizer, device, writer, timestamp, args):
    # checkpoint path
    if os.path.exists(os.path.join(args.save_dir, 'checkpoints')):
        print('Checkpoint Directory Already Exists - if continue will overwrite files inside. Press c to continue.')
        pdb.set_trace()
    else:
        os.makedirs(os.path.join(args.save_dir, 'checkpoints'))

    model = model.to(device)
    num_epochs = int(args.n_updates / len(loader_train) + 0.5)
    print('Number of epochs to train:', num_epochs)
    best_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        model.train()
        results = {'embedding_loss': 0, 
                   'recon_errors': 0, 
                   'total_loss': 0,
                   'perplexities': 0}
        
        for i, x in enumerate(tqdm(loader_train, total=len(loader_train))):
        #for i, (x, _) in enumerate(tqdm(islice(loader_train, 100), total=100)): # len(loader_train): 45050
            x = x.to(device)
            x = torch.permute(x, (0, 2, 1)) # [32, 128, 600]
            optimizer.zero_grad()

            embedding_loss, x_hat, perplexity, z_q, min_encoding_indices = model(x)
            recon_loss = torch.mean((x_hat - x)**2)
            loss = recon_loss + embedding_loss
            loss.backward()
            optimizer.step()
            results["embedding_loss"] += embedding_loss.item()
            results["recon_errors"] += recon_loss.item()
            results["total_loss"]   += loss.item()
            results["perplexities"] += perplexity.item()
            
            """
            if i % args.log_interval == 0:
                if args.save:
                    hyperparameters = args.__dict__.save_model_and_results(model, results, hyperparameters, args.filename)
                #writer.add_scalar('Train/Recon_Loss', recon_loss.item(), step)
                #writer.add_scalar('Train/Perplexity', perplexity.item(), step)
                #writer.add_scalar('Train/Total_Loss', loss.item(), step)
            """
        avg_embed_error = results["embedding_loss"] / len(loader_train)
        avg_recon_error = results["recon_errors"] / len(loader_train)
        avg_perplexity = results["perplexities"] / len(loader_train)
        avg_total_loss = results["total_loss"] / len(loader_train)

        print(f'Epoch {epoch}/{num_epochs} - Loss: {avg_total_loss:.4f},'
              f'Recon: {avg_recon_error:.4f}, Embed: {avg_embed_error:.4f}, Perplexity: {avg_perplexity:.2f}')
        
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, args)
            print('Saved best model at epoch ', epoch)
        
        """
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            save_checkpoint(model, optimizer, epoch, args)
            print('Saved best model at epoch ', epoch)
        """
    
    # Save final model and results
    save_model(model, optimizer, args) # CHEKC PATH
    save_results(results, args)







def compute_representations(model, loader, device ,args):

    os.makedirs(args.save_dir + '/representations', exist_ok=True)
    model = model.to(device)
    model.eval()
    
    #all_representations = []
    all_encoding =  []
    all_encoding_indices = []
    #all_embeddings = []

    with torch.no_grad():
        for i, x in enumerate(loader):
        #for i, x in enumerate(tqdm(islice(loader, 100), total=100)):
            x = x.to(device)
            x = x.permute(0, 2, 1)
            z = model.encoder(x)
            z = model.pre_quant_conv(z)
            vq_loss, min_encodings, perplexity, min_encoding_indices = model.vq_layer(z)

            #all_representations.append(torch.squeeze(x_recon).cpu().numpy())
            all_encoding.append(torch.squeeze(min_encodings).permute(1,0).cpu().numpy())
            all_encoding_indices.append(torch.squeeze(min_encoding_indices).cpu().numpy())


    #all_representations = np.stack(all_representations, axis=0)
    all_encoding = np.stack(all_encoding, axis=0)
    all_encoding_indices = np.stack(all_encoding_indices, axis=0)

    np.save(args.save_dir + '/representations/vqvae_encodings.npy', all_encoding)
    np.save(args.save_dir + '/representations/vqvae_encoding_indices.npy', all_encoding_indices)
    
    codebook = model.vq_layer.embedding.weight.cpu().detach().numpy()
    np.save(args.save_dir + '/representations/vqvae_codebook.npy', codebook)





if __name__ == "__main__":

    timestamp = readable_timestamp()
    args = get_args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """Set up model"""
    model = VQVAE(args.in_dim, args.n_hiddens, 
                  args.n_residual_layers,  args.n_residual_hiddens,
                  args.n_embeddings, args.embedding_dim, args.beta, 
                  compression_factor=args.compression_factor).to(device)


if args.job == "train":
    """Set up data set and data loaders"""
    """
    dataset_train = MabeMouseDataset(path_to_data_dir=args.path_to_data_dir,
                                     sampling_rate=args.sampling_rate,
                                     num_frames=args.num_frames, 
                                     sliding_window=args.num_frames-1,
                                     if_fill=args.if_fill_holes,
                                     patch_size=args.patch_size,
                                     cache_path=args.cache_path, cache=args.cache,
                                     augmentations=args.data_augment, #centeralign=args.centeralign,
                                     include_testdata=args.include_testdata,)
    
    """
    dataset_train = LatentRepresentationDataset(path_to_latent_representations=args.path_to_data_dir)
    

    loader_train = DataLoader(dataset_train, #sampler=sampler_train,
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=args.pin_mem, drop_last=True,)
    

    """Set up optimizer and training loop"""
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    train(model, loader_train, optimizer, device, None, timestamp, args)






if args.job == "compute_representations":
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device, weights_only=False)["model"])
    #path_test_data = args.path_to_data_dir.replace("representations_train", "representations_test")
    path_test_data = args.path_to_data_dir
    print(args.path_to_data_dir)
    """
    dataset = MabeMouseDataset(path_to_data_dir=path_test_data,
                                sampling_rate=args.sampling_rate,
                                num_frames=args.num_frames, 
                                sliding_window=args.sliding_window,
                                if_fill=args.if_fill_holes,
                                patch_size=args.patch_size,
                                cache_path=args.cache_path, cache=args.cache,
                                augmentations=None,)
    """
    dataset = LatentRepresentationDataset(path_to_latent_representations=path_test_data, if_include_test=False)

    loader_test = DataLoader(dataset, #sampler=sampler_test, batch_size=args.batch_size, 
                             num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,)
    
    compute_representations(model, loader_test, device, args)
