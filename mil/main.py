import sys, argparse, os
import numpy as np
import yaml
import random
import warnings
from timm.optim.adamp import AdamP

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_fscore_support,f1_score,accuracy_score,precision_score,recall_score,balanced_accuracy_score
from sklearn.metrics import confusion_matrix
#from aeon.datasets import load_classification
#from mydataload import loadorean

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision.transforms.functional as VF                    
import torch.nn.functional as F

from utils import *
from lookhead import Lookahead
from models.timemil import TimeMIL

# Suppress all warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    if type(v) == bool:
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def train(trainloader, milnet, criterion, optimizer, epoch, args, device):
    milnet.train()
    x_train = []
    total_loss = 0

    for batch_id, (feats, label) in enumerate(trainloader):
        bag_feats = feats.to(device)#.cuda()
        bag_label = label.to(device)#.cuda()
        
        # Window-based random masking
        if args.dropout_patch > 0:
            selecy_window_indx = random.sample(range(10),int(args.dropout_patch*10))
            inteval = int(len(bag_feats)//10)
            for idx in selecy_window_indx:
                bag_feats[:, idx*inteval:idx*inteval+inteval,:] = torch.randn(1)#.cuda()
   
        optimizer.zero_grad()
   
        if epoch<args.epoch_des:
            x_representation, bag_prediction  = milnet(bag_feats, warmup = True)
        else:
            x_representation, bag_prediction  = milnet(bag_feats, warmup = False)
        
        x_train.append(x_representation.detach().cpu().numpy())
        bag_loss = criterion(bag_prediction, bag_label)
        loss = bag_loss 
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f  total loss: %.4f' % (batch_id, len(trainloader), bag_loss.item(),loss.item()))
        loss.backward()
        
        # avoid the overfitting by using gradient clip
        torch.nn.utils.clip_grad_norm_(milnet.parameters(), 2.0)
        optimizer.step()
        # total_loss = total_loss + loss.item()
        total_loss = total_loss + bag_loss

    x_train = np.concatenate(x_train, axis=0)  
    return x_train, total_loss / len(trainloader)




def test(testloader, milnet, criterion, args, device):
    milnet.eval()
    x_test = []
    # csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []

    with torch.no_grad():
        for batch_id, (feats, label) in enumerate(testloader):
            bag_feats = feats.to(device)
            bag_label = label.to(device)
            x_representation, bag_prediction = milnet(bag_feats)  #b*class
            x_test.append(x_representation.detach().cpu().numpy())

            bag_loss = criterion(bag_prediction, bag_label)
            loss = bag_loss
            total_loss = total_loss + loss.item()

            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (batch_id, len(testloader), loss.item()))
            test_labels.extend([label.cpu().numpy()])
            test_predictions.extend([torch.sigmoid(bag_prediction).cpu().numpy()])
    
    x_test = np.concatenate(x_test, axis=0)
    test_labels = np.vstack(test_labels)
    test_predictions = np.vstack(test_predictions)
    #test_predictions_prob = np.exp(test_predictions)/np.sum(np.exp(test_predictions),axis=1,keepdims=True)
    test_predictions = np.argmax(test_predictions,axis=1)
    test_labels = np.argmax(test_labels,axis=1)
    #print(f"Test labels: {test_labels}, Test predictions: {test_predictions}")

    cm = confusion_matrix(test_labels, test_predictions)
    print("Confusion Matrix:")
    print(cm)
    
    avg_score = accuracy_score(test_labels,test_predictions)
    balanced_avg_score = balanced_accuracy_score(test_labels,test_predictions)

    f1_marco = f1_score(test_labels,test_predictions,average='macro')
    f1_micro = f1_score(test_labels,test_predictions,average='micro')
    
    p_marco = precision_score(test_labels,test_predictions,average='macro')
    p_micro = precision_score(test_labels,test_predictions,average='micro')
    
    r_marco = recall_score(test_labels,test_predictions,average='macro')
    r_micro = recall_score(test_labels,test_predictions,average='micro')
    
    results = [avg_score, balanced_avg_score, f1_marco, f1_micro, p_marco, p_micro, r_marco, r_micro]
    return x_test, total_loss / len(testloader), results



def main():
    parser = argparse.ArgumentParser(description='time classification by TimeMIL')
    parser.add_argument('--dataset', default="sdannce", type=str, help='dataset ')
    parser.add_argument('--path_to_data_dir', default="/home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl")
    parser.add_argument("--config", type=str, default = '../configs/sdannce/fmr1.yaml')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=192, type=int, help='Dimension of the feature size [512] resnet-50 1024') 
    parser.add_argument('--lr', default=0.0005, type=float, help='1e-3 Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=70, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay 1e-4]')
    parser.add_argument('--dropout_patch', default=0.5, type=float, help='Patch dropout rate [0] 0.5')
    parser.add_argument('--dropout_node', default=0.25, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
   
    parser.add_argument('--optimizer', default='adamw', type=str, help='adamw sgd')
    parser.add_argument('--save_dir', default='./savemodel/', type=str, help='the directory used to save all the output')
    parser.add_argument('--epoch_des', default=5, type=int, help='turn on warmup')
    parser.add_argument('--embed', default=192, type=int, help='Number of embedding')
    parser.add_argument('--batchsize', default=16, type=int, help='batchsize')

    parser.add_argument('--if_interval', default=False, type=str2bool, help='if split the whole time series to intervals, each interval as an instance')
    parser.add_argument('--instance_len', default=5, type=int, help='the length of instance')
    parser.add_argument('--if_extract_feature', default=True, type=str2bool, help='if extract feature')
    parser.add_argument('--subseq_len', default=3000, type=int, help='the length of sub sequence')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.save_dir = args.save_dir+'InceptBackbone'
    maybe_mkdir_p(join(args.save_dir, f'{args.dataset}'))
    args.save_dir = make_dirs(join(args.save_dir, f'{args.dataset}'))
    maybe_mkdir_p(args.save_dir)
    
    # <------------- set up logging ------------->
    logging_path = os.path.join(args.save_dir, 'Train_log.log')
    logger = get_logger(logging_path)

    # <------------- save hyperparameters ------------->
    option = vars(args)
    file_name = os.path.join(args.save_dir, 'option.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(option.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

    # criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss(label_smoothing=0.0)#0.01
    criterion = nn.BCEWithLogitsLoss() # one-vs-rest binary MIL
    # scaler = GradScaler()

    if args.dataset in ["mabe_mice", "mocap","sdannce", "sdannce_kinematic" , "eyetract" ]:
        if args.dataset == "mabe_mice":
            X = np.load("/home/rguo_hpc/myfolder/code/pipeline/pretrain/outputs/representations/mae_representations.npy")[1600:] #(1600, 1800, 128)
            y = load_pickle('/home/rguo_hpc/myfolder/code/mocap/data/mabe_mice/mouse_test_labels.pkl')["strain"] #(3736,)
        elif args.dataset == "mocap":
            """
            # hbehave/MAE
            mouse_X = np.load("/home/rguo_hpc/myfolder/code/mocap/outputs/mocap/experiment1/test_submission_0.npy", allow_pickle=True).item()
            X = []
            for mouse_name, indices in mouse_X["frame_number_map"].items():
                X.append(mouse_X['embeddings'][indices[0]:indices[1]]) #(13, 1800)
            X = np.stack(X)
            """
            Xtr = np.load("/home/rguo_hpc/myfolder/mocap/outputs/representations/mae_mocap_tr.npy", allow_pickle=True)
            Xte = np.load("/home/rguo_hpc/myfolder/mocap/outputs/representations/mae_mocap_val.npy", allow_pickle=True)
            with open("/home/rguo_hpc/myfolder/data/mocap/data_FL2_v0.pkl", 'rb') as file:
                data = pickle.load(file)
            label_tr = []
            label_te = []
            for dataset_name in ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"]:
                for mouse_name in split["fold_1"][dataset_name]["train"]:
                    label_tr = label_tr + data[dataset_name][mouse_name]["drug"]
                for mouse_name in split["fold_1"][dataset_name]["valid"]:
                    label_te = label_te + data[dataset_name][mouse_name]["drug"]
            mapping = {s: i for i, s in enumerate(set(label_tr))}
            print(f"Mapping of drug labels to integers: {mapping}")
            ytr = [mapping[s] for s in label_tr]
            yte = [mapping[s] for s in label_te]
            """                    
            drug = []
            concentration = []
            for dataset_name in ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"]:
                for mouse_name in data[dataset_name].keys():
                    drug = drug + data[dataset_name][mouse_name]["drug"]
                    concentration = concentration + data[dataset_name][mouse_name]["concentration"]
            mapping = {s: i for i, s in enumerate(set(drug))}
            y = [mapping[s] for s in drug]
            """
        elif args.dataset == "eyetract":
            #X = np.load("/home/rguo_hpc/myfolder/mocap/outputs/eyetrack/representations/mae_eyetract_tr.npy")
            raw_data = np.load("/home/rguo_hpc/myfolder/data/eye/eyetrack.pkl", allow_pickle=True)
            X = raw_data["X"]
            y = raw_data["y"]
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        elif "sdannce" in args.dataset:
            with open(args.path_to_data_dir, 'rb') as file:
                data = pickle.load(file)
            N = int(90000 / args.subseq_len) # number of sequences per sequence
            
            if args.dataset == "sdannce":
                print("Loading SDANNCE dataset...")
                Xtr = np.load("/home/rguo_hpc/myfolder/mocap/outputs/sdannce/fmr1/50/representations/mae_sdannce_tr.npy")[:, 25:3025]
                Xte = np.load("/home/rguo_hpc/myfolder/mocap/outputs/sdannce/fmr1/50/representations/mae_sdannce_val.npy")[:, 25:3025]

            elif args.dataset == "sdannce_kinematic":
                Xtr = np.load("/home/rguo_hpc/myfolder/mocap/swav/kinematic/data/kinematic_tr.py.npy")
                Xte = np.load("/home/rguo_hpc/myfolder/mocap/swav/kinematic/data/kinematic_val.py.npy")
                Xtr = Xtr.reshape(-1, args.subseq_len, Xtr.shape[-1])
                Xte = Xte.reshape(-1, args.subseq_len, Xtr.shape[-1])
            
            #X_tr, X_te = []
            with open(args.config) as f:
                split = yaml.safe_load(f)
            label_tr = []
            label_te = []
            for mouse in split["fold_1"]["train"]:
                num_seq = len(data[mouse]['ratid'])
                gen  = str(data[mouse]["ratgen"])
                label_tr = label_tr + [gen for i in range(num_seq * N)]
                #m1 = data[mouse]['m1'].reshape(num_seq, 90000, -1)
                #X_tr.append(m1.reshape(num_seq * N, args.subseq_len, -1))
            for mouse in split["fold_1"]["valid"]:
                num_seq = len(data[mouse]["ratid"])
                gen = str(data[mouse]["ratgen"])
                label_te = label_te + [gen for i in range(num_seq * N)]
                #m1 = data[mouse]['m1'].reshape(num_seq, 90000, -1)
                #X_te.append(m1.reshape(num_seq * N, args.subseq_len, -1))
            #Xtr = np.concatenate(X_tr, axis=0)
            #Xte = np.concatenate(X_te, axis=0)
            mapping = {s: i for i, s in enumerate(set(label_tr))}
            print(f"Mapping of ratgen labels to integers: {mapping}")
            ytr = [mapping[s] for s in label_tr]
            yte = [mapping[s] for s in label_te]

        Xtr = torch.from_numpy(Xtr).float()
        Xte = torch.from_numpy(Xte).float() 
        ytr = F.one_hot(torch.tensor(ytr)).float()
        yte = F.one_hot(torch.tensor(yte)).float()
        print(Xtr.shape, Xte.shape, ytr.shape, yte.shape)
        
        trainset = TensorDataset(Xtr, ytr)
        testset = TensorDataset(Xte, yte)
        args.feats_size = Xtr.shape[-1]
        args.num_classes =  len(set(label_tr))
        seq_len = Xtr.shape[1]
    """
    elif args.dataset in ["moseq","mabe_mouse_72"]:
        Xtr, ytr = load_classification_pkl(path = "../../../data/MaBe/mouse/hbehave/train_lights_hbehave.pkl") 
        Xte, yte = load_classification_pkl(path = "../../../data/MaBe/mouse/hbehave/test_lights_hbehave.pkl")
        if args.if_interval:
            Xtr = torch.from_numpy(Xtr).permute(0,2,3,1).float() #(400, 1200, 30, 10)  (B, N, T, D)
            Xte = torch.from_numpy(Xte).permute(0,2,3,1).float()
        else:
            Xtr = torch.from_numpy(Xtr)#.permute(0,2,1).float()# [400, len, dim]
            Xte = torch.from_numpy(Xte)#.permute(0,2,1).float()# [101, len, 10]
        ytr = F.one_hot(torch.tensor(ytr)).float()
        yte = F.one_hot(torch.tensor(yte)).float()
        trainset = TensorDataset(Xtr,ytr)
        testset = TensorDataset(Xte, yte)

        args.feats_size = Xte.shape[-1]
        L_in = Xte.shape[-1]
        num_classes = yte.shape[-1]
        args.num_classes =  yte.shape[-1]
        seq_len = Xte.shape[1]
    """    
    # <------------- define TimeMIL network ------------->
    milnet = TimeMIL(in_features=args.feats_size, mDim=args.embed, n_classes=args.num_classes, 
                     dropout=args.dropout_node, max_seq_len=seq_len, if_extract_feature=args.if_extract_feature, 
                     if_interval=args.if_interval, instance_len=args.instance_len)
    milnet = milnet.to(device) 
    
    # total number of trainable model parameters
    total_params = sum(p.numel() for p in  milnet.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')

    if  args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer = Lookahead(optimizer)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(milnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay) 
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer) 
    elif args.optimizer == 'adamp':
        optimizer = AdamP(milnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer =Lookahead(optimizer) 
    
    trainloader = DataLoader(trainset, args.batchsize, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    testloader = DataLoader(testset, 128, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    best_score = 0
    best_balanced_score = 0
    save_path = join(args.save_dir, 'weights')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(join(args.save_dir,'lesion'), exist_ok=True)
    results_best = None
    
    for epoch in range(1, args.num_epochs + 1):
        writer = SummaryWriter(log_dir=args.save_dir+'/logs/')
        x_train, train_loss_bag = train(trainloader, milnet, criterion, optimizer, epoch, args, device) # iterate all bags
        x_test, test_loss_bag, results= test(testloader, milnet, criterion, args, device)
        writer.add_scalar("Loss/Train", train_loss_bag, epoch)
        writer.add_scalar("Loss/Test", test_loss_bag, epoch)

        [avg_score,balanced_avg_score,f1_marco,f1_micro,p_marco,p_micro,r_marco,r_micro] = results
        writer.add_scalar("Loss/Accuracy", avg_score, epoch)
        logger.info('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, accuracy: %.4f, bal. average score: %.4f, f1 marco: %.4f   f1 mirco: %.4f  p marco: %.4f   p mirco: %.4f r marco: %.4f   r mirco: %.4f  ' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, balanced_avg_score,f1_marco,f1_micro, p_marco, p_micro, r_marco, r_micro)) 
        
        # scheduler.step()
        current_score = avg_score
        current_balanced_score = balanced_avg_score
        is_tied = abs(current_balanced_score - best_balanced_score) < 1e-7
        is_better = (current_balanced_score > best_balanced_score or (is_tied and current_score > best_score))
        
        if is_better:
            results_best = results
            best_score = current_score
            best_balanced_score = current_balanced_score
            print(current_score)
            save_name = os.path.join(save_path, 'best_model.pth')
            #x = np.concatenate((x_train, x_test), axis=0)
            np.save(os.path.join(args.save_dir, 'representations.npy'), x_test) # save test only

            torch.save(milnet.state_dict(), save_name)
            logger.info('Best model saved at: ' + save_name)
            # logger.info('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
    
    writer.close()

    [avg_score, balanced_avg_score, f1_marco, f1_micro, p_marco, p_micro, r_marco, r_micro,] = results_best
    logger.info('\r Best  Results: accuracy: %.4f, bal. average score: %.4f, f1 marco: %.4f   f1 mirco: %.4f  p marco: %.4f   p mirco: %.4f r marco: %.4f   r mirco: %.4f ' % 
                  (avg_score, balanced_avg_score, f1_marco, f1_micro, p_marco, p_micro, r_marco, r_micro)) 
        # if args.weight_div>0:
        #     if epoch%10==0:
        #         print('--------------------Clustering--------------------\n')
        #         cluster_idx_dict = pre_cluter(trainloader, milnet, criterion, optimizer, args,init= False)
        #         print('--------------------Clustering finished--------------------\n')
if __name__ == '__main__':
    main()