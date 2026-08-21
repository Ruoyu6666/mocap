import sys, argparse, os
import numpy as np
import random
import warnings
from timm.optim.adamp import AdamP

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
 
from utils import *
from lookhead import Lookahead
from models.timemil import TimeMIL
 
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
    total_loss = 0

    for batch_id, (feats, label) in enumerate(trainloader):
        bag_feats = feats.to(device)
        bag_label = label.to(device)
        
        # Window-based random masking
        if args.dropout_patch>0:
            selecy_window_indx = random.sample(range(10),int(args.dropout_patch*10))
            inteval = int(len(bag_feats)//10)
            for idx in selecy_window_indx:
                bag_feats[:, idx*inteval:idx*inteval+inteval,:] = torch.randn(1).cuda()
   
        optimizer.zero_grad()
   
        if epoch<args.epoch_des:
            x_representation, bag_prediction = milnet(bag_feats,warmup = True)
        else:
            x_representation, bag_prediction = milnet(bag_feats,warmup = False)
        bag_loss = criterion(bag_prediction, bag_label)
       
        loss = bag_loss 
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f  total loss: %.4f' % (batch_id, len(trainloader), bag_loss.item(),loss.item()))
        loss.backward()
        
        # avoid the overfitting by using gradient clip
        torch.nn.utils.clip_grad_norm_(milnet.parameters(), 2.0)
        optimizer.step()
        total_loss += bag_loss
      
    return total_loss / len(trainloader)




def test(testloader, milnet, criterion, args, device):
    milnet.eval()
    x_test = []
    total_loss = 0
    test_labels = []
    test_predictions_raw = []

    with torch.no_grad():
        for batch_id, (feats, label) in enumerate(testloader):
            bag_feats = feats.to(device)
            bag_label = label.to(device)
            x_representation, bag_prediction  = milnet(bag_feats)
            x_test.append(x_representation.detach().cpu().numpy())
            bag_loss = criterion(bag_prediction, bag_label)
            
            loss = bag_loss
            total_loss = total_loss + loss.item()

            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (batch_id, len(testloader), loss.item()))

            test_labels.extend([label.cpu().numpy()])
            test_predictions_raw.extend([torch.sigmoid(bag_prediction).cpu().numpy()])

    x_test = np.concatenate(x_test, axis=0)

    test_labels_onehot = np.vstack(test_labels)
    test_predictions_raw = np.vstack(test_predictions_raw)

    test_predictions_prob = np.exp(test_predictions_raw)/np.sum(np.exp(test_predictions_raw), axis=1, keepdims=True)
    test_predictions = np.argmax(test_predictions_raw, axis=1)
    test_labels = np.argmax(test_labels_onehot,axis=1)

    avg_score = accuracy_score(test_labels, test_predictions)
    balanced_avg_score = balanced_accuracy_score(test_labels, test_predictions)
    f1_marco = f1_score(test_labels,test_predictions,average='macro')
    f1_weighted = f1_score(test_labels, test_predictions, average='weighted')

    results = [avg_score, balanced_avg_score, f1_marco, f1_weighted]
    return x_test, total_loss / len(testloader), results, test_labels, test_predictions, test_predictions_prob


def compute_metrics(test_labels, test_predictions):
    """Same metric computation as inside test(), factored out so it can be reused on OOF results."""
    avg_score = accuracy_score(test_labels, test_predictions)
    balanced_avg_score = balanced_accuracy_score(test_labels, test_predictions)
    f1_macro = f1_score(test_labels, test_predictions, average='macro')
    f1_weighted = f1_score(test_labels, test_predictions, average='weighted')
 
    return {'accuracy': avg_score, 'balanced_accuracy': balanced_avg_score, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted,}


def main():
    parser = argparse.ArgumentParser(description='time classification by TimeMIL')
    parser.add_argument('--dataset', default="eyetract", type=str, help='dataset ')
    parser.add_argument('--data_path', default="/home/rguo_hpc/myfolder/code/mocap/data/")
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers used in dataloader [4]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512] resnet-50 1024')
    parser.add_argument('--lr', default=5e-4, type=float, help='1e-3 Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=70, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay 1e-4]')
    parser.add_argument('--dropout_patch', default=0.5, type=float, help='Patch dropout rate [0] 0.5')
    parser.add_argument('--dropout_node', default=0.2, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
   
    parser.add_argument('--optimizer', default='adamw', type=str, help='adamw sgd')
    parser.add_argument('--save_dir', default='./savemodel/', type=str, help='the directory used to save all the output')
    parser.add_argument('--epoch_des', default=5, type=int, help='turn on warmup')
    parser.add_argument('--embed', default=128, type=int, help='Number of embedding')
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')

    parser.add_argument('--if_interval', default=False, type=str2bool, help='if split the whole time series to intervals, each interval as an instance')
    parser.add_argument('--instance_len', default=5, type=int, help='the length of instance')
    parser.add_argument('--if_extract_feature', default=True, type=str2bool, help='if extract feature')
    
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

    criterion = nn.BCEWithLogitsLoss() # one-vs-rest binary MIL

    """
    # <------------- load data ------------->
    if args.dataset in ["mabe_mice", "mocap"]:
        if args.dataset == "mocap":
            # k fold validation based on mouse
            mocap_fold_1 = {
                "CP1A": {"train": ["M14", "M15", "M19"], "valid": ["M1"]},
                "CP1B": {"train": ["M2", "M3", "M4", "M5", "M6"], "valid": ["M1"]},
                "INH1": {"train": ["M2", "M3", "M4", "M5", "M7", "M8", "M9", "M10"], "valid": ["M1", "M6"]},
                "INH2": {"train": ["M2", "M3", "M4", "M5", "M7", "M8", "M9", "M10", "M12"], "valid": ["M1", "M6", "M11"]},
                "MOS1aD": {"train": ["M5", "M6", "M8", "M9", "M10"], "valid": ["M4"]}
            }
            mocap_fold_2 = {
                "CP1A": {"train": ["M1", "M15", "M19"], "valid": ["M14"]},
                "CP1B": {"train": ["M1", "M3", "M4", "M5", "M6"], "valid": ["M2"]},
                "INH1": {"train": ["M1", "M3", "M4", "M5", "M6", "M8", "M9", "M10"], "valid": ["M2", "M7"]},
                "INH2": {"train": ["M1", "M3", "M4", "M5", "M6", "M8", "M9", "M10", "M11"], "valid": ["M2", "M7", "M12"]},
                "MOS1aD": {"train": ["M4", "M6", "M8", "M9", "M10"], "valid": ["M5"]}}
            mocap_fold_3 = {
                "CP1A": {"train": ["M1", "M14", "M19"], "valid": ["M15"]},
                "CP1B": {"train": ["M1", "M2", "M4", "M5", "M6"], "valid": ["M3"]},
                "INH1": {"train": ["M1", "M2", "M4", "M5", "M6", "M7", "M9", "M10"], "valid": ["M3", "M8"]},
                "INH2": {"train": ["M1", "M2", "M4", "M5", "M6", "M7", "M9", "M11", "M12"], "valid": ["M3", "M8", "M10"]},
                "MOS1aD": {"train": ["M4", "M5", "M8", "M9", "M10"], "valid": ["M6"]}}
            mocap_fold_4 = {
                "CP1A": {"train": ["M1", "M14", "M15"],  "valid": ["M19"]},
                "CP1B": {"train": ["M1", "M2", "M3", "M5", "M6"],  "valid": ["M4"]},
                "INH1": {"train": ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M10"], "valid": ["M4", "M9"]},
                "INH2": {"train": ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M10", "M12"], "valid": ["M4", "M9", "M11"]},
                "MOS1aD": {"train": ["M4", "M5", "M6", "M9", "M10"], "valid": ["M8"]}}
            folds = [mocap_fold_1, mocap_fold_2, mocap_fold_3, mocap_fold_4]
            # Skeleton MAE Style
            X = np.load("/home/rguo_hpc/myfolder/code/mocap/outputs/representations/mae_mocap.npy", allow_pickle=True)
            print(X.shape)
            X = X.reshape(202, 1200, -1)
            with open("/home/rguo_hpc/myfolder/code/mocap/data/mocap/data_CLB.pkl", 'rb') as file:
                data = pickle.load(file)
            drug = []
            for dataset_name in ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"]:
                drug = drug + data[dataset_name]["drug"]
            mapping = {s: i for i, s in enumerate(set(drug))}
            y = [mapping[s] for s in drug]
    """
    if args.dataset == "eyetract":
        raw_data = np.load("/home/rguo_hpc/myfolder/data/eye/eyetrack.pkl", allow_pickle=True)
        X = raw_data["X"]
        #X = np.pad(X, ((0, 0), (0, 1), (0, 0)), mode='edge')
        y = raw_data["y"]
    
    num_classes = len(set(y.tolist()))
    args.num_classes = num_classes
    seq_len = X.shape[1]
    args.feats_size = X.shape[-1]

    # <------------- split data ------------->   
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    oof_labels = np.full(len(X), -1, dtype=np.int64)
    oof_preds = np.full(len(X), -1, dtype=np.int64)
    oof_probs = np.zeros((len(X), num_classes), dtype=np.float32)
    oof_feats = np.zeros((len(X), 72, args.embed), dtype=np.float32)

    per_fold_metrics = []
    X_order = np.zeros((len(X), 71, args.feats_size), dtype=np.float32)


    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        Xtr, Xte = X[train_index], X[test_index]
        print(test_index)
        X_order[test_index] = X[test_index]
        ytr, yte = y[train_index], y[test_index]
        Xtr = torch.from_numpy(Xtr).float()#.permute(0,2,1).float() #(2802, 128, 1800) -> (2802, 1800, 128)
        Xte = torch.from_numpy(Xte).float()#.permute(0,2,1).float()  
        ytr = F.one_hot(torch.tensor(ytr), num_classes=num_classes).float()
        yte = F.one_hot(torch.tensor(yte), num_classes=num_classes).float()
        trainset = TensorDataset(Xtr, ytr)
        testset = TensorDataset(Xte, yte)

        # <------------- define MIL network ------------->
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

        best_score = -1
        save_path = join(args.save_dir, f'weights_fold{fold_idx}')
        os.makedirs(save_path, exist_ok=True)
        results_best = None
        best_fold_labels = None
        best_fold_preds = None
        best_fold_probs = None
        
        for epoch in range(1, args.num_epochs + 1):
            train_loss_bag = train(trainloader, milnet, criterion, optimizer, epoch, args, device) # iterate all bags
            x_test, test_loss_bag, results, test_labels_ep, test_preds_ep, test_probs_ep= test(testloader, milnet, criterion, args, device)
            [avg_score, balanced_avg_score, f1_macro, f1_weighted] = results
        
            logger.info('\r Fold [%d] Epoch [%d/%d] train loss: %.4f test loss: %.4f, accuracy: %.4f, bal. accuracy: %.4f, f1 macro: %.4f, f1 weighted: %.4f' %
                (fold_idx, epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, balanced_avg_score, f1_macro, f1_weighted))

            current_score = avg_score
            if current_score >= best_score:
                best_score = current_score
                results_best = results
                best_fold_labels = test_labels_ep
                best_fold_preds = test_preds_ep
                best_fold_probs = test_probs_ep
                feats  = x_test

                save_name = os.path.join(save_path, 'best_model.pth')
                torch.save(milnet.state_dict(), save_name)
                logger.info('Best model saved at: ' + save_name)
        
        # <------------- stash this fold's best-epoch OOF predictions ------------->
        oof_labels[test_index] = best_fold_labels
        oof_preds[test_index] = best_fold_preds
        oof_probs[test_index] = best_fold_probs
        oof_feats[test_index] = feats
 
        [avg_score, balanced_avg_score, f1_macro, f1_weighted] = results_best
        per_fold_metrics.append(results_best)
        logger.info('\r Fold [%d] Best Results: accuracy: %.4f, bal. accuracy: %.4f, f1 macro: %.4f, f1 weighted: %.4f' %
            (fold_idx, avg_score, balanced_avg_score, f1_macro, f1_weighted))


    # <------------- aggregate per-fold metrics (mean +/- std) ------------->
    per_fold_metrics = np.array(per_fold_metrics)
    metric_names = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']
    print('\n===== Per-fold metrics (mean +/- std across folds) =====')
    for i, name in enumerate(metric_names):
        print(f'{name}: {per_fold_metrics[:, i].mean():.4f} +/- {per_fold_metrics[:, i].std():.4f}')
 
    # <------------- overall metrics on concatenated out-of-fold results ------------->
    assert (oof_labels != -1).all(), "Some samples were never placed in a test fold — check StratifiedKFold split."
    oof_metrics = compute_metrics(oof_labels, oof_preds)
    print('\n===== Overall (out-of-fold, full dataset) metrics =====')
    for k, v in oof_metrics.items():
        print(f'{k}: {v:.4f}')
 
    # <------------- final confusion matrix on the full (out-of-fold) dataset ------------->
    cm = confusion_matrix(oof_labels, oof_preds, labels=list(range(num_classes)))
    print('\n===== Confusion matrix (rows: true label, cols: predicted label) =====')
    print(cm)

    # <------------- save X, y, OOF results and confusion matrix ------------->
    out_path = os.path.join(args.save_dir, 'cv_results.npz')
    np.savez(out_path, X=X_order, oof_labels=oof_labels, oof_preds=oof_preds, oof_probs=oof_probs, oof_feats=oof_feats, confusion_matrix=cm)
    print(f'\nSaved X, y, out-of-fold predictions and confusion matrix to: {out_path}')
 
 
if __name__ == '__main__':
    main()