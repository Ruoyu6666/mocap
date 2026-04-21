# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim


from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast                     
import torch.nn.functional as F

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_fscore_support,f1_score,accuracy_score,precision_score,recall_score,balanced_accuracy_score
from sklearn.datasets import load_svmlight_file

# comppute confustion metrix
from sklearn.metrics import multilabel_confusion_matrix #gives a 2×2 confusion matrix for each class (one-vs-rest style).
from sklearn.metrics import confusion_matrix # standard confucsion matrix

from collections import OrderedDict

from aeon.datasets import load_classification
from mydataload import loadorean

import random
import warnings
from timm.optim.adamp import AdamP

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



def train(trainloader, milnet, criterion, optimizer, epoch, args):
    milnet.train()
    total_loss = 0
    
    
    for batch_id, (feats, label) in enumerate(trainloader):
        bag_feats = feats.cuda()
        bag_label = label.cuda()
        
        # Window-based random masking
        if args.dropout_patch > 0:
            selecy_window_indx = random.sample(range(10),int(args.dropout_patch*10))
            inteval = int(len(bag_feats)//10)
            for idx in selecy_window_indx:
                bag_feats[:, idx*inteval:idx*inteval+inteval,:] = torch.randn(1).cuda()
   
        optimizer.zero_grad()
   
        if epoch<args.epoch_des:
            bag_prediction  = milnet(bag_feats, warmup = True)
        else:
            bag_prediction  = milnet(bag_feats, warmup = False)
        bag_loss = criterion(bag_prediction, bag_label)
        loss = bag_loss 
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f  total loss: %.4f' % \
                            (batch_id, len(trainloader), bag_loss.item(),loss.item()))
        loss.backward()
        
        # avoid the overfitting by using gradient clip
        torch.nn.utils.clip_grad_norm_(milnet.parameters(), 2.0)
        optimizer.step()
        # total_loss = total_loss + loss.item()
        total_loss = total_loss + bag_loss
      
    return total_loss / len(trainloader)




def test(testloader, milnet, criterion, args):
    milnet.eval()
    # csvs = shuffle(test_df).reset_index(drop=True)
    total_loss = 0
    test_labels = []
    test_predictions = []

    with torch.no_grad():
        for batch_id, (feats, label) in enumerate(testloader):
            bag_feats = feats.cuda()
            bag_label = label.cuda()
            bag_prediction = milnet(bag_feats)  #b*class
            bag_loss = criterion(bag_prediction, bag_label)
            
            loss = bag_loss
            total_loss = total_loss + loss.item()

            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (batch_id, len(testloader), loss.item()))
            
            # test_labels.extend([label.squeeze().cpu().numpy()])
            test_labels.extend([label.cpu().numpy()])
            test_predictions.extend([torch.sigmoid(bag_prediction).cpu().numpy()])
    
    test_labels = np.vstack(test_labels)
    test_predictions = np.vstack(test_predictions)
    test_predictions_prob = np.exp(test_predictions)/np.sum(np.exp(test_predictions),axis=1,keepdims=True)
    test_predictions = np.argmax(test_predictions,axis=1)

    test_labels = np.argmax(test_labels,axis=1)
    """
    print(test_labels)
    print(test_predictions)
    
    mcm = multilabel_confusion_matrix(test_labels, test_predictions) # Compute separate 2×2 matrix per class 
    for i, cm in enumerate(mcm):
        print(f"Class {i} confusion matrix:")
        print(cm)
    
    cm = confusion_matrix(test_labels, test_predictions)
    print(cm)
    """

    avg_score = accuracy_score(test_labels,test_predictions)
    balanced_avg_score = balanced_accuracy_score(test_labels,test_predictions)

    f1_marco = f1_score(test_labels,test_predictions,average='macro')
    f1_micro = f1_score(test_labels,test_predictions,average='micro')
    
    p_marco = precision_score(test_labels,test_predictions,average='macro')
    p_micro = precision_score(test_labels,test_predictions,average='micro')
    
    r_marco = recall_score(test_labels,test_predictions,average='macro')
    r_micro = recall_score(test_labels,test_predictions,average='micro')
    
    r_marco = recall_score(test_labels,test_predictions,average='macro')
    r_micro = recall_score(test_labels,test_predictions,average='micro')
    """
    if args.num_classes ==2:
        roc_auc_ovo_marco = roc_auc_score(test_labels,test_predictions_prob[:,1],average='macro')
        roc_auc_ovo_micro = 0.# roc_auc_score(test_labels,test_predictions_prob,average='micro',multi_class='ovo')
    
        roc_auc_ovr_marco = roc_auc_score(test_labels,test_predictions_prob[:,1],average='macro')
        roc_auc_ovr_micro = 0.# roc_auc_score(test_labels,test_predictions_prob,average='micro',multi_class='ovr')
    
    else:
        roc_auc_ovo_marco = roc_auc_score(test_labels,test_predictions_prob,average='macro',multi_class='ovo')
        roc_auc_ovo_micro = 0.# roc_auc_score(test_labels,test_predictions_prob,average='micro',multi_class='ovo')

        roc_auc_ovr_marco = roc_auc_score(test_labels,test_predictions_prob,average='macro',multi_class='ovr')
        roc_auc_ovr_micro = 0.# 
    """
    roc_auc_ovo_marco = 0
    roc_auc_ovr_marco = 0
    #results = [avg_score,balanced_avg_score,f1_marco,f1_micro, p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco,roc_auc_ovo_micro,roc_auc_ovr_marco,roc_auc_ovr_micro]
    results = [avg_score,balanced_avg_score,f1_marco,f1_micro, p_marco,p_micro,r_marco,r_micro, roc_auc_ovo_marco, roc_auc_ovr_marco]
    return total_loss / len(testloader), results



def main():
    parser = argparse.ArgumentParser(description='time classification by TimeMIL')
    parser.add_argument('--dataset', default="mocap", type=str, help='dataset ')
    parser.add_argument('--data_path', default="/home/rguo_hpc/myfolder/code/mocap/data/mocap")
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers used in dataloader [4]')
    # Doesn't matter for skeletonMAE since we load the pre-extracted features
    parser.add_argument('--feats_size', default=0, type=int, help='Dimension of the feature size [512] resnet-50 1024') 
    parser.add_argument('--lr', default=5e-4, type=float, help='1e-3 Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay 1e-4]')
    parser.add_argument('--dropout_patch', default=0.5, type=float, help='Patch dropout rate [0] 0.5')
    parser.add_argument('--dropout_node', default=0.2, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--seed', default='0', type=int, help='random seed')
   
    parser.add_argument('--optimizer', default='adamw', type=str, help='adamw sgd')
    parser.add_argument('--save_dir', default='./savemodel/', type=str, help='the directory used to save all the output')
    parser.add_argument('--epoch_des', default=5, type=int, help='turn on warmup')
    parser.add_argument('--embed', default=192, type=int, help='Number of embedding')
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')

    parser.add_argument('--if_interval', default=False, type=str2bool, help='if split the whole time series to intervals, each interval as an instance')
    parser.add_argument('--instance_len', default=30, type=int, help='the length of instance')
    parser.add_argument('--if_extract_feature', default=True, type=str2bool, help='if extract feature')
    
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
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
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.0)#0.01
    criterion = nn.BCEWithLogitsLoss() # one-vs-rest binary MIL
    # scaler = GradScaler()

    if args.dataset in ['JapaneseVowels','SpokenArabicDigits','CharacterTrajectories','InsectWingbeat']:
        trainset = loadorean(args, split='train')
        testset = loadorean(args, split='test')
        seq_len, num_classes, L_in = trainset.max_len, trainset.num_class, trainset.feat_in
        print(f'max lenght {seq_len}')
        args.feats_size = L_in
        args.num_classes =  num_classes
        print(f'num class:{args.num_classes}')
    
    elif args.dataset in ["mabe_mice", "mocap"]:
        # load embedding
        if args.dataset == "mabe_mice":
        #Skeleton MAE
            X = np.load("/home/rguo_hpc/myfolder/code/pipeline/pretrain/outputs/representations/mae_representations.npy")[1600:] #(1600, 1800, 128)
            y = load_pickle('/home/rguo_hpc/myfolder/code/mocap/data/mabe_mice/mouse_test_labels.pkl')["strain"] #(3736,)
    
        elif args.dataset == "mocap":
            """
            # hbehave/MAE style
            mouse_X = np.load("/home/rguo_hpc/myfolder/code/mocap/outputs/mocap/experiment1/test_submission_0.npy", allow_pickle=True).item()
            X = []
            for mouse_name, indices in mouse_X["frame_number_map"].items():
                X.append(mouse_X['embeddings'][indices[0]:indices[1]]) #(13, 1800)
            X = np.stack(X)
            """
            # Skeleton MAE style
            #X = np.load("/home/rguo_hpc/myfolder/code/mocap/outputs/representations/mae_mocap.npy", allow_pickle=True)
            #X = X.reshape(202, 1200, -1)
            
            fold_1 = {
                "CP1A": {"train": ["M14", "M15", "M19"], 
                        "valid": ["M1"]},
                "CP1B": {"train": ["M2", "M3", "M4", "M5", "M6"], 
                        "valid": ["M1"]},
                "INH1": {"train": ["M2", "M3", "M4", "M5", "M7", "M8", "M9", "M10"],
                        "valid": ["M1", "M6"]},
                "INH2": {"train": ["M2", "M3", "M4", "M5", "M7", "M8", "M9", "M10", "M12"],
                        "valid": ["M1", "M6", "M11"]},
                "MOS1aD": {"train": ["M5", "M6", "M8", "M9", "M10"],
                        "valid": ["M4"]}
            }
            fold_2 = {
                "CP1A": {"train": ["M1", "M15", "M19"], 
                        "valid": ["M14"]},
                "CP1B": {"train": ["M1", "M3", "M4", "M5", "M6"], 
                        "valid": ["M2"]},
                "INH1": {"train": ["M1", "M3", "M4", "M5", "M6", "M8", "M9", "M10"],
                        "valid": ["M2", "M7"]},
                "INH2": {"train": ["M1", "M3", "M4", "M5", "M6", "M8", "M9", "M10", "M11"],
                        "valid": ["M2", "M7", "M12"]},
                "MOS1aD": {"train": ["M4", "M6", "M8", "M9", "M10"],
                            "valid": ["M5"]}
            }
            fold_3 = {
                "CP1A": {"train": ["M1", "M14", "M19"], 
                        "valid": ["M15"]},
                "CP1B": {"train": ["M1", "M2", "M4", "M5", "M6"], 
                        "valid": ["M3"]},
                "INH1": {"train": ["M1", "M2", "M4", "M5", "M6", "M7", "M9", "M10"],
                        "valid": ["M3", "M8"]},
                "INH2": {"train": ["M1", "M2", "M4", "M5", "M6", "M7", "M9", "M11", "M12"],
                        "valid": ["M3", "M8", "M10"]},
                "MOS1aD": {"train": ["M4", "M5", "M8", "M9", "M10"],
                            "valid": ["M6"]}
            }
            fold_4 = {
                "CP1A": {"train": ["M1", "M14", "M15"], 
                        "valid": ["M19"]},
                "CP1B": {"train": ["M1", "M2", "M3", "M5", "M6"], 
                        "valid": ["M4"]},
                "INH1": {"train": ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M10"],
                        "valid": ["M4", "M9"]},
                "INH2": {"train": ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M10", "M12"],
                        "valid": ["M4", "M9", "M11"]},
                "MOS1aD": {"train": ["M4", "M5", "M6", "M9", "M10"],
                        "valid": ["M8"]}
            }


            Xtr = np.load("/home/rguo_hpc/myfolder/mocap/outputs/representations/mae_mocap_tr.npy", allow_pickle=True)
            Xte = np.load("/home/rguo_hpc/myfolder/mocap/outputs/representations/mae_mocap_val.npy", allow_pickle=True)
            with open("/home/rguo_hpc/myfolder/mocap/data/mocap/data_FL2.pkl", 'rb') as file:
                data = pickle.load(file)
            drug_tr = []
            drug_te = []
            for dataset_name in ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"]:
                for mouse_name in fold_1[dataset_name]["train"]:
                    drug_tr = drug_tr + data[dataset_name][mouse_name]["drug"]
                for mouse_name in fold_1[dataset_name]["valid"]:
                    drug_te = drug_te + data[dataset_name][mouse_name]["drug"]
            mapping = {s: i for i, s in enumerate(set(drug_tr))}
            
            ytr = [mapping[s] for s in drug_tr]
            yte = [mapping[s] for s in drug_te]
            """                    
            drug = []
            concentration = []
            for dataset_name in ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"]:
                for mouse_name in data[dataset_name].keys():
                    drug = drug + data[dataset_name][mouse_name]["drug"]
                    concentration = concentration + data[dataset_name][mouse_name]["concentration"]
            mapping = {s: i for i, s in enumerate(set(drug))}
            y = [mapping[s] for s in drug]"""

        #from sklearn.model_selection import train_test_split
        #Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        
        
        Xtr = torch.from_numpy(Xtr)#.permute(0,2,1).float() #(2802, 128, 1800) -> (2802, 1800, 128)
        Xte = torch.from_numpy(Xte)#.permute(0,2,1).float()  
        ytr = F.one_hot(torch.tensor(ytr)).float()
        yte = F.one_hot(torch.tensor(yte)).float()

        trainset = TensorDataset(Xtr,ytr)
        testset = TensorDataset(Xte, yte)

        args.feats_size = Xtr.shape[-1]
        L_in = Xtr.shape[-1]
        num_classes = len(set(drug_tr))
        args.num_classes =  len(set(drug_tr))
        print(f'num class:{args.num_classes}' )
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
        print(f'num class:{args.num_classes}' )
        seq_len = Xte.shape[1]
    else:
        Xtr, ytr, meta = load_classification(name=args.dataset, split='train', return_metadata=True)
        word_to_idx = {}
        for i in range(len(meta['class_values'])):
            word_to_idx[meta['class_values'][i]]=i # {'n': 0, 's': 1, 't': 2}
        Xtr = torch.from_numpy(Xtr).permute(0,2,1).float() # -> [15, 640, 2]
        ytr = [word_to_idx[i] for i in ytr]
        ytr =  F.one_hot(torch.tensor(ytr)).float()        # [15, 3]
        trainset = TensorDataset(Xtr,ytr)
        Xte, yte = load_classification(name=args.dataset, split='test')
        Xte = torch.from_numpy(Xte).permute(0,2,1).float()
        yte = [word_to_idx[i] for i in yte]
        yte = F.one_hot(torch.tensor(yte)).float()
        testset = TensorDataset(Xte,yte)

        args.feats_size = Xte.shape[-1]
        L_in = Xte.shape[-1]
        num_classes = yte.shape[-1]
        args.num_classes =  yte.shape[-1]
        seq_len=  max(21, Xte.shape[1])
        print(f'num class:{args.num_classes}' )
    """    
    
    # <------------- define MIL network ------------->
    milnet = TimeMIL(in_features=args.feats_size, mDim=args.embed, n_classes=num_classes, dropout=args.dropout_node, max_seq_len=seq_len, 
                     if_extract_feature=args.if_extract_feature, if_interval=args.if_interval, instance_len=args.instance_len).cuda()
    
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
    # if args.batchsize==1:
    #     testloader = DataLoader(testset, args.batchsize, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)
    # else:
    testloader = DataLoader(testset, 128, shuffle=False, num_workers=args.num_workers, drop_last=False, pin_memory=True)

    
    best_score = 0
    save_path = join(args.save_dir, 'weights')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(join(args.save_dir,'lesion'), exist_ok=True)
    results_best = None
    
    for epoch in range(1, args.num_epochs + 1):
        writer = SummaryWriter(log_dir=args.save_dir+'/logs/')
        train_loss_bag = train(trainloader, milnet, criterion, optimizer, epoch, args) # iterate all bags
        test_loss_bag, results= test(testloader, milnet, criterion, args)

        writer.add_scalar("Loss/Train", train_loss_bag, epoch)
        writer.add_scalar("Loss/Test", test_loss_bag, epoch)


        [avg_score,balanced_avg_score,f1_marco,f1_micro,p_marco,p_micro,r_marco,r_micro, roc_auc_ovo_marco, roc_auc_ovr_marco] = results
        writer.add_scalar("Loss/Accuracy", avg_score, epoch)
        
        logger.info('\r Epoch [%d/%d] train loss: %.4f test loss: %.4f, accuracy: %.4f, bal. average score: %.4f, f1 marco: %.4f   f1 mirco: %.4f  p marco: %.4f   p mirco: %.4f r marco: %.4f   r mirco: %.4f  roc_auc ovo marco: %.4f   roc_auc ovr marco: %.4f ' % 
                  (epoch, args.num_epochs, train_loss_bag, test_loss_bag, avg_score, balanced_avg_score,f1_marco,f1_micro, p_marco, p_micro, r_marco, r_micro, roc_auc_ovo_marco, roc_auc_ovr_marco )) 
        
        
        # scheduler.step()
        # current_score = (sum(aucs) + avg_score)/3
        current_score = avg_score
        if current_score >= best_score:
            
            results_best = results
            best_score = current_score
            print(current_score)
            save_name = os.path.join(save_path, 'best_model.pth')
            torch.save(milnet.state_dict(), save_name)
            #torch.save(milnet, save_name)
            logger.info('Best model saved at: ' + save_name)
            # logger.info('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
    
    writer.close()

    [avg_score, balanced_avg_score, f1_marco, f1_micro, p_marco, p_micro, r_marco, r_micro,
        roc_auc_ovo_marco, roc_auc_ovr_marco] = results_best
    logger.info('\r Best  Results: accuracy: %.4f, bal. average score: %.4f, f1 marco: %.4f   f1 mirco: %.4f  p marco: %.4f   p mirco: %.4f r marco: %.4f   r mirco: %.4f  roc_auc ovo marco: %.4f  roc_auc ovr marco: %.4f' % 
                  ( avg_score,balanced_avg_score,f1_marco,f1_micro, p_marco,p_micro,r_marco,r_micro,roc_auc_ovo_marco, roc_auc_ovr_marco )) 
    
        # if args.weight_div>0:
        #     if epoch%10==0:
        #         print('--------------------Clustering--------------------\n')
        #         cluster_idx_dict = pre_cluter(trainloader, milnet, criterion, optimizer, args,init= False)
        #         print('--------------------Clustering finished--------------------\n')

if __name__ == '__main__':
    main()