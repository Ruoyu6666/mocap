import pickle
import yaml
import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


parser = argparse.ArgumentParser(description='time classification by TimeMIL')
parser.add_argument('--dataset', default="mabe_mice", type=str, help='dataset ')
parser.add_argument('--label', default="drug", type=str, help='dataset ')
parser.add_argument('--subseq_len', default=3000, type=int, help='the length of sub sequence')
parser.add_argument('--path_to_data_dir', default="/home/rguo_hpc/myfolder/data/sdannce/data_fmr1.pkl", type=str)
parser.add_argument("--config", type=str, default = '../configs/sdannce/fmr1.yaml')



args = parser.parse_args()

if "sdannce" in args.dataset:
    if args.dataset == "sdannce":
        Xtr = np.load("/home/rguo_hpc/myfolder/mocap/outputs/sdannce/fmr1/50/representations/mae_sdannce_tr.npy")[:, 25:3025]
        Xte = np.load("/home/rguo_hpc/myfolder/mocap/outputs/sdannce/fmr1/50/representations/mae_sdannce_val.npy")[:, 25:3025]
        print(Xtr.shape, Xte.shape)
    if args.dataset == "sdannce_kinematic":
        Xtr = np.load("/home/rguo_hpc/myfolder/mocap/swav/kinematic/data/kinematic_tr.py.npy")
        Xte = np.load("/home/rguo_hpc/myfolder/mocap/swav/kinematic/data/kinematic_val.py.npy")
        Xtr = Xtr.reshape(-1, args.subseq_len, Xtr.shape[-1])
        Xte = Xte.reshape(-1, args.subseq_len, Xtr.shape[-1])

    with open(args.path_to_data_dir, 'rb') as file:
        data = pickle.load(file)
    with open(args.config) as f:
        split = yaml.safe_load(f)
    
    N = int(90000 / args.subseq_len) # number of sequences per sequence
    label_tr = []
    label_te = []
    for mouse in split["fold_1"]["train"]:
        num_seq = len(data[mouse]["ratid"])
        gen  = str(data[mouse]["ratgen"])
        label_tr = label_tr + [gen for i in range(num_seq * N)]
    for mouse in split["fold_1"]["valid"]:
        num_seq = len(data[mouse]["ratid"])
        gen = str(data[mouse]["ratgen"])
        label_te = label_te + [gen for i in range(num_seq * N)]
    mapping = {s: i for i, s in enumerate(set(label_tr))}
    print(f"Mapping of ratgen labels to integers: {mapping}")
    ytr = [mapping[s] for s in label_tr]
    yte = [mapping[s] for s in label_te]
    y_train = np.array(ytr)
    y_test = np.array(yte)


elif args.dataset == "mabe_mice":
    X = np.load("/home/rguo_hpc/myfolder/mocap/outputs/mabe/features_pca.npy", allow_pickle=True).item()["embeddings"]
    X = X.reshape(3736, -1 ,128)  # Reshape to (N, C)
    test_labels = np.load("/home/rguo_hpc/myfolder/data/mabe_mice/mouse_triplets_test_labels.npy", allow_pickle=True).item()
    y = test_labels["label_array"][-1][::1800]  # strain/lights
    
    indices = np.arange(X.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=2, shuffle=True)
    Xtr, Xte = X[train_idx], X[test_idx][::2]
    y_train, y_test = y[train_idx], y[test_idx][::2]





elif args.dataset == "mocap":
    Xtr = np.load("/home/rguo_hpc/myfolder/mocap/outputs/representations/CLB/mae_mocap_tr_f1.npy", allow_pickle=True)
    Xte = np.load("/home/rguo_hpc/myfolder/mocap/outputs/representations/CLB/mae_mocap_val_f1.npy", allow_pickle=True)

    with open("/home/rguo_hpc/myfolder/data/mocap/data_CLB.pkl", 'rb') as file:
        data = pickle.load(file)
    if args.label == "drug":
        drug_tr = []
        drug_te = []
        for dataset_name in ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"]:
            for mouse_name in mocap_fold_2[dataset_name]["train"]:
                drug_tr = drug_tr + data[dataset_name][mouse_name]["drug"]
            for mouse_name in mocap_fold_2[dataset_name]["valid"]:
                drug_te = drug_te + data[dataset_name][mouse_name]["drug"]
        mapping = {s: i for i, s in enumerate(set(drug_tr))}
        ytr = [mapping[s] for s in drug_tr]
        yte = [mapping[s] for s in drug_te]
        y_train = np.array(ytr)
        y_test = np.array(yte)
            
    if args.label == "experiment":
        expe_tr = []
        expe_te = []
        for dataset_name in ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"]:
            for mouse_name in mocap_fold_1[dataset_name]["train"]:
                expe_tr = expe_tr + [dataset_name]* len(data[dataset_name][mouse_name]["drug"])
            for mouse_name in mocap_fold_1[dataset_name]["valid"]:
                expe_te = expe_te + [dataset_name]* len(data[dataset_name][mouse_name]["drug"])
            mapping = {s: i for i, s in enumerate(set(expe_tr))}
            print(mapping)
            ytr = [mapping[s] for s in expe_tr]
            yte = [mapping[s] for s in expe_te]
            y_train = np.array(ytr)
            y_test = np.array(yte)

    if args.label == "type":
        type_tr = []
        type_te = []
        for dataset_name in ["CP1A", "CP1B", "INH1", "INH2", "MOS1aD"]:
            for mouse_name in mocap_fold_1[dataset_name]["train"]:
                    type_tr = type_tr + data[dataset_name][mouse_name]["type"]
            for mouse_name in mocap_fold_1[dataset_name]["valid"]:
                    type_te = type_te + data[dataset_name][mouse_name]["type"]
        mapping = {s: i for i, s in enumerate(set(type_tr))}
        print(mapping)
        ytr = [mapping[s] for s in type_tr]
        yte = [mapping[s] for s in type_te]
        y_train = np.array(ytr)
        y_test = np.array(yte)



# ---- Step 1: Reduce time dimension ----
# Mean pooling across timestamps
X_train_reduced = Xtr.mean(axis=1)   # shape: (150, 192)
X_test_reduced = Xte.mean(axis=1)     # shape: (50, 192)
print(X_train_reduced.shape)
print(X_test_reduced.shape)
print(y_train.shape)
print(y_test.shape)

# ---- Step 2: Train simple regression model ----
model = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)

model.fit(X_train_reduced, y_train)

# ---- Step 3: Predict ----
y_pred = model.predict(X_test_reduced)

# ---- Step 4: Evaluate ----
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))