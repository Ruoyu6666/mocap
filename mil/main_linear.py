import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



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

Xtr = np.load("/home/rguo_hpc/myfolder/code/mocap/outputs/representations/fold_1/mae_mocap_tr.npy", allow_pickle=True)
Xtr= Xtr.reshape(158, 1200, -1)
Xte = np.load("/home/rguo_hpc/myfolder/code/mocap/outputs/representations/fold_1/mae_mocap_val.npy", allow_pickle=True)
Xte= Xte.reshape(44, 1200, -1)
with open("/home/rguo_hpc/myfolder/code/mocap/data/mocap/data_CLB.pkl", 'rb') as file:
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
y_train = np.array(ytr)
y_test = np.array(yte)


# ---- Step 1: Reduce time dimension ----
# Mean pooling across timestamps
X_train_reduced = Xtr.mean(axis=1)   # shape: (150, 192)
X_test_reduced = Xte.mean(axis=1)     # shape: (50, 192)

# ---- Step 2: Train simple regression model ----
model = LogisticRegression(max_iter=1000, multi_class='multinomial')

model.fit(X_train_reduced, y_train)

# ---- Step 3: Predict ----
y_pred = model.predict(X_test_reduced)

# ---- Step 4: Evaluate ----
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))