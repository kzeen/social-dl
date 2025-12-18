import pandas as pd

path = "train_clean.jsonl"
path_test = "test_clean.jsonl"

df = pd.read_json(path, lines=True)
df_test = pd.read_json(path_test, lines=True)

TARGET_COL = "label"   # <- change if it has another name

cols_to_drop = [
    "text_clean",
    "user.description",
    "user.location",
    "created_at_dt",
    "in_reply_to_screen_name",
    TARGET_COL,          # we DROP the label from X features
]

# ---- TRAIN (df) ----
meta_cols_train = [c for c in df.columns if c not in cols_to_drop]
df_xgb = df[meta_cols_train].copy()

# ---- TEST (df_t) ----
# Use the *same* columns for test (intersection with df_t just in case)
meta_cols_test = [c for c in meta_cols_train if c in df_test.columns]
df_t_xgb = df_test[meta_cols_test].copy()

# Target
y_all = df[TARGET_COL].values

import pandas as pd

cat_cols = ["source", "user.translator_type", "quoted_status.lang"]

for col in cat_cols:
    if col in df_xgb.columns and col in df_t_xgb.columns:
        # 1) Build union of categories on train + test
        all_cats = pd.concat([df_xgb[col], df_t_xgb[col]], axis=0).astype("category").cat.categories
        
        # 2) Shared CategoricalDtype
        cat_type = pd.api.types.CategoricalDtype(categories=all_cats)
        
        # 3) Apply to BOTH, then take codes
        df_xgb[col]   = df_xgb[col].astype(cat_type).cat.codes
        df_t_xgb[col] = df_t_xgb[col].astype(cat_type).cat.codes


LABEL_COL = "label"
TEXT_COLS = ["text_clean", "user.description", "user.location"]

# --- create text_all in BOTH train and test ---
for d in (df, df_test):
    d["text_all"] = (
        d["text_clean"].fillna("") + " [DESC] " +
        d["user.description"].fillna("") + " [LOC] " +
        d["user.location"].fillna("")
    ).str.strip()

# Train set
texts_train = df["text_all"].astype(str).values
y_all       = df[LABEL_COL].astype(int).values

# Test set
texts_test  = df_test["text_all"].astype(str).values
import numpy as np
from sklearn.model_selection import StratifiedKFold

# ----- Basic stuff -----
N_SPLITS = 5  # or 10 if you want, but 5 is standard
RANDOM_STATE = 42

X_meta_all  = df_xgb.values        # already defined
X_meta_test = df_t_xgb.values

X_text_all  = texts_train          # already defined
X_text_test = texts_test

y_all = df[TARGET_COL].astype(int).values

# ----- CV splitter -----
skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# ----- OOF containers -----
# For XGBoost (one prob per sample, binary classification)
oof_xgb = np.zeros(len(df), dtype=float)
test_pred_xgb = np.zeros((len(df_test), N_SPLITS), dtype=float)

# For CamemBERT (we’ll also store probabilities)
oof_cam = np.zeros(len(df), dtype=float)
test_pred_cam = np.zeros((len(df_test), N_SPLITS), dtype=float)

# Later: these two will be stacked as features for the MLP:
# X_meta_level2 = np.column_stack([oof_xgb, oof_cam])


import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

def modelfit(alg, X_train, y_train, X_valid, y_valid,
             useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    """
    Fit an XGBoost model with optional CV to tune n_estimators,
    then train on train set and evaluate on train + valid.
    Also plots CV AUC (if useTrainCV) and feature importance.
    """
    # ---------- STEP 1: CV with DMatrix to pick best n_estimators ----------
    if useTrainCV:
        print("Starting cross-validation with early stopping...")
        xgb_param = alg.get_xgb_params()
        
        # DMatrix for CV
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        cvresult = xgb.cv(
            params=xgb_param,
            dtrain=dtrain,
            num_boost_round=alg.get_params()['n_estimators'],
            nfold=cv_folds,
            metrics='auc',
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True
        )
        
        # set best n_estimators
        best_n = cvresult.shape[0]
        alg.set_params(n_estimators=best_n)
        print(f"\nBest number of trees (n_estimators): {best_n}")
        
        # ---- Plot CV AUC over boosting rounds ----
        plt.figure(figsize=(7,4))
        plt.plot(cvresult['train-auc-mean'], label='train AUC')
        plt.plot(cvresult['test-auc-mean'], label='valid AUC')
        plt.xlabel("Boosting round")
        plt.ylabel("AUC")
        plt.title("XGBoost CV AUC")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # ---------- STEP 2: Fit on training set, evaluate on train + valid ----------
    print("\nFitting final model on training data...")
    alg.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        verbose=True
        # no eval_metric here → already set in constructor
        # you *could* add early_stopping_rounds=50 if your version supports it
    )
    
    # Predictions
    y_train_pred = alg.predict(X_train)
    y_train_proba = alg.predict_proba(X_train)[:, 1]
    
    y_valid_pred = alg.predict(X_valid)
    y_valid_proba = alg.predict_proba(X_valid)[:, 1]
    
    # Metrics
    print("\n=== Model Report (TRAIN) ===")
    print(f"Accuracy : {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"AUC      : {roc_auc_score(y_train, y_train_proba):.4f}")
    
    print("\n=== Model Report (VALID) ===")
    print(f"Accuracy : {accuracy_score(y_valid, y_valid_pred):.4f}")
    print(f"AUC      : {roc_auc_score(y_valid, y_valid_proba):.4f}")
    
    # ---------- STEP 3: Plot feature importance ----------
    booster = alg.get_booster()
    # importance_type can be: 'weight', 'gain', 'cover', etc.
    fmap = booster.get_score(importance_type='gain')
    
    if len(fmap) == 0:
        print("\nNo feature importance found (check model / training).")
        return alg
    
    # sort and keep top 20
    importance_df = pd.DataFrame(
        list(fmap.items()), columns=['feature', 'importance']
    ).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(8, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.gca().invert_yaxis()
    plt.xlabel("Gain")
    plt.title("Top 20 Feature Importances (gain)")
    plt.tight_layout()
    plt.show()
    
    return alg


from xgboost import XGBClassifier
best_max_depth = 10
best_min_child_weight = 1
best_gamma = 0

xgb_base_params = dict(
    learning_rate=0.05,
    n_estimators=1000,                # upper bound; CV will pick best
    max_depth=best_max_depth,
    min_child_weight=best_min_child_weight,
    gamma=best_gamma,
    subsample=1.0,
    reg_alpha=0.1,
    reg_lambda=0.1,
    colsample_bytree=0.6,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    device="cuda",                    # or remove if no GPU
)


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizerFast, CamembertForSequenceClassification
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")


class TextDataset(Dataset):
    def __init__(self, texts, labels=None, max_len=128):
        self.texts = list(texts)
        self.labels = None if labels is None else np.array(labels, dtype=np.int64)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_camembert_one_fold(
    texts_tr, y_tr,
    texts_val, y_val,
    texts_test,
    epochs=2,
    batch_size=16,
    lr=2e-5,
    max_len=256
):
    # Datasets
    train_ds = TextDataset(texts_tr, y_tr, max_len=max_len)
    val_ds   = TextDataset(texts_val, y_val, max_len=max_len)
    test_ds  = TextDataset(texts_test, labels=None, max_len=max_len)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Fresh model for this fold
    model = CamembertForSequenceClassification.from_pretrained(
        "camembert-base",
        num_labels=2
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_auc = 0.0
    best_state_dict = None

    # ----- Training loop -----
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{epochs}] Train loss: {avg_train_loss:.4f}")

        # ----- Validation -----
        model.eval()
        val_probs = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1]  # prob of class 1

                val_probs.append(probs.cpu().numpy())
                val_true.append(labels.cpu().numpy())

        val_probs = np.concatenate(val_probs)
        val_true = np.concatenate(val_true)

        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(val_true, val_probs)
        print(f"[Epoch {epoch+1}/{epochs}] Val AUC: {val_auc:.4f}")

        # Keep best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    print("Best Val AUC for this fold:", best_val_auc)

    # ----- Reload best model on device -----
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    # ----- Final val predictions with best model -----
    model.eval()
    val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            val_probs.append(probs.cpu().numpy())
    val_probs = np.concatenate(val_probs)

    # ----- Test predictions -----
    test_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            test_probs.append(probs.cpu().numpy())
    test_probs = np.concatenate(test_probs)

    return val_probs, test_probs


for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta_all, y_all)):
    print(f"\n===== FOLD {fold + 1} / {N_SPLITS} =====")

    # ---- Split metadata ----
    X_meta_tr, X_meta_val = X_meta_all[train_idx], X_meta_all[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]

    # ---- Split text ----
    X_text_tr = X_text_all[train_idx]
    X_text_val = X_text_all[val_idx]

    # Here we will:
    # 1) Train XGBoost on (X_meta_tr, y_tr)
    

    # 2) Predict on X_meta_val  -> fill oof_xgb[val_idx]
    # 3) Predict on X_meta_test -> store in test_pred_xgb[:, fold]
    #
    # 4) Train CamemBERT on (X_text_tr, y_tr)
    # 5) Predict on X_text_val  -> fill oof_cam[val_idx]
    # 6) Predict on X_text_test -> store in test_pred_cam[:, fold]
from sklearn.metrics import roc_auc_score
import numpy as np

for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta_all, y_all)):
    print(f"\n========== FOLD {fold + 1} / {N_SPLITS} ==========")

    # Split metadata
    X_tr, X_val = X_meta_all[train_idx], X_meta_all[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]

    # Fresh model for this fold
    xgb_model = XGBClassifier(**xgb_base_params)

    # Use your modelfit to tune n_estimators on this fold + train
    xgb_model = modelfit(
        xgb_model,
        X_train=X_tr,
        y_train=y_tr,
        X_valid=X_val,
        y_valid=y_val,
        useTrainCV=True,
        cv_folds=5,
        early_stopping_rounds=50,
    )

    # --- OOF for this fold ---
    val_proba = xgb_model.predict_proba(X_val)[:, 1]
    oof_xgb[val_idx] = val_proba

    # --- Test preds for this fold ---
    test_proba = xgb_model.predict_proba(X_meta_test)[:, 1]
    test_pred_xgb[:, fold] = test_proba

    

# After all folds: global OOF metric
print("\n===== XGBoost OOF performance =====")
print("OOF AUC:", roc_auc_score(y_all, oof_xgb))

# This is what you'll use later for stacking:
xgb_test_mean = test_pred_xgb.mean(axis=1)  # fold-averaged test proba

oof_cam = np.zeros(len(df), dtype=float)
test_pred_cam = np.zeros((len(df_test), N_SPLITS), dtype=float)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_text_all, y_all)):
    print(f"\n========== CamemBERT FOLD {fold + 1} / {N_SPLITS} ==========")

    texts_tr = X_text_all[train_idx]
    texts_val = X_text_all[val_idx]
    y_tr = y_all[train_idx]
    y_val = y_all[val_idx]

    # Train + get preds for this fold
    val_probs, test_probs = train_camembert_one_fold(
        texts_tr, y_tr,
        texts_val, y_val,
        X_text_test,
        epochs=2,         # you can increase later
        batch_size=16,
        lr=2e-5,
        max_len=128
    )

    oof_cam[val_idx] = val_probs
    test_pred_cam[:, fold] = test_probs

from sklearn.metrics import roc_auc_score
print("\n===== CamemBERT OOF performance =====")
print("OOF AUC:", roc_auc_score(y_all, oof_cam))

cam_test_mean = test_pred_cam.mean(axis=1)  # mean over folds


import os
import numpy as np
import pandas as pd  # you already have this

out_dir = "oof_outputs"
os.makedirs(out_dir, exist_ok=True)

# Mean over folds for test
xgb_test_mean = test_pred_xgb.mean(axis=1)
cam_test_mean = test_pred_cam.mean(axis=1)

# ------- 1) Save as .npy (fast & safe) -------
np.save(os.path.join(out_dir, "oof_xgb.npy"), oof_xgb)
np.save(os.path.join(out_dir, "test_pred_xgb.npy"), test_pred_xgb)

np.save(os.path.join(out_dir, "oof_cam.npy"), oof_cam)
np.save(os.path.join(out_dir, "test_pred_cam.npy"), test_pred_cam)

np.save(os.path.join(out_dir, "xgb_test_mean.npy"), xgb_test_mean)
np.save(os.path.join(out_dir, "cam_test_mean.npy"), cam_test_mean)

print("Saved .npy files in", out_dir)

ID_COL = "ID"  # change if your id column is named differently

# OOF per train row
df_oof = pd.DataFrame({
    ID_COL: df[ID_COL].values,
    "label": y_all,
    "oof_xgb": oof_xgb,
    "oof_cam": oof_cam,
})
df_oof.to_csv(os.path.join(out_dir, "oof_train.csv"), index=False)

# Test preds per test row (mean over folds)
df_test_preds = pd.DataFrame({
    ID_COL: df_test[ID_COL].values,
    "xgb_mean": xgb_test_mean,
    "cam_mean": cam_test_mean,
})
df_test_preds.to_csv(os.path.join(out_dir, "test_preds.csv"), index=False)

print("Saved CSVs in", out_dir)
