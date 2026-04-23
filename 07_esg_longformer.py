# 07_esg_longformer.py
# AIGB 7290 — ESG Litigation Classifier
# Phase 6: Longformer Fine-Tuning
# Huang, Kiernan, Sooknanan | Fordham University
#
# Requires Google Colab GPU runtime (T4 or A100, 16GB VRAM).
# Colab install:
#   !pip install transformers accelerate datasets scikit-learn -q
#
# Architecture choice is a direct consequence of the token audit finding:
# 99.1% of the 444 filtered cases exceed the 512-token limit of standard
# BERT-family models (avg 8,824 tokens/case). Longformer-base-4096 with
# global attention on the [CLS] token and a sliding window over remaining
# positions handles the full distribution without truncation loss.

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LongformerTokenizerFast,
    LongformerForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
try:
    from config import ROOT, ESG_CORPUS_OUTPUTS, ESG_CORPUS_LABELS_CSV
except ImportError:
    ROOT = Path("/content/drive/Shared Drives/ESG DL Project/esg_project")
    ESG_CORPUS_OUTPUTS    = ROOT / "esg_corpus_outputs"
    ESG_CORPUS_LABELS_CSV = ESG_CORPUS_OUTPUTS / "esg_corpus_labels.csv"

CLEANED_CSV  = ESG_CORPUS_OUTPUTS / "ESG_corpus_cleaned_v1.csv"
OUTPUTS_DIR  = ESG_CORPUS_OUTPUTS / "longformer"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT = ROOT / "07_manifest.json"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LABEL_ORDER   = ["E", "S", "G", "Non-ESG"]
MODEL_NAME    = "allenai/longformer-base-4096"
MAX_LEN       = 4096
BATCH_SIZE    = 2       # safe for 16GB VRAM at 4096 tokens
GRAD_ACCUM    = 8       # effective batch size = 16
LR            = 2e-5
EPOCHS        = 5
SEEDS         = [42, 123, 7]
FREEZE_LAYERS = 8       # bottom 8 encoder layers frozen to prevent catastrophic forgetting

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("\nLoading data...")
df = pd.read_csv(CLEANED_CSV)
df = df[df["label"].isin(LABEL_ORDER)].copy()
df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)
print(f"  Cases: {len(df)}")
print(df["label"].value_counts().to_string())

le = LabelEncoder()
le.fit(LABEL_ORDER)
y     = le.transform(df["label"].values)
texts = df["cleaned_text"].tolist()

# ---------------------------------------------------------------------------
# Class weights — penalize errors on sparse Social and Greenwash labels
# ---------------------------------------------------------------------------
classes  = np.unique(y)
cw_vals  = compute_class_weight("balanced", classes=classes, y=y)
class_weights_tensor = torch.tensor(cw_vals, dtype=torch.float).to(DEVICE)
print(f"\n  Class weights: {dict(zip(LABEL_ORDER, cw_vals.round(3)))}")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = LongformerTokenizerFast.from_pretrained(MODEL_NAME)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ESGDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # Global attention on the [CLS] token (position 0) — attends to all tokens
        global_attention_mask = torch.zeros_like(enc["attention_mask"])
        global_attention_mask[0, 0] = 1

        return {
            "input_ids":             enc["input_ids"].squeeze(0),
            "attention_mask":        enc["attention_mask"].squeeze(0),
            "global_attention_mask": global_attention_mask.squeeze(0),
            "labels":                torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model():
    model = LongformerForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_ORDER),
        ignore_mismatched_sizes=True,
    )
    for i in range(FREEZE_LAYERS):
        for param in model.longformer.encoder.layer[i].parameters():
            param.requires_grad = False
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Params: {total:,} total | {trainable:,} trainable ({100*trainable/total:.1f}%)")
    return model.to(DEVICE)

# ---------------------------------------------------------------------------
# Single seed training (5-fold CV)
# ---------------------------------------------------------------------------
def train_one_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y)):
        print(f"    Fold {fold+1}/5...")

        train_ds = ESGDataset([texts[i] for i in train_idx], y[train_idx])
        val_ds   = ESGDataset([texts[i] for i in val_idx],   y[val_idx])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model     = build_model()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LR, weight_decay=0.01,
        )
        total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
        warmup_steps = int(0.10 * total_steps)
        scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        loss_fn      = nn.CrossEntropyLoss(weight=class_weights_tensor)

        best_f1      = 0.0
        best_path    = OUTPUTS_DIR / f"longformer_s{seed}_f{fold+1}.pt"

        for epoch in range(EPOCHS):
            # ----- Train -----
            model.train()
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                out  = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    global_attention_mask=batch["global_attention_mask"].to(DEVICE),
                )
                loss = loss_fn(out.logits, batch["labels"].to(DEVICE)) / GRAD_ACCUM
                loss.backward()
                if (step + 1) % GRAD_ACCUM == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # ----- Validate -----
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in val_loader:
                    out = model(
                        input_ids=batch["input_ids"].to(DEVICE),
                        attention_mask=batch["attention_mask"].to(DEVICE),
                        global_attention_mask=batch["global_attention_mask"].to(DEVICE),
                    )
                    preds.extend(out.logits.argmax(-1).cpu().tolist())
                    trues.extend(batch["labels"].tolist())

            fold_f1 = f1_score(trues, preds, average="macro")
            print(f"      Epoch {epoch+1}: Val Macro-F1 = {fold_f1:.4f}")
            if fold_f1 > best_f1:
                best_f1 = fold_f1
                torch.save(model.state_dict(), best_path)

        print(f"    → Best fold F1: {best_f1:.4f} (saved: {best_path.name})")
        fold_f1s.append(best_f1)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return fold_f1s

# ---------------------------------------------------------------------------
# Extract document embeddings from best model (seed 42, fold 1)
# Used by 08_esg_xai_visualizations.py for t-SNE and attention heatmaps
# ---------------------------------------------------------------------------
def extract_embeddings(model_path):
    print("\nExtracting document embeddings for visualization...")
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    full_ds     = ESGDataset(texts, y)
    full_loader = DataLoader(full_ds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)
    embeddings  = []

    with torch.no_grad():
        for batch in full_loader:
            out = model.longformer(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                global_attention_mask=batch["global_attention_mask"].to(DEVICE),
            )
            # CLS token representation
            cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)

    embeddings = np.vstack(embeddings)
    np.save(OUTPUTS_DIR / "longformer_embeddings.npy", embeddings)
    np.save(OUTPUTS_DIR / "longformer_labels.npy", y)
    print(f"  Embeddings saved: {OUTPUTS_DIR / 'longformer_embeddings.npy'} — shape {embeddings.shape}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Multi-seed run
# ---------------------------------------------------------------------------
all_results = {}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    fold_f1s   = train_one_seed(seed)
    seed_mean  = float(np.mean(fold_f1s))
    seed_std   = float(np.std(fold_f1s))
    all_results[seed] = {
        "fold_f1s": [round(v, 4) for v in fold_f1s],
        "mean":     round(seed_mean, 4),
        "std":      round(seed_std, 4),
    }
    print(f"Seed {seed} — Macro-F1: {seed_mean:.4f} ± {seed_std:.4f}")

all_f1s     = [v for r in all_results.values() for v in r["fold_f1s"]]
global_mean = float(np.mean(all_f1s))
global_std  = float(np.std(all_f1s))
print(f"\nFinal Longformer Macro-F1 ({len(SEEDS)} seeds × 5 folds): {global_mean:.4f} ± {global_std:.4f}")

# Extract embeddings from seed-42 fold-1 best checkpoint for visualization
best_checkpoint = OUTPUTS_DIR / "longformer_s42_f1.pt"
if best_checkpoint.exists():
    extract_embeddings(best_checkpoint)

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
manifest = {
    "script":               "07_esg_longformer.py",
    "model":                MODEL_NAME,
    "max_len":              MAX_LEN,
    "batch_size":           BATCH_SIZE,
    "grad_accum":           GRAD_ACCUM,
    "effective_batch":      BATCH_SIZE * GRAD_ACCUM,
    "lr":                   LR,
    "epochs":               EPOCHS,
    "seeds":                SEEDS,
    "freeze_layers":        FREEZE_LAYERS,
    "per_seed_results":     {str(k): v for k, v in all_results.items()},
    "global_macro_f1_mean": round(global_mean, 4),
    "global_macro_f1_std":  round(global_std,  4),
    "outputs_dir":          str(OUTPUTS_DIR),
}
with open(MANIFEST_OUT, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest written: {MANIFEST_OUT}")
print("Phase 6 — Longformer fine-tuning complete.")
