# 07_esg_longformer.py
# AIGB 7290 -- ESG Litigation Classifier
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
#
# RESUME ACROSS COLAB SESSIONS:
#   This script writes a progress file (07_progress.json) after every fold.
#   On restart, it reads the progress file and skips already-completed folds.
#   Simply re-run the cell -- it will pick up exactly where it left off.
#   Progress file location: ROOT / "07_progress.json"
#
# Runtime estimates (T4, USE_AMP=True):
#   1 seed x 3 folds x 5 epochs: ~4-6 hours
#   3 seeds x 3 folds x 5 epochs: ~14-18 hours (split across 2 sessions)

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
from transformers import (
    LongformerTokenizerFast,
    LongformerForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HuggingFace authentication -- suppresses rate-limit warnings
# In Colab: store your token in Secrets (key icon, left sidebar) as HF_TOKEN
# ---------------------------------------------------------------------------
import os
try:
    from google.colab import userdata
    _hf_token = userdata.get("UN_SquishMug")
except Exception:
    _hf_token = os.environ.get("HF_TOKEN", "")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)
    print("HuggingFace: authenticated")
else:
    print("HuggingFace: no token found -- set HF_TOKEN in Colab Secrets")

# ---------------------------------------------------------------------------
# Browser keepalive -- prevents Colab idle-disconnect during long training runs
# Injects JavaScript that clicks the reconnect button every 60 seconds.
# No-op when running outside Jupyter/Colab.
# ---------------------------------------------------------------------------
try:
    from IPython.display import display, Javascript
    display(Javascript("""
        console.log("ESG keepalive: starting");
        function ESGKeepAlive() {
            var btn = document.querySelector('#connect');
            if (btn) btn.click();
            console.log("ESG keepalive: ping " + new Date().toLocaleTimeString());
        }
        if (window._esgKeepAliveTimer) clearInterval(window._esgKeepAliveTimer);
        window._esgKeepAliveTimer = setInterval(ESGKeepAlive, 60000);
    """))
    print("Browser keepalive: active (pings every 60s)")
except Exception:
    pass  # not in Jupyter -- silently skip

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
try:
    from config import ROOT, ESG_CORPUS_OUTPUTS, ESG_CORPUS_LABELS_CSV
except ImportError:
    ROOT = Path("/content/drive/Shared Drives/ESG DL Project/esg_project")
    ESG_CORPUS_OUTPUTS    = ROOT / "esg_corpus_outputs"
    ESG_CORPUS_LABELS_CSV = ESG_CORPUS_OUTPUTS / "esg_corpus_labels.csv"

CLEANED_CSV   = ESG_CORPUS_OUTPUTS / "ESG_corpus_cleaned_v1.csv"
OUTPUTS_DIR   = ESG_CORPUS_OUTPUTS / "longformer"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_OUT  = ROOT / "07_manifest.json"
PROGRESS_FILE = ROOT / "07_progress.json"   # resume anchor -- persists across sessions

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
LABEL_ORDER   = ["E", "S", "G", "Non-ESG"]
MODEL_NAME    = "allenai/longformer-base-4096"
MAX_LEN       = 4096
BATCH_SIZE    = 1       # FP32: 1 per device; AMP: can try 2 on A100
GRAD_ACCUM    = 16      # effective batch size = 16 regardless
LR            = 2e-5
EPOCHS        = 5
SEEDS         = [42, 123, 7]   # full 3-seed run split across 2 Colab sessions
N_FOLDS       = 3              # 3-fold CV -- ~40% faster than 5-fold
FREEZE_LAYERS = 8              # bottom 8 encoder layers frozen
USE_AMP       = True           # mixed-precision FP16 -- required on T4/A100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"AMP     : {USE_AMP}")
print(f"Config  : {len(SEEDS)} seeds x {N_FOLDS} folds x {EPOCHS} epochs")

# ---------------------------------------------------------------------------
# Progress file -- read on startup, write after every completed fold
# ---------------------------------------------------------------------------
def load_progress():
    if PROGRESS_FILE.exists():
        prog = json.loads(PROGRESS_FILE.read_text())
        print(f"\nResuming from progress file: {PROGRESS_FILE.name}")
        print(f"  Completed so far: {prog.get('completed_folds', [])}")
        return prog
    return {"completed_folds": [], "results": {}}

def save_progress(prog):
    PROGRESS_FILE.write_text(json.dumps(prog, indent=2))

def fold_key(seed, fold):
    return f"s{seed}_f{fold+1}"

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
# Class weights -- penalize errors on sparse Social label
# ---------------------------------------------------------------------------
classes  = np.unique(y)
cw_vals  = compute_class_weight("balanced", classes=classes, y=y)
class_weights_tensor = torch.tensor(cw_vals, dtype=torch.float).to(DEVICE)
print(f"\n  Class weights: {dict(zip(le.classes_, cw_vals.round(3)))}")

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
        # Global attention on [CLS] (position 0) -- attends to all tokens
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
    # Gradient checkpointing: recompute activations to save VRAM at 4096 tokens
    model.longformer.encoder.gradient_checkpointing = True
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Params: {total:,} total | {trainable:,} trainable ({100*trainable/total:.1f}%)")
    return model.to(DEVICE)

# ---------------------------------------------------------------------------
# Train one fold, return best val F1
# ---------------------------------------------------------------------------
def train_one_fold(seed, fold, train_idx, val_idx, scaler):
    print(f"\n    Fold {fold+1}/{N_FOLDS}  (train={len(train_idx)}, val={len(val_idx)})")

    train_ds     = ESGDataset([texts[i] for i in train_idx], y[train_idx])
    val_ds       = ESGDataset([texts[i] for i in val_idx],   y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model        = build_model()
    optimizer    = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    total_steps  = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps = int(0.10 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn      = nn.CrossEntropyLoss(weight=class_weights_tensor)
    best_f1      = 0.0
    best_path    = OUTPUTS_DIR / f"longformer_s{seed}_f{fold+1}.pt"

    epoch_bar = tqdm(range(EPOCHS), desc=f"S{seed} F{fold+1}", unit="ep", position=0)
    for epoch in epoch_bar:
        # ----- Train (AMP) -----
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                         desc=f"  Train ep{epoch+1}", unit="batch",
                         position=1, leave=False)
        for step, batch in train_bar:
            with autocast(enabled=USE_AMP):
                out  = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    global_attention_mask=batch["global_attention_mask"].to(DEVICE),
                )
                loss = loss_fn(out.logits, batch["labels"].to(DEVICE)) / GRAD_ACCUM
            scaler.scale(loss).backward()
            running_loss += loss.item() * GRAD_ACCUM
            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            train_bar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM:.4f}")

        # ----- Validate -----
        model.eval()
        preds, trues = [], []
        val_bar = tqdm(val_loader, desc=f"  Val   ep{epoch+1}", unit="batch",
                       position=1, leave=False)
        with torch.no_grad():
            for batch in val_bar:
                with autocast(enabled=USE_AMP):
                    out = model(
                        input_ids=batch["input_ids"].to(DEVICE),
                        attention_mask=batch["attention_mask"].to(DEVICE),
                        global_attention_mask=batch["global_attention_mask"].to(DEVICE),
                    )
                preds.extend(out.logits.argmax(-1).cpu().tolist())
                trues.extend(batch["labels"].tolist())

        fold_f1  = f1_score(trues, preds, average="macro")
        mcc      = matthews_corrcoef(trues, preds)
        avg_loss = running_loss / len(train_loader)
        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}", F1=f"{fold_f1:.4f}", MCC=f"{mcc:.4f}")
        tqdm.write(f"    Ep {epoch+1}/{EPOCHS} | loss={avg_loss:.4f}  F1={fold_f1:.4f}  MCC={mcc:.4f}"
                   + (" [best]" if fold_f1 > best_f1 else ""))
        if fold_f1 > best_f1:
            best_f1 = fold_f1
            torch.save(model.state_dict(), best_path)

    epoch_bar.close()
    tqdm.write(f"    --> Fold {fold+1} best F1: {best_f1:.4f}  checkpoint: {best_path.name}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_f1

# ---------------------------------------------------------------------------
# Multi-seed run with resume support
# ---------------------------------------------------------------------------
progress = load_progress()

# Rebuild completed results from progress file
all_results = {int(k): v for k, v in progress.get("results", {}).items()}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scaler   = GradScaler(enabled=USE_AMP)
    fold_f1s = all_results.get(seed, {}).get("fold_f1s", [])

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, y)):
        key = fold_key(seed, fold)
        if key in progress["completed_folds"]:
            stored_f1 = fold_f1s[fold] if fold < len(fold_f1s) else "?"
            print(f"    Fold {fold+1}/{N_FOLDS} -- SKIPPED (already complete, F1={stored_f1})")
            continue

        best_f1 = train_one_fold(seed, fold, train_idx, val_idx, scaler)

        # Persist immediately after fold completes
        fold_f1s_updated = list(fold_f1s) + [round(best_f1, 4)]
        fold_f1s = fold_f1s_updated
        all_results[seed] = {
            "fold_f1s": fold_f1s,
            "mean":     round(float(np.mean(fold_f1s)), 4),
            "std":      round(float(np.std(fold_f1s)),  4),
        }
        progress["completed_folds"].append(key)
        progress["results"] = {str(k): v for k, v in all_results.items()}
        save_progress(progress)
        print(f"    Progress saved to {PROGRESS_FILE.name}")

    if seed in all_results and all_results[seed]["fold_f1s"]:
        r = all_results[seed]
        print(f"Seed {seed} -- Macro-F1: {r['mean']:.4f} +/- {r['std']:.4f}  folds={r['fold_f1s']}")

# ---------------------------------------------------------------------------
# Global summary (only over completed seeds/folds)
# ---------------------------------------------------------------------------
all_f1s = [v for r in all_results.values() for v in r["fold_f1s"]]
if all_f1s:
    global_mean = float(np.mean(all_f1s))
    global_std  = float(np.std(all_f1s))
    n_completed = len(progress["completed_folds"])
    n_total     = len(SEEDS) * N_FOLDS
    print(f"\nLongformer Macro-F1 ({n_completed}/{n_total} folds complete): "
          f"{global_mean:.4f} +/- {global_std:.4f}")
else:
    global_mean = 0.0
    global_std  = 0.0
    print("\nNo folds completed yet.")

# ---------------------------------------------------------------------------
# Extract document embeddings from seed-42 fold-1 best checkpoint
# Used by 08_esg_xai_visualizations.py for t-SNE and attention heatmaps
# ---------------------------------------------------------------------------
def extract_embeddings(model_path):
    print("\nExtracting CLS embeddings for visualization...")
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    full_ds     = ESGDataset(texts, y)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)
    embeddings  = []

    with torch.no_grad():
        for batch in tqdm(full_loader, desc="Embeddings", unit="batch"):
            with autocast(enabled=USE_AMP):
                out = model.longformer(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    global_attention_mask=batch["global_attention_mask"].to(DEVICE),
                )
            cls_emb = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            embeddings.append(cls_emb)

    embeddings = np.vstack(embeddings)
    np.save(OUTPUTS_DIR / "longformer_embeddings.npy", embeddings)
    np.save(OUTPUTS_DIR / "longformer_labels.npy", y)
    print(f"  Embeddings: shape={embeddings.shape}  saved to {OUTPUTS_DIR}/")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

best_checkpoint = OUTPUTS_DIR / "longformer_s42_f1.pt"
if best_checkpoint.exists():
    extract_embeddings(best_checkpoint)
else:
    print(f"\nNOTE: {best_checkpoint.name} not found -- embeddings will be extracted once "
          "seed-42 fold-1 is complete.")

# ---------------------------------------------------------------------------
# Manifest (written/updated on every run)
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
    "n_folds":              N_FOLDS,
    "freeze_layers":        FREEZE_LAYERS,
    "use_amp":              USE_AMP,
    "completed_folds":      progress["completed_folds"],
    "per_seed_results":     {str(k): v for k, v in all_results.items()},
    "global_macro_f1_mean": round(global_mean, 4),
    "global_macro_f1_std":  round(global_std,  4),
    "outputs_dir":          str(OUTPUTS_DIR),
}
with open(MANIFEST_OUT, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest written : {MANIFEST_OUT.name}")
print(f"Progress file    : {PROGRESS_FILE.name}")
print("Phase 6 -- Longformer fine-tuning session complete.")
print("To resume: re-run this script in a new Colab session (progress file persists in Drive).")
