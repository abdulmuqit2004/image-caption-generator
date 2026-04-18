"""Generates the Colab notebook programmatically. Easier to maintain than hand-writing .ipynb JSON."""
import json
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ============================================================
# SECTION 1: TITLE + OVERVIEW
# ============================================================
md("""# Fine-Tuning BLIP for Radiology Image Captioning (ROCO)

This notebook fine-tunes `Salesforce/blip-image-captioning-base` on the
[`mdwiratathya/ROCO-radiology`](https://huggingface.co/datasets/mdwiratathya/ROCO-radiology)
dataset and compares the fine-tuned model to the pretrained baseline on
BLEU-4, METEOR, and BERTScore.

**Runs end-to-end on Colab T4 (16 GB VRAM).** No local deployment — that
happens in a separate notebook.

### Section map
1. Environment setup + GPU check
2. Dataset inspection (verify columns, splits, caption distribution)
3. Data filtering + train/val subsets
4. Model + processor + hyperparameters
5. Dataloader + training loop with per-epoch checkpointing
6. Evaluation: pretrained vs fine-tuned, BLEU / METEOR / BERTScore
7. Qualitative comparison (6 side-by-side examples)
8. Loss curves
9. Save model + zip + download

> **Medical disclaimer:** outputs are for educational research only and
> must not be used for clinical decisions.""")

# ============================================================
# SECTION 2: SETUP
# ============================================================
md("""## 1. Environment setup

All `pip` installs happen in this single cell. **Restart the runtime if
Colab prompts you to after install.**""")

code("""# --- one-shot install ---
!pip install -q --upgrade \\
    "transformers>=4.40.0,<4.50.0" \\
    "datasets>=2.19.0" \\
    "accelerate>=0.29.0" \\
    "evaluate>=0.4.1" \\
    "bert_score>=0.3.13" \\
    "sacrebleu>=2.4.0" \\
    "nltk>=3.8.1" \\
    "Pillow>=10.0.0" \\
    "matplotlib>=3.8.0"

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
print("Install complete.")""")

code("""# --- GPU assertion: fail loudly if no CUDA ---
import torch, sys
assert torch.cuda.is_available(), (
    "CUDA GPU not available. In Colab: Runtime -> Change runtime type -> "
    "Hardware accelerator -> T4 GPU, then rerun."
)
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name}")
print(f"VRAM: {vram_gb:.1f} GB")
print(f"PyTorch: {torch.__version__}")
import transformers
print(f"Transformers: {transformers.__version__}")""")

code("""# --- global paths + reproducibility ---
import os, random, numpy as np, torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

OUT_DIR       = "/content/blip_roco_outputs"
CKPT_DIR      = f"{OUT_DIR}/checkpoints"
FINAL_MODEL   = f"{OUT_DIR}/blip-roco-finetuned"
PLOTS_DIR     = f"{OUT_DIR}/plots"
ZIP_PATH      = f"{OUT_DIR}/blip-roco-finetuned.zip"

for d in (OUT_DIR, CKPT_DIR, FINAL_MODEL, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)
print("Output dirs ready under", OUT_DIR)""")

# ============================================================
# SECTION 3: INSPECTION
# ============================================================
md("""## 2. Dataset inspection — verify assumptions before training

We do not assume column names, splits, or caption lengths. This section
loads the dataset and prints what is actually there.

### What a prior inspection found (verified 2026-04)
- **Splits:** `train` (65,419), `validation` (8,175), `test` (8,176).
- **Columns:** `image` (PIL.Image, RGB or L mode), `image_id` (str),
  `caption` (str).
- **Captions** have leading spaces and a trailing newline — the
  notebook strips them before training.
- **Image sizes** vary (typical range ~300-1200 px on each side); the
  BLIP processor resizes to 384x384.
- We subset down to 8,000 train / 1,000 val after filtering captions
  to 10-120 words.

The cells below re-verify these findings at runtime — if the dataset
changes upstream the notebook will detect it and either adapt or fail
with a clear message.""")

code("""from datasets import load_dataset

print("Loading mdwiratathya/ROCO-radiology ...")
raw_ds = load_dataset("mdwiratathya/ROCO-radiology")

print("\\n=== SPLITS ===")
for split_name, split in raw_ds.items():
    print(f"  {split_name}: {len(split):,} examples")

first_split = list(raw_ds.keys())[0]
print(f"\\n=== COLUMNS in '{first_split}' ===")
print(raw_ds[first_split].column_names)
print("\\n=== FEATURES ===")
print(raw_ds[first_split].features)""")

code("""# --- detect image + caption columns robustly ---
# ROCO variants commonly use ('image', 'caption'), but we detect dynamically.
from PIL import Image

sample = raw_ds[first_split][0]
IMAGE_COL, CAPTION_COL = None, None
for k, v in sample.items():
    if IMAGE_COL is None and isinstance(v, Image.Image):
        IMAGE_COL = k
    elif CAPTION_COL is None and isinstance(v, str) and len(v.split()) > 2:
        CAPTION_COL = k

# Fallbacks by name if auto-detect misses
if IMAGE_COL is None:
    for cand in ("image", "img", "pixel_values"):
        if cand in sample: IMAGE_COL = cand; break
if CAPTION_COL is None:
    for cand in ("caption", "text", "report", "findings"):
        if cand in sample: CAPTION_COL = cand; break

assert IMAGE_COL and CAPTION_COL, (
    f"Could not detect image/caption columns from {list(sample.keys())}"
)
print(f"IMAGE_COL   = {IMAGE_COL!r}")
print(f"CAPTION_COL = {CAPTION_COL!r}")""")

code("""# --- 15 sample captions with word counts ---
print(f"=== 15 CAPTION SAMPLES from '{first_split}' ===\\n")
for i in range(15):
    cap = raw_ds[first_split][i][CAPTION_COL]
    print(f"[{len(cap.split()):3d} words] {cap[:240]}")
    print("-" * 80)""")

code("""# --- caption length distribution + image sanity check ---
import matplotlib.pyplot as plt

sample_n = min(3000, len(raw_ds[first_split]))
word_counts = [
    len(raw_ds[first_split][i][CAPTION_COL].split())
    for i in range(sample_n)
]
print(f"Caption word-count stats over {sample_n} samples:")
print(f"  min    = {min(word_counts)}")
print(f"  median = {int(np.median(word_counts))}")
print(f"  mean   = {np.mean(word_counts):.1f}")
print(f"  p95    = {int(np.percentile(word_counts, 95))}")
print(f"  max    = {max(word_counts)}")
print(f"  in 10-120 word window: "
      f"{sum(10 <= w <= 120 for w in word_counts):,} / {sample_n}")

plt.figure(figsize=(8,3))
plt.hist(word_counts, bins=60)
plt.xlabel("caption word count"); plt.ylabel("frequency")
plt.title("ROCO caption length distribution")
plt.axvline(10, color="red", linestyle="--", label="10 (min keep)")
plt.axvline(120, color="red", linestyle="--", label="120 (max keep)")
plt.legend(); plt.tight_layout(); plt.show()

# image sanity
img0 = raw_ds[first_split][0][IMAGE_COL]
print(f"\\nFirst image: type={type(img0).__name__}, size={img0.size}, mode={img0.mode}")""")

code("""# --- decide splits: use native val/test if present, else split manually ---
split_keys = list(raw_ds.keys())
TRAIN_SPLIT = "train" if "train" in split_keys else split_keys[0]

val_candidates = [k for k in split_keys if k in ("validation", "valid", "val", "test")]
if val_candidates:
    VAL_SPLIT = val_candidates[0]
    use_manual_split = False
    print(f"Using native splits: train='{TRAIN_SPLIT}'  val='{VAL_SPLIT}'")
else:
    VAL_SPLIT = None
    use_manual_split = True
    print(f"No native val split. Will split '{TRAIN_SPLIT}' with seed={SEED}.")""")

# ============================================================
# SECTION 4: FILTERING + SUBSETS
# ============================================================
md("""## 3. Filter captions and build train / val subsets

- Drop captions with <10 or >120 words
- Drop unreadable images
- Take 8,000 train and 1,000 val examples
- Deterministic with `seed=42`""")

code("""TRAIN_SIZE = 8000
VAL_SIZE   = 1000
MIN_WORDS  = 10
MAX_WORDS  = 120

def caption_ok(ex):
    cap = ex.get(CAPTION_COL, "")
    if not isinstance(cap, str): return False
    n = len(cap.strip().split())
    return MIN_WORDS <= n <= MAX_WORDS

# Filter raw splits (ROCO captions have leading spaces / trailing \\n)
train_pool = raw_ds[TRAIN_SPLIT].filter(caption_ok)
print(f"Filtered train pool: {len(train_pool):,}")

if use_manual_split:
    shuffled = train_pool.shuffle(seed=SEED)
    need = TRAIN_SIZE + VAL_SIZE
    assert len(shuffled) >= need, f"need {need}, have {len(shuffled)}"
    train_ds = shuffled.select(range(TRAIN_SIZE))
    val_ds   = shuffled.select(range(TRAIN_SIZE, need))
else:
    val_pool = raw_ds[VAL_SPLIT].filter(caption_ok)
    print(f"Filtered val pool:   {len(val_pool):,}")
    train_ds = train_pool.shuffle(seed=SEED).select(
        range(min(TRAIN_SIZE, len(train_pool)))
    )
    val_ds = val_pool.shuffle(seed=SEED).select(
        range(min(VAL_SIZE, len(val_pool)))
    )

print(f"\\nFinal train: {len(train_ds):,}")
print(f"Final val:   {len(val_ds):,}")""")

# ============================================================
# SECTION 5: MODEL + HYPERPARAMS
# ============================================================
md("""## 4. Model, processor, and hyperparameters

All training hyperparameters live here. Nothing is hardcoded further down.

### Why these values
| param | value | reason |
|---|---|---|
| base model | `Salesforce/blip-image-captioning-base` | ~224M params, fits T4 comfortably |
| epochs | 5 | captions converge fast; more overfits 8k samples |
| batch size | 8 | fits fp16 BLIP-base in ~10 GB VRAM on T4 |
| grad accum | 4 | effective batch = 32, stable for captioning |
| lr | 5e-5 | standard BLIP fine-tuning LR |
| scheduler | cosine | smooth decay, no manual step |
| weight decay | 0.01 | AdamW default |
| grad clip | 1.0 | prevents rare loss spikes |
| max tgt len | 128 | covers >99% of filtered captions |
| fp16 | on | ~2x throughput, half the VRAM |
| freeze vision | toggle | decoder-only is faster; full usually scores higher |""")

code("""# ============= HYPERPARAMETERS — edit here, nowhere else =============
MODEL_ID           = "Salesforce/blip-image-captioning-base"
FREEZE_VISION      = False      # True = decoder-only fine-tuning
EPOCHS             = 5
BATCH_SIZE         = 8          # drop to 4 if CUDA OOM
GRAD_ACCUM_STEPS   = 4          # effective batch = 32
LEARNING_RATE      = 5e-5
WEIGHT_DECAY       = 0.01
GRAD_CLIP          = 1.0
WARMUP_RATIO       = 0.05
MAX_TARGET_LEN     = 128
USE_FP16           = True
NUM_WORKERS        = 2
# =====================================================================

EFF_BATCH = BATCH_SIZE * GRAD_ACCUM_STEPS
print(f"Effective batch size: {EFF_BATCH}")
print(f"Fine-tuning mode:     "
      f"{'decoder-only (vision frozen)' if FREEZE_VISION else 'full'}")""")

code("""from transformers import BlipProcessor, BlipForConditionalGeneration

print(f"Loading {MODEL_ID} ...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model     = BlipForConditionalGeneration.from_pretrained(MODEL_ID)

if FREEZE_VISION:
    for p in model.vision_model.parameters():
        p.requires_grad = False
    print("Vision encoder frozen.")

device = torch.device("cuda")
model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M "
      f"({100*trainable/total:.1f}%)")""")

# ============================================================
# SECTION 6: DATALOADER + COLLATE
# ============================================================
md("""## 5. Dataloader and collate function

The collate function calls the BLIP processor on a batch of PIL images +
captions. Labels with `pad_token_id` are masked with `-100` so loss
ignores padding.""")

code("""from torch.utils.data import DataLoader

def blip_collate(batch):
    images   = [ex[IMAGE_COL].convert("RGB") for ex in batch]
    captions = [ex[CAPTION_COL].strip() for ex in batch]

    enc = processor(
        images=images,
        text=captions,
        padding="max_length",
        truncation=True,
        max_length=MAX_TARGET_LEN,
        return_tensors="pt",
    )
    labels = enc["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    enc["labels"] = labels
    return enc

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, collate_fn=blip_collate, pin_memory=True,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, collate_fn=blip_collate, pin_memory=True,
)
print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")""")

# ============================================================
# SECTION 7: TRAINING LOOP
# ============================================================
md("""## 6. Training loop

- fp16 autocast + `GradScaler`
- Gradient accumulation over `GRAD_ACCUM_STEPS` mini-batches
- Cosine LR schedule with linear warmup
- Gradient clipping at 1.0
- **Checkpoint after every epoch** (Colab can disconnect)
- Best checkpoint (lowest val loss) is copied to `FINAL_MODEL`""")

code("""from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import shutil, time, math

steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
total_steps     = steps_per_epoch * EPOCHS
warmup_steps    = int(total_steps * WARMUP_RATIO)

optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
)
scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)

print(f"Optimizer steps per epoch: {steps_per_epoch}")
print(f"Total optimizer steps:     {total_steps}")
print(f"Warmup steps:              {warmup_steps}")""")

code("""from tqdm.auto import tqdm

@torch.no_grad()
def eval_loss(model, loader):
    model.eval()
    tot, n = 0.0, 0
    for batch in tqdm(loader, desc="val", leave=False):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=USE_FP16):
            out = model(**batch)
        tot += out.loss.item() * batch["input_ids"].size(0)
        n   += batch["input_ids"].size(0)
    model.train()
    return tot / max(n, 1)

train_losses, val_losses = [], []
best_val = float("inf")
t_start = time.time()

try:
    for epoch in range(1, EPOCHS + 1):
        print(f"\\n===== Epoch {epoch}/{EPOCHS} =====")
        model.train()
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"train e{epoch}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=USE_FP16):
                out = model(**batch)
                loss = out.loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    GRAD_CLIP,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running += out.loss.item() * batch["input_ids"].size(0)
            seen    += batch["input_ids"].size(0)
            pbar.set_postfix(loss=f"{running/seen:.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

        tr_loss = running / max(seen, 1)
        vl_loss = eval_loss(model, val_loader)
        train_losses.append(tr_loss); val_losses.append(vl_loss)
        print(f"epoch {epoch}: train={tr_loss:.4f}  val={vl_loss:.4f}")

        # --- always checkpoint (Colab may disconnect) ---
        ep_dir = f"{CKPT_DIR}/epoch_{epoch}"
        model.save_pretrained(ep_dir)
        processor.save_pretrained(ep_dir)
        print(f"checkpoint -> {ep_dir}")

        if vl_loss < best_val:
            best_val = vl_loss
            if os.path.exists(FINAL_MODEL):
                shutil.rmtree(FINAL_MODEL)
            shutil.copytree(ep_dir, FINAL_MODEL)
            print(f"new best val={vl_loss:.4f}  -> {FINAL_MODEL}")

except torch.cuda.OutOfMemoryError:
    print("\\nCUDA OOM. Lower BATCH_SIZE to 4 (raise GRAD_ACCUM_STEPS to 8 to "
          "keep effective batch=32) and rerun the training cell.")
    raise

print(f"\\nTotal train time: {(time.time()-t_start)/60:.1f} min")
print(f"Best val loss:    {best_val:.4f}")""")

# ============================================================
# SECTION 8: EVAL
# ============================================================
md("""## 7. Evaluation — pretrained vs fine-tuned

We generate captions on the **entire validation set** with
`num_beams=4` for both the original pretrained BLIP and the fine-tuned
checkpoint, then compute BLEU-4, METEOR, and BERTScore.""")

code("""# free VRAM before loading a second copy of BLIP
import gc
del optimizer, scheduler, scaler
gc.collect(); torch.cuda.empty_cache()

print("Loading pretrained baseline ...")
pretrained = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
pretrained.eval()

print(f"Loading fine-tuned from {FINAL_MODEL} ...")
finetuned = BlipForConditionalGeneration.from_pretrained(FINAL_MODEL).to(device)
finetuned.eval()""")

code("""@torch.no_grad()
def generate_captions(model_, loader_ds, batch_size=8, num_beams=4, max_new_tokens=128):
    preds, refs, imgs = [], [], []
    model_.eval()
    for start in tqdm(range(0, len(loader_ds), batch_size), desc="generate"):
        chunk = [loader_ds[i] for i in range(start, min(start+batch_size, len(loader_ds)))]
        images   = [ex[IMAGE_COL].convert("RGB") for ex in chunk]
        captions = [ex[CAPTION_COL].strip() for ex in chunk]

        pv = processor(images=images, return_tensors="pt").to(device)
        with torch.cuda.amp.autocast(enabled=USE_FP16):
            out = model_.generate(
                **pv, num_beams=num_beams, max_new_tokens=max_new_tokens,
                early_stopping=True,
            )
        decoded = processor.batch_decode(out, skip_special_tokens=True)
        preds.extend([d.strip() for d in decoded])
        refs.extend(captions)
        imgs.extend(images)
    return preds, refs, imgs

print("Generating with pretrained ...")
pre_preds, refs, val_images = generate_captions(pretrained, val_ds)

print("Generating with fine-tuned ...")
ft_preds, _, _ = generate_captions(finetuned, val_ds)

print(f"Generated {len(pre_preds)} captions from each model.")""")

code("""import evaluate

bleu      = evaluate.load("bleu")
meteor    = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")

def score_all(preds, refs):
    b = bleu.compute(predictions=preds, references=[[r] for r in refs], max_order=4)
    m = meteor.compute(predictions=preds, references=refs)
    bs = bertscore.compute(
        predictions=preds, references=refs,
        lang="en", model_type="distilbert-base-uncased",
        batch_size=16, device="cuda",
    )
    return {
        "BLEU-4":      b["bleu"],
        "METEOR":      m["meteor"],
        "BERTScore-F1": float(np.mean(bs["f1"])),
    }

print("Scoring pretrained ...")
pre_scores = score_all(pre_preds, refs)
print("Scoring fine-tuned ...")
ft_scores  = score_all(ft_preds, refs)

import pandas as pd
cmp_df = pd.DataFrame(
    {"Pretrained BLIP": pre_scores, "Fine-tuned BLIP": ft_scores}
).round(4)
cmp_df["delta"] = (cmp_df["Fine-tuned BLIP"] - cmp_df["Pretrained BLIP"]).round(4)
print("\\n=== Metric comparison on validation set ===")
print(cmp_df)
cmp_df.to_csv(f"{OUT_DIR}/metrics.csv")
""")

# ============================================================
# SECTION 9: QUALITATIVE
# ============================================================
md("""## 8. Qualitative examples — pretrained vs fine-tuned side by side""")

code("""import matplotlib.pyplot as plt

N_SHOW = 6
idxs = np.random.default_rng(SEED).choice(len(val_images), N_SHOW, replace=False)

fig, axes = plt.subplots(N_SHOW, 1, figsize=(10, 4.2*N_SHOW))
if N_SHOW == 1: axes = [axes]
for ax, i in zip(axes, idxs):
    ax.imshow(val_images[i], cmap="gray")
    ax.axis("off")
    txt = (f"GROUND TRUTH:\\n{refs[i][:260]}\\n\\n"
           f"PRETRAINED:\\n{pre_preds[i][:260]}\\n\\n"
           f"FINE-TUNED:\\n{ft_preds[i][:260]}")
    ax.set_title(txt, loc="left", fontsize=9, wrap=True)

plt.tight_layout()
qpath = f"{PLOTS_DIR}/qualitative_examples.png"
plt.savefig(qpath, dpi=130, bbox_inches="tight")
plt.show()
print(f"Saved {qpath}")""")

# ============================================================
# SECTION 10: LOSS CURVES
# ============================================================
md("""## 9. Training / validation loss curves""")

code("""plt.figure(figsize=(7,4))
ep = list(range(1, len(train_losses)+1))
plt.plot(ep, train_losses, marker="o", label="train")
plt.plot(ep, val_losses,   marker="s", label="val")
plt.xlabel("epoch"); plt.ylabel("loss")
plt.title("BLIP fine-tuning loss")
plt.grid(alpha=0.3); plt.legend()
lpath = f"{PLOTS_DIR}/loss_curves.png"
plt.tight_layout(); plt.savefig(lpath, dpi=130); plt.show()
print(f"Saved {lpath}")""")

# ============================================================
# SECTION 11: SAVE + ZIP + DOWNLOAD
# ============================================================
md("""## 10. Package the fine-tuned model and download

The zip contains the HuggingFace model + processor, ready to load with

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("blip-roco-finetuned")
model     = BlipForConditionalGeneration.from_pretrained("blip-roco-finetuned")
```""")

code("""import shutil

print(f"Zipping {FINAL_MODEL} ...")
shutil.make_archive(ZIP_PATH.replace(".zip",""), "zip", FINAL_MODEL)
sz_mb = os.path.getsize(ZIP_PATH) / 1e6
print(f"Zip: {ZIP_PATH}  ({sz_mb:.1f} MB)")

# plots + metrics bundle
BUNDLE_ZIP = f"{OUT_DIR}/training_artifacts.zip"
shutil.make_archive(BUNDLE_ZIP.replace(".zip",""), "zip", OUT_DIR)
print(f"Artifacts bundle: {BUNDLE_ZIP}")""")

code("""# --- trigger browser download to user's local machine ---
try:
    from google.colab import files
    print("Starting browser download of fine-tuned model zip ...")
    files.download(ZIP_PATH)
except Exception as e:
    print("files.download not available (not in Colab?):", e)
    print("Download manually from the file panel:", ZIP_PATH)""")

md("""### Optional: save to Google Drive

Uncomment the cell below if your Drive is mounted at `/content/drive`.""")

code("""# from google.colab import drive
# drive.mount('/content/drive')
# import shutil
# dst = "/content/drive/MyDrive/blip-roco-finetuned.zip"
# shutil.copy(ZIP_PATH, dst)
# print("Copied to", dst)""")

md("""---
### Summary

- Fine-tuned model dir: `/content/blip_roco_outputs/blip-roco-finetuned/`
- Zip for download: `/content/blip_roco_outputs/blip-roco-finetuned.zip`
- Loss curve: `/content/blip_roco_outputs/plots/loss_curves.png`
- Qualitative examples: `/content/blip_roco_outputs/plots/qualitative_examples.png`
- Metrics CSV: `/content/blip_roco_outputs/metrics.csv`

Deployment / demo is handled in a separate notebook.""")

# ============================================================
# write
# ============================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
}
with open("blip_roco_finetune.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote blip_roco_finetune.ipynb")
print(f"Cells: {len(cells)}")
