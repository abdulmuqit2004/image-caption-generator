# Fine-Tuning BLIP for Radiology Image Captioning

**CSCI 4052U – Machine Learning II – Final Project**

Fine-tunes [**BLIP**](https://huggingface.co/Salesforce/blip-image-captioning-base)
(`Salesforce/blip-image-captioning-base`) on the
[**ROCO radiology dataset**](https://huggingface.co/datasets/mdwiratathya/ROCO-radiology)
and measures the improvement over the pretrained baseline on BLEU-4,
METEOR, and BERTScore.

The entire fine-tuning pipeline lives in a **single self-contained
Google Colab notebook** — `blip_roco_finetune.ipynb`. There is no
`requirements.txt`, no setup script, and no local dataset download.
All dependencies, the dataset, and the base model are pulled from the
internet by the notebook itself. It runs end-to-end on a **Colab T4
GPU (16 GB VRAM)**.

Deployment of the resulting model as an interactive demo (Gradio /
HuggingFace Spaces) is a **separate next step** that consumes the
zip this notebook produces — it is not part of this pipeline.

> **Medical disclaimer:** this project is for **educational research
> only**. The captions produced by the model are not clinically
> validated and must not be used to guide any patient care decision.

---

## The problem we are solving

Writing a natural-language description of a radiology image — X-ray,
CT, MRI, ultrasound — is the kind of task that classical computer
vision could not do. Hand-engineered features (HOG, SIFT, edge maps)
paired with classical classifiers can at best recognise that
*something* is in the image, but they cannot bridge the gap from raw
pixels to a fluent, domain-appropriate sentence such as *"chest
radiograph showing bilateral lower-lobe opacities consistent with
pneumonia."* Radiology captioning fuses two subproblems — (a) visual
understanding of modality + anatomy + pathology, and (b) grounded
text generation that respects medical phrasing — and classical
pipelines handled them separately and poorly.

### Why a neural network, and which one

Modern vision-language models solve both halves jointly. We use
**BLIP** (Bootstrapping Language-Image Pretraining, [Li et al.
2022](https://arxiv.org/abs/2201.12086)), which couples a ViT image
encoder with a BERT-style text decoder via cross-attention. BLIP is
pretrained on large web image–caption corpora, so out of the box it
produces general-purpose captions ("a person holding a phone") —
fluent but useless for radiology. The contribution of this project
is to **fine-tune BLIP on radiology image–caption pairs** so its
output distribution shifts into the medical domain, and to quantify
that shift against the pretrained baseline on held-out references.

---

## What the notebook does, step by step

1. Installs every pip dependency inline (first cell) — transformers,
   datasets, evaluate, bert_score, sacrebleu, nltk, matplotlib.
2. Asserts a CUDA GPU is available and prints VRAM / library versions.
3. Downloads the ROCO radiology dataset from HuggingFace with
   `load_dataset("mdwiratathya/ROCO-radiology")` — no manual download.
4. Inspects the dataset (splits, columns, caption length distribution)
   and auto-detects the image and caption column names.
5. Filters captions to a 10–120 word window and subsets 8,000 training
   and 1,000 validation examples (seed = 42).
6. Downloads the BLIP base checkpoint and processor from HuggingFace.
7. Fine-tunes BLIP end-to-end (full model by default; a single flag
   switches to decoder-only training with the vision encoder frozen).
8. Checkpoints after every epoch and keeps the lowest-validation-loss
   checkpoint as the final model.
9. Compares pretrained vs fine-tuned BLIP on the 1k val set using
   BLEU-4, METEOR, and BERTScore, plus six qualitative side-by-side
   examples and a loss curve.
10. Zips the fine-tuned model directory + processor and triggers a
    browser download via `google.colab.files.download`. An optional
    cell copies the zip to Google Drive instead.

---

## BLIP architecture in one paragraph

BLIP is a three-part vision-language model. A **ViT-B/16 image
encoder** turns the input image into a grid of patch embeddings. A
**text decoder** (BERT-style transformer) generates the caption
autoregressively. The two are bridged by **cross-attention** layers
inside the decoder that attend from text tokens back to the image
patches. For captioning, BLIP is trained with a standard
language-modelling loss on (image, caption) pairs — the decoder
learns to predict the next token conditioned on both the image
features and the previously generated text. The base checkpoint used
here has ~224M parameters, small enough to fine-tune end-to-end on a
Colab T4 under fp16.

## ROCO dataset

ROCO (Radiology Objects in COntext) is a large collection of
radiology-related figures scraped from open-access articles on PubMed
Central. Each example is an image paired with the caption the paper
authors wrote for that figure. Captions are natural language and
vary in length and style, which is why we filter to 10–120 words
before training. The `mdwiratathya/ROCO-radiology` release on
HuggingFace is a cleaned radiology-only variant and is what this
notebook consumes.

The dataset comes with native `train` / `validation` / `test` splits.
The notebook auto-detects them and uses `train` as the training pool
and `validation` as the evaluation pool.

---

## End-to-end fine-tuning pipeline

```
[HuggingFace Hub]
   - mdwiratathya/ROCO-radiology  (dataset)
   - Salesforce/blip-image-captioning-base  (weights + processor)
          |
          v  load_dataset / from_pretrained
  caption_ok filter  ──►  8k train / 1k val subset
          |                        |
          v                        v
    BlipProcessor ──► tokens + pixel_values ──► BLIP (ViT + decoder)
                                                       |
                                                       v
                                              LM loss backward
                                                       |
                                                       v
                                               AdamW + cosine LR
                                                       |
                                                       v
                                     best-val checkpoint ──► generate()
                                                                  |
                                                                  v
                                             BLEU-4 / METEOR / BERTScore
                                                                  |
                                                                  v
                                      blip-roco-finetuned.zip  (downloaded)
```

### How application data becomes tensors
`BlipProcessor` is the single conversion boundary. For each
`(PIL image, caption)` pair, the processor produces `pixel_values`
(a normalised float tensor shaped `[B, 3, 384, 384]`) and `input_ids`
(tokenised caption, padded to `MAX_TARGET_LEN = 128`). Pad tokens in
the label copy are replaced with `-100` so the language-modelling
loss masks them. At inference time the same processor encodes the
image, and `model.generate(..., num_beams=4)` produces token IDs that
are decoded back to a string with `processor.batch_decode`.

### How the code interfaces with the network
The training loop is hand-written PyTorch (no `Trainer`):
`DataLoader` → collate → fp16 autocast forward → `GradScaler`
backward → AdamW + cosine schedule → per-epoch checkpoint →
best-val-loss tracking. Evaluation loads a second copy of BLIP (the
pretrained baseline) alongside the fine-tuned one and runs `generate`
over the entire val set, then scores with the `evaluate` library.

---

## Fine-tuning strategy

We default to **full fine-tuning**: both the vision encoder and the
text decoder are updated. The model is small enough (~224M params)
that this fits on a T4 under fp16, and in practice it outperforms
decoder-only fine-tuning on radiology captions because the domain is
far enough from natural-image pretraining that the vision encoder
benefits from adapting too.

A `FREEZE_VISION` flag at the top of the training section flips the
notebook into **decoder-only mode**, which trains roughly 2× faster
with about half the trainable parameters — useful when iterating
quickly or running a quota-constrained session.

## Hyperparameters

| param | value | why |
|---|---|---|
| base model | `Salesforce/blip-image-captioning-base` | ~224M params, fits T4 |
| epochs | 5 | captioning converges fast on 8k examples; more epochs overfit |
| batch size | 8 | fits BLIP-base + fp16 + max_len=128 in ~10 GB VRAM |
| grad accum | 4 | effective batch 32, which stabilises the LM loss |
| learning rate | 5e-5 | standard published BLIP fine-tuning LR |
| scheduler | cosine + 5% warmup | smooth decay, no manual step tuning |
| weight decay | 0.01 | AdamW default, mild regulariser |
| grad clip | 1.0 | prevents rare fp16 loss spikes |
| max target len | 128 | covers >99% of captions after 10–120 word filter |
| precision | fp16 | ~2× throughput, half the VRAM |
| seed | 42 | reproducible shuffles + subset selection |

If you hit `CUDA OutOfMemoryError`, drop `BATCH_SIZE` to 4 and raise
`GRAD_ACCUM_STEPS` to 8 to keep the effective batch at 32.

## Evaluation

All three metrics are computed on the 1,000-example validation set
after generating captions with `num_beams=4`.

- **BLEU-4** — n-gram precision against the reference caption.
- **METEOR** — unigram F-measure with stemming and synonym matching
  (better for captioning than BLEU alone).
- **BERTScore-F1** — cosine similarity of contextual embeddings from
  DistilBERT; captures semantic overlap even when wording differs.

## Results

| Metric | Pretrained BLIP | Fine-tuned BLIP | Δ |
|---|---|---|---|
| BLEU-4 | _TBD_ | _TBD_ | _TBD_ |
| METEOR | _TBD_ | _TBD_ | _TBD_ |
| BERTScore-F1 | _TBD_ | _TBD_ | _TBD_ |

Filled in after running the notebook — `metrics.csv` is written to
`/content/blip_roco_outputs/metrics.csv` and the loss curve to
`/content/blip_roco_outputs/plots/loss_curves.png`.

---

## Files in this repo

| file | purpose |
|---|---|
| `blip_roco_finetune.ipynb` | the single self-contained Colab notebook (all installs, dataset loading, training, evaluation, and model export) |
| `README.md` | this file |

## How to run

1. Open `blip_roco_finetune.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → Run all. No `input()` prompts; the notebook is
   non-interactive.
4. At the end, the fine-tuned model zip is downloaded to your browser.
   (Or uncomment the final optional cell to copy it to Google Drive.)

Expected wall-clock on a Colab T4 in full fine-tuning mode: roughly
40–60 minutes for the full 5 epochs on 8k training examples, plus
~10 minutes for generation and scoring on the 1k val set.

## Output artifacts produced inside Colab

After a successful run, `/content/blip_roco_outputs/` contains:

| path | what it is |
|---|---|
| `blip-roco-finetuned/` | final HuggingFace model directory (best val loss) |
| `blip-roco-finetuned.zip` | zipped version of the above, downloaded to your browser |
| `checkpoints/epoch_{1..5}/` | per-epoch checkpoints (kept for safety) |
| `plots/loss_curves.png` | training vs validation loss |
| `plots/qualitative_examples.png` | 6 side-by-side caption comparisons |
| `metrics.csv` | BLEU-4 / METEOR / BERTScore table (pretrained vs fine-tuned) |
| `training_artifacts.zip` | small bundle of the plots + metrics.csv |

## Loading the saved model later (e.g. for a Gradio / HF Spaces demo)

The fine-tuned model is saved in standard HuggingFace format, so any
deployment framework can load it with two lines. Example (this code
would live in the separate deployment notebook, not here):

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

# after unzipping blip-roco-finetuned.zip
processor = BlipProcessor.from_pretrained("blip-roco-finetuned")
model     = BlipForConditionalGeneration.from_pretrained("blip-roco-finetuned")

from PIL import Image
img = Image.open("xray.png").convert("RGB")
ids = model.generate(
    **processor(img, return_tensors="pt"),
    num_beams=4, max_new_tokens=128,
)
print(processor.decode(ids[0], skip_special_tokens=True))
```

The zip contains a standard HuggingFace model directory —
`config.json`, `preprocessor_config.json`, tokenizer files, and
`model.safetensors`.

---

## Screenshots / demo

_To be added after the notebook is run and the separate deployment
demo is recorded._

## References

- Li, J., Li, D., Xiong, C., & Hoi, S. (2022). *BLIP: Bootstrapping
  Language-Image Pre-training for Unified Vision-Language
  Understanding and Generation.* ICML.
  <https://arxiv.org/abs/2201.12086>
- Pelka, O., Koitka, S., Rückert, J., Nensa, F., & Friedrich, C. M.
  (2018). *Radiology Objects in COntext (ROCO): A Multimodal Image
  Dataset.* MICCAI LABELS Workshop.
- HuggingFace dataset card: <https://huggingface.co/datasets/mdwiratathya/ROCO-radiology>
