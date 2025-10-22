# TextClassification

**IMDB movie review sentiment classification with an LSTM (PyTorch)**

---

## Project overview

This repository contains a notebook that trains an end‑to‑end sentiment classifier on the IMDB dataset (via the Hugging Face `datasets` library). The model is an embedding + bidirectional LSTM that predicts positive/negative sentiment from raw text. The notebook demonstrates dataset downloading, a minimal tokenizer, vocabulary construction, batch collation with padding, training/validation loops, model saving, and a small inference helper.

## What the notebook implements

* Downloads the IMDB dataset using `datasets.load_dataset('imdb')`.
* Implements a `simple_tokenize` function (regex-based) and builds a top‑N vocabulary (`VOCAB_SIZE`) from the training split.
* Numericizes tokens and wraps the splits into a `torch.utils.data.Dataset` (`IMDBHF`) and `DataLoader` with a custom `collate_fn` that pads/truncates to `MAX_LEN`.
* Defines an `LSTMSentiment` model: `nn.Embedding` → `nn.LSTM` (bidirectional optional) → `nn.Linear` with sigmoid output.
* Uses `BCELoss` + `Adam` optimizer and reports binary accuracy during training.
* Saves the trained model and the `itos` (index→token list) to `models/imdb_lstm_hf.pt`.
* Includes a `predict_text` helper for quick inference on new strings.

## Results (from the notebook run)

* Final training/validation progression (sample):

  * Epochs 1–6 showed final `val_acc` ≈ **0.8102** (after 6 epochs)
* The notebook prints example predictions such as:

  * `0.985 POS` for a positive review sample
  * `0.022 NEG` for a negative review sample

## Files created by the notebook

* `models/imdb_lstm_hf.pt` — saved model state dict and `itos` (vocab)

## How to run locally

1. Clone the repo and open the notebook in Jupyter Lab / Notebook.
2. Install dependencies .
3. Run cells in order. The notebook is self-contained and downloads the IMDB dataset automatically.

## Dependencies
```
torch
datasets
numpy
matplotlib
tqdm
```

(Install a `torch` wheel appropriate for your CUDA/CPU environment.)

## Reproducibility notes

* Seeds are set (`random`, `numpy`, `torch`) with `SEED = 42`.
* The notebook uses `MAX_LEN` and `VOCAB_SIZE` constants so training runs are repeatable given the same environment.
* DataLoader shuffling and `num_workers` can still introduce non-determinism.

## Contact

I'm open to contract and full-time opportunities. Connect with me on LinkedIn: [Nat Andrew](https://www.linkedin.com/in/natandrew).

---
