from pathlib import Path
import json
import io
import tempfile
import streamlit as st
from typing import List, Tuple
from PIL import Image
import os

# -----------------------
# Config & constants
# -----------------------
DEMO_DIR = Path("demo_assets")
DEMO_JSON = DEMO_DIR / "demo_preds.json"
DEFAULT_MODEL = Path("models/imdb_lstm_hf.pt")  # expected checkpoint format: {'model_state_dict', 'itos'}

SEED = 42
MAX_LEN = 300  # should match the notebook default

# -----------------------
# Try to import torch (conditional)
# -----------------------
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# -----------------------
# Tokenizer & helpers (same as notebook)
# -----------------------
import re
from collections import Counter
_token_pattern = re.compile(r"\b\w+\b")
def simple_tokenize(text: str):
    return _token_pattern.findall(text.lower())

# tokens -> indices
def tokens_to_indices(tokens, stoi, unk_idx=1):
    return [stoi.get(t, unk_idx) for t in tokens]

# -----------------------
# Model class (same arch as your notebook)
# -----------------------
if TORCH_AVAILABLE:
    class LSTMSentiment(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, hidden_size=128, num_layers=1, bidirectional=True, dropout=0.5, padding_idx=0):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
            self.embedding_dropout = nn.Dropout1d(0.2)
            self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

        def forward(self, x, lengths=None):
            emb = self.embedding(x)  # (batch, seq_len, embed_dim)
            emb = emb.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
            emb = self.embedding_dropout(emb)
            emb = emb.permute(0, 2, 1)
            out, (h_n, c_n) = self.lstm(emb)
            if self.lstm.bidirectional:
                h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                h_final = h_n[-1]
            h_final = self.dropout(h_final)
            return torch.sigmoid(self.fc(h_final)).squeeze(1)

    @st.cache_resource
    def load_model_checkpoint(path: str | None):
        """
        Load model checkpoint and return (model, itos list, stoi dict, device)
        If path is None or loading fails, returns (None, None, None, None)
        Expected checkpoint format: {'model_state_dict':..., 'itos': [...]}
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if path is None or not os.path.exists(path):
            return None, None, None, device
        try:
            ckpt = torch.load(path, map_location=device)
            itos = ckpt.get("itos", None)
            if itos is None:
                st.warning("Checkpoint does not contain 'itos' -- inference may not work as expected.")
                return None, None, None, device
            stoi = {tok: i for i, tok in enumerate(itos)}
            vocab_size = len(itos)
            model = LSTMSentiment(vocab_size).to(device)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
            model.eval()
            return model, itos, stoi, device
        except Exception as e:
            st.warning(f"Failed to load checkpoint: {e}")
            return None, None, None, device

    def predict_text_with_model(text: str, model, stoi, device, max_len=MAX_LEN) -> float:
        tokens = simple_tokenize(text)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        ids = tokens_to_indices(tokens, stoi)
        tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)
        with torch.no_grad():
            prob = model(tensor, lengths).item()
        return prob

else:
    # placeholders for type hints / UI flow
    def load_model_checkpoint(path: str | None):
        return None, None, None, None
    def predict_text_with_model(*a, **kw):
        raise RuntimeError("Torch not available")

# -----------------------
# Demo-mode helpers (precomputed predictions)
# -----------------------
def load_demo_preds():
    if DEMO_JSON.exists():
        try:
            with open(DEMO_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to read demo JSON: {e}")
            return {}
    return {}

# -----------------------
# Small UI helpers
# -----------------------
def save_uploaded_file(uploaded) -> str | None:
    if uploaded is None:
        return None
    suffix = Path(uploaded.name).suffix
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(uploaded.getbuffer())
    tf.flush()
    tf.close()
    return tf.name

def interpret_label(prob: float) -> Tuple[str, float]:
    label = "POS" if prob >= 0.5 else "NEG"
    return label, prob

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="TextClassification — Streamlit demo", layout="centered")
st.title("TextClassification — IMDB sentiment demo")

st.markdown(
    """
Try the model on your own sentences or pick one of the demo examples.
- If PyTorch is installed in the runtime, the app can load a checkpoint and run real inference.
- Otherwise it uses committed precomputed demo predictions so the public demo works without heavy binary deps.
"""
)

if not TORCH_AVAILABLE:
    st.warning("PyTorch not available in this environment — running in DEMO mode with precomputed predictions. To do real inference, run the app locally with PyTorch installed or host a remote inference service.")

# Sidebar controls: model / checkpoint
st.sidebar.header("Model / checkpoint")
use_default = st.sidebar.checkbox(f"Load default checkpoint ({DEFAULT_MODEL}) if present", value=True)
uploaded_ckpt = st.sidebar.file_uploader("Upload checkpoint (.pt/.pth)", type=["pt", "pth", "bin"])
remote_ckpt_url = st.sidebar.text_input("Optional: download checkpoint from URL (raw link)")

# determine checkpoint path (uploaded -> local -> remote)
checkpoint_path = None
if uploaded_ckpt is not None:
    tmp = save_uploaded_file(uploaded_ckpt)
    if tmp:
        checkpoint_path = tmp
elif use_default and DEFAULT_MODEL.exists():
    checkpoint_path = str(DEFAULT_MODEL)
elif remote_ckpt_url:
    # try to download once
    try:
        os.makedirs("models", exist_ok=True)
        dst = Path("models") / Path(remote_ckpt_url).name
        if not dst.exists():
            import requests
            with requests.get(remote_ckpt_url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
        checkpoint_path = str(dst)
    except Exception as e:
        st.sidebar.warning(f"Failed to download checkpoint: {e}")

# Load model if available
model, itos, stoi, device = load_model_checkpoint(checkpoint_path if TORCH_AVAILABLE else None) if TORCH_AVAILABLE else (None, None, None, None)

# Main input area
st.header("Input text")
col1, col2 = st.columns([2,1])

with col1:
    user_text = st.text_area("Enter text to classify", value="This movie was fantastic. The acting and story were great.", height=150)
    if st.button("Run inference"):
        if TORCH_AVAILABLE and model is not None:
            try:
                prob = predict_text_with_model(user_text, model, stoi, device)
                label, p = interpret_label(prob)
                st.success(f"{label} — {p*100:.2f}% positive")
            except Exception as e:
                st.error(f"Model inference failed: {e}")
        else:
            st.info("Torch not available or model not loaded. Using demo-mode predictions if available.")
            demo_preds = load_demo_preds()
            # Exact-match fallback: if the exact text is in demo preds, show it, otherwise show a random demo sample
            if user_text in demo_preds:
                obj = demo_preds[user_text]
                st.write(f"Demo precomputed: {obj['label']} — {obj['p']*100:.2f}%")
            else:
                # if nothing matched, show examples
                st.info("No precomputed match found. Pick a demo example on the right or run locally with PyTorch for live results.")

with col2:
    st.markdown("**Demo examples**")
    demo_preds = load_demo_preds()
    if demo_preds:
        keys = list(demo_preds.keys())
        choice = st.selectbox("Pick demo example", keys)
        if choice:
            obj = demo_preds[choice]
            st.write("Text:")
            st.write(choice)
            st.write("Precomputed prediction:")
            st.write(f"**{obj['label']}** — {obj['p']*100:.2f}%")
    else:
        st.info("No demo assets found. Generate `demo_assets/demo_preds.json` locally using gen_demo_assets.py and commit it to the repo.")
