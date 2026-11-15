"""
Project 2 - Notebook 5: Streamlit app (TeamKAI: KeiTonia, Ashley, India)

Run from the repo root:
    streamlit run app/P2_Images-5_App_TeamKAI.py

Needs (besides the packages in your environment: streamlit, tensorflow, pillow, numpy, pandas, matplotlib):
    app/image_explainers.py        the provided explainer module (copy it next to this file)
    models/P2_N3_<application>.keras   one trained probe per Keras application (Notebook 3)
    models/P2_label_map.json           label -> index mapping (Notebook 3)

What it does: one uploaded image goes through EVERY Keras application's joined model (no picker),
one model per row, with a sorted probability bar plot and three explanation maps per model.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # quiet TensorFlow start-up messages

import sys
import json
import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from matplotlib.figure import Figure

st.set_page_config(page_title="Cloud classifier + explainers", page_icon="☁️", layout="wide")

# ------------------------------------------------------------------ paths and settings
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent                      # repo root (the folder you run streamlit from)
MODEL_DIR = ROOT / "models"
for p in (APP_DIR, ROOT / "notebooks"):    # find image_explainers.py in app/ (or notebooks/)
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import image_explainers as xai
except ImportError:
    st.error("Could not import `image_explainers`. Copy `image_explainers.py` into the `app/` folder "
             "(next to this script) and run `streamlit run app/P2_Images-5_App_TeamKAI.py` again.")
    st.stop()

import tensorflow as tf
from tensorflow import keras

# Must match Notebooks 2 and 3 exactly.
IMG_SIZE = 224          # target size from Notebook 2
PAD_VALUE = 128         # letterbox pad value from Notebook 2
SMOOTH_DEFAULT = 32     # SmoothGrad samples, as in Notebook 4
FIX_EXIF_ORIENTATION = True   # phone photos carry a rotation tag; Notebook 2 did not need this for scraped files
MAP_WIDTH = 280        # pixel width of each explanation map
LOW_CONFIDENCE = 0.60   # rule-of-thumb line for the "low confidence" warning (not a calibrated threshold)

# Same applications, same order (small -> large) as Notebook 3. The key is the name in P2_N3_<key>.keras.
APPS = {
    "mobilenetv2": dict(name="MobileNetV2",     size="small",  builder=keras.applications.MobileNetV2),
    "effnetv2s":   dict(name="EfficientNetV2S", size="medium", builder=keras.applications.EfficientNetV2S),
    "resnet50":    dict(name="ResNet50",        size="medium", builder=keras.applications.ResNet50),
    "effnetv2m":   dict(name="EfficientNetV2M", size="large",  builder=keras.applications.EfficientNetV2M),
}


# ------------------------------------------------------------------ Notebook 2 resize (copied exactly)
def letterbox(path_or_img, size=IMG_SIZE, pad_value=PAD_VALUE):
    """RGB, scale to fit inside size x size preserving aspect ratio, center on a gray canvas. Returns uint8."""
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    im = im.convert("RGB")
    scale = size / max(im.size)
    new = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
    im = im.resize(new, Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (pad_value,) * 3)
    canvas.paste(im, ((size - new[0]) // 2, (size - new[1]) // 2))
    return np.asarray(canvas, dtype=np.uint8)


# ------------------------------------------------------------------ models and labels (cached)
@st.cache_resource(show_spinner=False)
def load_models():
    """Rebuild each application as in Notebook 3 and join it to its probe (U2-2_CNN-6 section 5)."""
    joined, missing = {}, []
    for key, spec in APPS.items():
        probe_path = MODEL_DIR / f"P2_N3_{key}.keras"
        if not probe_path.exists():
            missing.append(key)
            continue
        base = spec["builder"](weights="imagenet", include_top=False, pooling="avg",
                               input_shape=(IMG_SIZE, IMG_SIZE, 3))
        base.trainable = False
        probe = keras.models.load_model(probe_path)
        model = keras.Sequential([keras.Input((IMG_SIZE, IMG_SIZE, 3)), base, probe], name=f"joined_{key}")
        model.trainable = False
        joined[key] = model
    return joined, missing


@st.cache_data(show_spinner=False)
def load_class_names():
    """Label -> index mapping saved in Notebook 3, returned as a list ordered by index."""
    with open(MODEL_DIR / "P2_label_map.json") as f:
        mapping = json.load(f)
    if all(str(k).isdigit() for k in mapping):          # accept {index: name} as well
        mapping = {v: int(k) for k, v in mapping.items()}
    inverse = {i: name for name, i in mapping.items()}
    return [inverse[i] for i in range(len(inverse))]


# ------------------------------------------------------------------ per-image computations (cached)
def as_probs(p, n_classes):
    p = np.asarray(p, dtype="float64").squeeze()
    if p.ndim != 1 or len(p) != n_classes:
        raise ValueError(f"xai.predict_proba returned shape {p.shape}; expected {n_classes} probabilities.")
    return p


@st.cache_data(show_spinner=False, max_entries=64)
def get_probs(key, img_hash, n_classes, _model, _img):
    return as_probs(xai.predict_proba(_model, _img), n_classes)


@st.cache_data(show_spinner=False, max_entries=64)
def get_maps(key, img_hash, smooth, _model, _img):
    """Three maps from image_explainers, each overlaid on the image with xai.overlay."""
    t0 = time.time()
    heat = {
        "Raw saliency": xai.saliency(_model, _img),
        f"SmoothGrad (smooth={smooth})": xai.saliency(_model, _img, smooth=smooth),
        "Grad-CAM": xai.grad_cam(_model, _img),
    }
    overlays = {name: np.asarray(xai.overlay(_img, h)) for name, h in heat.items()}
    return overlays, time.time() - t0


def prob_plot(p, class_names):
    """Horizontal bar plot of all class probabilities, highest first (at the top)."""
    order = np.argsort(-p)
    fig = Figure(figsize=(4.4, 2.7))
    ax = fig.subplots()
    ax.barh(range(len(order)), p[order], color=["#1f77b4"] + ["#9ecae1"] * (len(order) - 1))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([class_names[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    for y, i in enumerate(order):
        ax.text(min(p[i] + 0.02, 0.86), y, f"{p[i]:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    return fig


def normalized_entropy(p):
    """0 = all probability on one class, 1 = spread evenly over all classes."""
    q = np.clip(p, 1e-12, 1)
    return float(-(q * np.log(q)).sum() / np.log(len(q)))


# ------------------------------------------------------------------ page
st.title("☁️ Cloud type classifier with explanations")
st.caption("Upload a sky photo. Every Keras application we compared in Notebook 3 classifies it, "
           "and we show what each one is looking at (Notebook 4's methods).")

with st.sidebar:
    st.header("Settings")
    smooth = st.slider("SmoothGrad samples", 8, 64, SMOOTH_DEFAULT, step=8,
                       help="Notebook 4 used 32. Fewer samples is faster but noisier.")
    show_checks = st.checkbox("Show confidence check", value=True)
    st.markdown("---")
    st.markdown("**Preprocessing**  \nResize: letterbox to 224 x 224, gray (128) padding, exactly as Notebook 2. "
                "The raw resized image goes straight to `image_explainers`, which applies each "
                "application's own `preprocess_input`.")

if not (MODEL_DIR / "P2_label_map.json").exists():
    st.error(f"Missing `{MODEL_DIR / 'P2_label_map.json'}`. Run Notebook 3's export cell first.")
    st.stop()
class_names = load_class_names()
n_classes = len(class_names)

with st.spinner("Loading models (first run only, this can take a minute)..."):
    JOINED, missing = load_models()
if missing:
    st.warning("No probe found for: " + ", ".join(APPS[k]["name"] for k in missing) +
               ". These are skipped. Finish Notebook 3 to include them.")
if not JOINED:
    st.error(f"No `P2_N3_*.keras` files found in `{MODEL_DIR}`. Run Notebook 3 first.")
    st.stop()

# ---- Upload
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
if uploaded is None:
    st.info("Upload a photo to begin. Try a clear cloud photo, a photo with no clouds, and a hard case.")
    st.stop()

try:
    pil = Image.open(uploaded)
    if FIX_EXIF_ORIENTATION:
        pil = ImageOps.exif_transpose(pil)
    pil.load()
except Exception as e:
    st.error(f"Could not read that file as an image: {e}")
    st.stop()

img = letterbox(pil)                                   # raw 0-255 uint8, 224 x 224 x 3
img_hash = hashlib.md5(img.tobytes()).hexdigest()

c1, c2 = st.columns(2)
with c1:
    st.image(pil, caption=f"Your upload ({pil.width} x {pil.height})", width=380)
with c2:
    st.image(img, caption=f"What the models receive: letterboxed to {IMG_SIZE} x {IMG_SIZE}", width=380)

# ---- Stage 1: probabilities for every model (fast)
probs = {}
for key, model in JOINED.items():
    try:
        probs[key] = get_probs(key, img_hash, n_classes, model, img)
    except Exception as e:
        st.error(f"{APPS[key]['name']}: could not classify ({e})")

if probs and show_checks:
    st.subheader("Summary")
    rows = []
    for key, p in probs.items():
        rows.append({"Model": APPS[key]["name"], "Size": APPS[key]["size"],
                     "Prediction": class_names[int(p.argmax())],
                     "Confidence": f"{p.max():.1%}",
                     "Uncertainty (0 to 1)": f"{normalized_entropy(p):.2f}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    preds = {class_names[int(p.argmax())] for p in probs.values()}
    if len(preds) > 1:
        st.warning("The models **disagree** about this image. Treat any single answer with caution.")
    else:
        st.success(f"All models agree on **{next(iter(preds))}**. Agreement is not proof: the models "
                   "share training data and could share the same blind spot.")
    weakest = min(p.max() for p in probs.values())
    if weakest < LOW_CONFIDENCE:
        st.warning(f"At least one model's top probability is below {LOW_CONFIDENCE:.0%}.")
    st.caption("The probabilities always add up to 100% over our five cloud types, so every model must pick "
               "one of them, even for an image that is not a cloud.")

# ---- Stage 2: one row per model (slower: explanation maps)
st.subheader("Every model, one row each")
for key, model in JOINED.items():
    if key not in probs:
        continue
    spec, p = APPS[key], probs[key]
    st.markdown(f"### {spec['name']} · {spec['size']}")
    st.markdown(f"**Predicted: {class_names[int(p.argmax())]}** ({p.max():.1%}) · "
                f"{model.count_params() / 1e6:.1f} M parameters (application + probe)")
    cols = st.columns([1.3, 1, 1, 1])
    with cols[0]:
        fig = prob_plot(p, class_names)
        st.pyplot(fig, clear_figure=True)
    try:
        with st.spinner(f"Computing {spec['name']} maps..."):
            overlays, secs = get_maps(key, img_hash, smooth, model, img)
        for col, (name, ov) in zip(cols[1:], overlays.items()):
            with col:
                st.image(ov, caption=name, width=MAP_WIDTH)
        st.caption(f"Maps took {secs:.0f}s (cached for this image and setting).")
    except Exception as e:
        st.error(f"{spec['name']}: could not compute maps ({e})")
    st.divider()

with st.expander("What this app does NOT know"):
    st.markdown(
        "- It only knows **five** cloud types and always answers with one of them.\n"
        "- A confident bar plot is not evidence of a correct answer. Softmax confidence is not "
        "calibrated and does not detect images that are not clouds.\n"
        "- The models were trained on a small scraped set that includes stock-photo watermarks and banners; "
        "check the maps for attention on those.\n"
        "- The training photos were mostly landscape web images. Unusual framing, night, or indoor shots are "
        "outside what we tested."
    )
