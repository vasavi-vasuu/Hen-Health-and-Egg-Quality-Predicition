# Hen Health & Egg Quality Prediction

End-to-end image classification project with two models:
1. **Hen Health** — classifies a hen photo as `healthy` or `diseased`
2. **Egg Quality** — classifies an egg photo as `good_quality` or `poor_quality`

## 🆕 Named condition, not just healthy/diseased
When a hen is predicted `diseased` (or an egg `poor_quality`), the app now also
shows a **likely condition name** — e.g. *Fowl Pox*, *Coccidiosis*, *Avian
Influenza (suspected)* for hens, or *Cracked Shell*, *Blood Spot*, *Shell
Staining* for eggs — implemented in `disease_reference.py`.

This is a **rule-based heuristic lookup**, not a second trained model: the
dataset only has `healthy`/`diseased` and `good`/`poor` labels, so there's
nothing for a model to learn specific disease names from. Instead, once the
main model flags something as diseased/poor, `disease_reference.py` looks at
visual cues (dark-spot ratio, texture roughness, colour balance) and matches
them to a small reference table of well-known poultry disease / egg-defect
symptoms. It's shown with a clear on-screen disclaimer and is meant to help
narrow things down, not replace a vet/lab diagnosis.

## ⚠️ Read this first: trained on 5 real photos per class

This project was retrained on **5 real photos per class** (20 total) provided
directly, replacing the earlier synthetic demo images. Two things were done to
make this workable:

1. **Augmentation** (`augment_real_images.py`) — each of the 5 real photos is
   expanded into 10 variants (rotation, flip, brightness/contrast/color jitter,
   zoom-crop) → 50 images per class in `dataset/`. The untouched originals live
   in `dataset_original/`.
2. **Leave-one-source-photo-out cross-validation** (`retrain_on_real_images.py`,
   also in `Lab 12_Retraining on Real Photos.ipynb`) — evaluates the model on a
   held-out *real* photo it has never seen any variant of, instead of a random
   split that would let near-duplicate augmented siblings leak between train
   and test and produce a fake high accuracy number.

**Honest result: ~55-65% cross-validated accuracy** (50% = random guessing),
saved in `model_accuracies.pkl` and shown as a warning banner in the app.
This is not a bug — it's the expected, unavoidable result of training on only
5 real examples per class. See `Lab 12_Retraining on Real Photos.ipynb` for
the full explanation and per-fold breakdown.

### How to actually improve accuracy
- **Add more real photos** — 30-50+ genuinely different photos per class
  (different birds/eggs, different lighting/backgrounds), not just more
  augmented copies of the same 5. Drop new photos into `dataset_original/...`
  and re-run `augment_real_images.py` then `retrain_on_real_images.py`.
- **Keep photo conditions consistent** — similar framing/lighting across
  photos reduces irrelevant variation the model has to filter out.
- **Upgrade to a pretrained CNN** (e.g. MobileNetV2 transfer learning) once
  you have 30+ images per class — pretrained visual features generalize much
  better than the handcrafted features used here, even with modest data.

## Files
- `dataset_original/hens/{healthy,diseased}/`, `dataset_original/eggs/{good_quality,poor_quality}/` — your 5 real photos per class, untouched
- `dataset/...` — same structure, augmented to 50 images/class (used for training)
- `augment_real_images.py` — regenerate `dataset/` from `dataset_original/` after adding more photos
- `retrain_on_real_images.py` — retrain both models with honest group-aware cross-validation
- `sample_images/` — 4 real images copied out for quick manual testing of `app.py`
- `Lab 11_End-to-End-Hen Health and Egg Quality Prediction.ipynb` — original pipeline walkthrough (synthetic images)
- `Lab 12_Retraining on Real Photos.ipynb` — retraining on your real photos, with full honest evaluation
- `Practice_HenHealth_Classification.ipynb` / `end to end egg quality classification practice.ipynb` — lighter single-task notebooks (synthetic images)
- `app.py` — Streamlit app, two tabs, shows the honest accuracy warning banner and the named condition/defect
- `disease_reference.py` — rule-based lookup that turns a "diseased"/"poor_quality" result into a likely condition name (Fowl Pox, Coccidiosis, Cracked Shell, etc.)
- `hen_health_model.pkl`, `hen_health_scaler.pkl` — deployed hen classifier + scaler (trained on real+augmented data)
- `egg_quality_model.pkl`, `egg_quality_scaler.pkl` — deployed egg classifier + scaler (trained on real+augmented data)
- `model_accuracies.pkl` — cross-validated accuracy per model, read by `app.py`
- `requirements.txt` — run `pip install -r requirements.txt`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then upload an image from `sample_images/` (or your own) to test each tab.

## Add more photos and retrain
```bash
# 1. Drop new real photos into the matching dataset_original/ subfolder
# 2. Regenerate the augmented training set
python augment_real_images.py
# 3. Retrain and re-save the models (prints honest cross-validated accuracy)
python retrain_on_real_images.py
```

## Approach
Rather than training a CNN from raw pixels (which needs a much bigger dataset
and ideally a GPU), this project extracts a compact handcrafted feature vector
per image — color histogram, brightness stats, GLCM texture
(contrast/homogeneity/energy), and a dark-spot ratio — then feeds it into a
Random Forest classifier. This is fast and dependency-light, but its accuracy
ceiling is fundamentally limited by dataset size — see the warning above.

## ✨ Generative AI Feature

This updated version adds an optional **Generative AI report** using the
Google Gemini API. The existing ML prediction remains the main classifier;
Gemini turns that result into a student-friendly explanation, observation
points, and next-step guidance.

### What the GenAI button does

After predicting a hen or egg:
1. The app takes the ML result and confidence.
2. It also takes the existing rule-based reference label/note.
3. Gemini generates a short Markdown report.
4. The report explains the result and gives practical educational guidance.
5. The prompt explicitly tells Gemini not to present the output as a confirmed
   veterinary diagnosis or laboratory food-safety result.

### Free API setup

Google AI Studio provides Gemini API access with free-tier availability that
can vary by account, model, and current quota. Create an API key in Google AI
Studio, then keep the key outside your source code.

For local use, set:

```text
GEMINI_API_KEY=your_key_here
```

or use the included `.env.example` as a reference.

For Streamlit Cloud, add the secret in your app's **Settings → Secrets**:

```toml
GEMINI_API_KEY = "your_key_here"
GEMINI_MODEL = "gemini-2.5-flash"
```

The code reads the key from Streamlit Secrets first and then from the
`GEMINI_API_KEY` environment variable.

### Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

If no Gemini key is configured, the normal ML prediction still works; only the
GenAI report button will show a setup message.

### Important

Do not put a real API key directly inside `app.py`, upload it to GitHub, or
share it in screenshots. Free-tier quotas and model availability can change,
so an API error can be caused by quota or account access rather than by the
project code.

## Hardcoded API-Key Placeholder

Open `gemini_ai.py` and replace:

```python
GEMINI_API_KEY = "PASTE_YOUR_GEMINI_API_KEY_HERE"
```

with your own Gemini API key.

The code then uses:

```python
client = genai.Client(api_key=GEMINI_API_KEY)
```

Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

This ZIP contains only a placeholder, not a real secret. Do not commit a real
API key to GitHub or share a ZIP containing the real key.

## 🛠️ Important Fix: Generative AI Button

The Generative AI button uses Streamlit `session_state`, so the prediction is
saved before the AI button is clicked. This is important because Streamlit
reruns the script whenever a button is pressed.

Correct flow:

1. Upload image.
2. Click **Predict Hen Health** or **Predict Egg Quality**.
3. The prediction is saved.
4. Click **Generate AI Explanation & Care Guidance**.
5. Gemini generates and displays the AI report below the button.

If the AI report still does not appear, check the red error message below the
button. Common causes are an invalid/revoked API key, unavailable model, quota
limit, or missing `google-genai` package.
