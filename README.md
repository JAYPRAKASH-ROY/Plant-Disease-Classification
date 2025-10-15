
# 🌿 Plant Disease Classification — Streamlit App

This is a Streamlit web app to serve a plant disease image classifier trained via transfer learning.
It supports both **Keras (.keras)** and **TensorFlow Lite (.tflite)** models.

## Files
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- *(Optional)* `plant_disease_classifier_final.keras` — your trained Keras model
- *(Optional)* `plant_disease_classifier_fp16.tflite` — your TFLite model
- *(Optional)* `labels.txt` — one class name per line in the training order

## Local Run
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

## Provide Labels
- Paste comma-separated labels in the sidebar **or**
- Upload `labels.txt` (one class per line).

## Deploy on Streamlit Community Cloud
1. Push these files and your model file to a public GitHub repo.
2. Go to https://share.streamlit.io
3. Create a new app → point to `app.py` in your repo.
4. Set **Python version** to 3.10+ if needed.
5. (Optional) If using large Keras model, consider using the `.tflite` file to speed up cold starts.

## Notes
- Grad-CAM is only available for the Keras backend.
- TFLite backend is faster and lighter but does not support Grad-CAM in this demo.
- Ensure labels order matches the model's training class order.
