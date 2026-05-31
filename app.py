import os
import pickle
import urllib.request

import numpy as np
import streamlit as st
from PIL import Image


MODEL_PATH = "asl_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"

GOOGLE_DRIVE_FILE_ID = "1NdVNuR6kpzQNL_ehgSHgpSz19ZYGP_AW"


st.set_page_config(
    page_title="Alphabet Hand Sign Recognition",
    page_icon="",
    layout="centered",
)


st.markdown(
    """
    <style>
        :root {
            --ink: #102033;
            --muted: #607086;
            --blue: #2563eb;
            --teal: #0f766e;
            --amber: #f59e0b;
            --panel: #ffffff;
            --line: #dbe3ef;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.16), transparent 32rem),
                radial-gradient(circle at bottom right, rgba(15, 118, 110, 0.14), transparent 30rem),
                linear-gradient(135deg, #f7fbff 0%, #f8fafc 45%, #fff7ed 100%);
        }

        .main .block-container {
            max-width: 980px;
            padding-top: 2.5rem;
            padding-bottom: 3rem;
        }

        .hero {
            border: 1px solid rgba(37, 99, 235, 0.18);
            border-radius: 8px;
            padding: 1.65rem;
            background:
                linear-gradient(135deg, rgba(37, 99, 235, 0.11), rgba(15, 118, 110, 0.08)),
                var(--panel);
            margin-bottom: 1.25rem;
            box-shadow: 0 14px 35px rgba(16, 32, 51, 0.08);
        }

        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.1rem;
            line-height: 1.15;
            letter-spacing: 0;
            color: var(--ink);
        }

        .hero p {
            margin: 0;
            color: var(--muted);
            font-size: 1rem;
        }

        .result-card {
            border: 1px solid rgba(15, 118, 110, 0.18);
            border-radius: 8px;
            padding: 1.25rem;
            background:
                linear-gradient(135deg, rgba(15, 118, 110, 0.12), rgba(245, 158, 11, 0.12)),
                #ffffff;
            margin-top: 1rem;
            box-shadow: 0 12px 28px rgba(16, 32, 51, 0.08);
        }

        .prediction {
            font-size: 2.25rem;
            font-weight: 700;
            margin: 0.1rem 0 0.25rem 0;
            color: var(--teal);
        }

        .muted {
            color: var(--muted);
            font-size: 0.95rem;
        }

        [data-testid="stFileUploader"] section,
        [data-testid="stCameraInput"] {
            border-color: var(--line);
            background: rgba(255, 255, 255, 0.82);
            border-radius: 8px;
        }

        .stRadio [role="radiogroup"] {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.35rem 0.5rem;
        }

        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, var(--teal), var(--blue));
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            color: var(--blue);
        }

        .notice-card {
            border-left: 5px solid var(--amber);
            border-radius: 8px;
            padding: 1rem 1.15rem;
            background: rgba(255, 251, 235, 0.92);
            color: #7c4a03;
            margin: 1rem 0 1.25rem 0;
            box-shadow: 0 8px 20px rgba(16, 32, 51, 0.06);
        }

        .notice-card strong {
            color: #92400e;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def download_model_from_google_drive(file_id, output_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as error:
        raise RuntimeError(
            "Could not download the model from Google Drive. Make sure the file "
            "sharing is set to 'Anyone with the link' and the file ID is correct."
        ) from error

    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1024:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(
            "The downloaded model file looks invalid. Recheck your Google Drive "
            "sharing settings and file ID."
        )


@st.cache_resource(show_spinner=False)
def load_model_and_encoder():
    try:
        from tensorflow import keras
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "TensorFlow is not installed in the Python environment running this app. "
            "Start the app with: .\\venv_asl\\Scripts\\python.exe -m streamlit run app.py"
        ) from error

    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(
            f"Missing {LABEL_ENCODER_PATH}. Upload it to your GitHub repository."
        )

    if not os.path.exists(MODEL_PATH):
        if GOOGLE_DRIVE_FILE_ID == "PASTE_YOUR_GOOGLE_DRIVE_FILE_ID_HERE":
            raise ValueError(
                "Add your Google Drive file ID to GOOGLE_DRIVE_FILE_ID in app.py."
            )

        download_model_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)

    model = keras.models.load_model(MODEL_PATH)

    with open(LABEL_ENCODER_PATH, "rb") as file:
        label_encoder = pickle.load(file)

    return model, label_encoder


def preprocess_image(image):
    resized = image.convert("RGB").resize((64, 64), Image.Resampling.LANCZOS)
    normalized = np.array(resized).astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def predict_sign(image):
    model, label_encoder = load_model_and_encoder()
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = float(predictions[predicted_index])

    top_indices = np.argsort(predictions)[-3:][::-1]
    top_predictions = [
        (
            label_encoder.inverse_transform([int(index)])[0],
            float(predictions[index]),
        )
        for index in top_indices
    ]

    return predicted_label, confidence, top_predictions


st.markdown(
    """
    <section class="hero">
        <h1>Alphabet Hand Sign Recognition</h1>
        <p>Upload a clear hand sign image and the model will predict the ASL alphabet class.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice-card">
        <strong>Accuracy notice:</strong> This project is a learning/demo model.
        Predictions are not 100% accurate and may have low accuracy, especially
        with blurry images, unusual lighting, backgrounds, or hand positions
        that are different from the training dataset.
    </div>
    """,
    unsafe_allow_html=True,
)

input_mode = st.radio(
    "Choose input method",
    ["Upload image", "Use camera"],
    horizontal=True,
)

uploaded_image = None

if input_mode == "Upload image":
    uploaded_image = st.file_uploader(
        "Upload a hand sign image",
        type=["jpg", "jpeg", "png"],
    )
else:
    uploaded_image = st.camera_input("Take a hand sign photo")

if uploaded_image is None:
    st.info("Add an image to start recognition.")
    st.stop()

image = Image.open(uploaded_image)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.image(image, caption="Input image", use_container_width=True)

with right_col:
    with st.spinner("Analyzing hand sign..."):
        try:
            label, confidence, top_predictions = predict_sign(image)
        except Exception as error:
            st.error(str(error))
            st.stop()

    st.markdown(
        f"""
        <div class="result-card">
            <div class="muted">Predicted sign</div>
            <div class="prediction">{label}</div>
            <div class="muted">Confidence: {confidence:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(confidence, 1.0))

st.subheader("Top Predictions")

for prediction_label, prediction_confidence in top_predictions:
    st.metric(
        label=str(prediction_label),
        value=f"{prediction_confidence:.2%}",
    )
