import hashlib
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


API_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 60

st.set_page_config(
    page_title="Digit Studio",
    page_icon="✳",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root { color-scheme: dark; }
    .stApp {
        background:
            radial-gradient(ellipse at 18% 0%, rgba(141,124,255,.13), transparent 34rem),
            radial-gradient(ellipse at 95% 55%, rgba(196,240,106,.07), transparent 28rem),
            #11131b;
        color: #f3f2f8;
        font-family: 'DM Sans', sans-serif;
    }
    .block-container { max-width: 1030px; padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.7px; }
    .brand { display:flex; align-items:center; gap:11px; font:700 17px 'Space Grotesk',sans-serif; letter-spacing:-.7px; }
    .brand-mark { display:grid; place-items:center; width:35px; height:35px; border-radius:11px; background:#c4f06a; color:#171a13; font-size:20px; }
    .brand-mark b { color:#7770e9; }
    .brand-muted { color:#9395a5; font-weight:400; }
    .eyebrow { margin-top:3rem; color:#c4f06a; font-size:10px; font-weight:700; letter-spacing:1.7px; text-align:center; }
    .hero-title { margin:.8rem 0 .45rem; font:600 clamp(2.4rem,6vw,3.8rem)/1.08 'Space Grotesk',sans-serif; letter-spacing:-2.7px; text-align:center; }
    .hero-title span { color:#8d8e9f; }
    .hero-copy { margin-bottom:2rem; color:#a4a5b2; text-align:center; font-size:14px; }
    .card {
        height:100%; min-height:450px; padding:24px;
        border:1px solid rgba(255,255,255,.08); border-radius:18px;
        background:rgba(25,28,39,.88); box-shadow:0 20px 70px rgba(0,0,0,.15);
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color:rgba(255,255,255,.08) !important;
        border-radius:18px !important;
        background:rgba(25,28,39,.88) !important;
    }
    .section-label { margin-bottom:5px; color:#8f91a4; font-size:9px; font-weight:700; letter-spacing:1.5px; }
    .card-title { margin:0 0 1.1rem; font:600 20px 'Space Grotesk',sans-serif; }
    .result-number { color:#c4f06a; font:600 100px/1 'Space Grotesk',sans-serif; text-align:center; text-shadow:0 0 48px rgba(196,240,106,.13); }
    .result-caption { margin:1.2rem 0 .4rem; color:#8f91a4; font-size:10px; font-weight:700; letter-spacing:1.6px; text-align:center; }
    .result-subtitle { color:#a7a8b5; font-size:12px; }
    .empty-state { padding:5rem 1rem; color:#9293a4; text-align:center; }
    .empty-icon { color:#383746; font:600 90px 'Space Grotesk',sans-serif; line-height:1; }
    .stButton > button {
        min-height:44px; border:0; border-radius:10px; background:#c4f06a; color:#181b12;
        font-weight:700; transition:transform .18s, background .18s;
    }
    .stButton > button:hover { border:0; background:#b6e45a; color:#181b12; transform:translateY(-1px); }
    .stButton > button:focus { box-shadow:0 0 0 2px #11131b, 0 0 0 4px #c4f06a; }
    [data-testid="stFileUploader"] { padding:.6rem; border:1px dashed rgba(255,255,255,.15); border-radius:12px; background:rgba(255,255,255,.025); }
    [data-testid="stMetric"] { padding:12px 14px; border:1px solid rgba(255,255,255,.07); border-radius:11px; background:rgba(255,255,255,.025); }
    [data-testid="stMetricLabel"] { color:#9b9cac; }
    [data-testid="stMetricValue"] { color:#c4f06a; }
    div[role="radiogroup"] { gap:1.4rem; }
    .footer { margin-top:1.4rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,.07); color:#727484; font-size:10px; }
    @media (max-width: 700px) {
        .block-container { padding:1rem 1rem 2rem; }
        .eyebrow { margin-top:2rem; }
        .card { min-height:0; padding:19px; }
        .result-number { font-size:80px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="brand"><span class="brand-mark">D<b>.</b></span>'
    '<span>digit<span class="brand-muted">studio</span></span></div>',
    unsafe_allow_html=True,
)

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    health_response.raise_for_status()
    model_ready = health_response.json().get("model_loaded", False)
except (requests.RequestException, ValueError):
    model_ready = False

if model_ready:
    st.success("CNN model connected", icon="✅")
else:
    st.warning(
        f"FastAPI is unavailable or its model is not loaded. Start the backend at {API_URL}.",
        icon="⚠️",
    )

st.markdown(
    '<div class="eyebrow">✳ &nbsp; YOUR HANDWRITING, UNDERSTOOD</div>'
    '<div class="hero-title">A little number.<br><span>A lot of intelligence.</span></div>'
    '<div class="hero-copy">Draw a digit and let the neural network take its best guess.</div>',
    unsafe_allow_html=True,
)

draw_column, result_column = st.columns(2, gap="medium")

with draw_column:
    with st.container(border=True):
        st.markdown(
            '<div class="section-label">THE CANVAS</div>'
            '<div class="card-title">Write a digit</div>',
            unsafe_allow_html=True,
        )
        mode = st.radio("Choose input", ["Draw", "Upload"], horizontal=True, label_visibility="collapsed")
        image_bytes = None

        if mode == "Draw":
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=18,
                stroke_color="#ffffff",
                background_color="#090a0f",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="digit_canvas",
                display_toolbar=True,
                update_streamlit=True,
            )
            if (
                canvas_result.image_data is not None
                and canvas_result.json_data
                and canvas_result.json_data.get("objects")
            ):
                drawn_image = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
                output = BytesIO()
                drawn_image.save(output, format="PNG")
                image_bytes = output.getvalue()
        else:
            uploaded_file = st.file_uploader(
                "Upload one digit image",
                type=["png", "jpg", "jpeg", "webp", "bmp"],
                help="Use a clear image with one digit. Maximum size: 5 MB.",
                max_upload_size=5,
            )
            if uploaded_file is not None:
                image_bytes = uploaded_file.getvalue()
                st.image(image_bytes, caption="Image to recognize", width=220)

        input_signature = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
        if st.session_state.get("prediction_input") != input_signature:
            st.session_state.pop("prediction", None)
            st.session_state["prediction_input"] = input_signature

        predict_clicked = st.button("Recognize digit  ↗", type="primary", use_container_width=True)
        if mode == "Draw":
            st.caption("Draw a single centered digit, then click recognize.")
        else:
            st.caption("Best results with one centered, clearly written digit.")

with result_column:
    with st.container(border=True):
        st.markdown(
            '<div class="section-label">THE PREDICTION</div>'
            '<div class="card-title">What I see</div>',
            unsafe_allow_html=True,
        )

        if predict_clicked:
            if image_bytes is None:
                st.error("Draw or upload a digit before asking for a prediction.")
            elif not model_ready:
                st.error("The CNN backend is not ready. Start FastAPI and ensure its model file exists.")
            else:
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": ("digit.png", image_bytes, "image/png")},
                        timeout=REQUEST_TIMEOUT,
                    )
                    response.raise_for_status()
                    result = response.json()
                    probabilities = result["probabilities"]
                    digit = int(result["digit"])
                    confidence = float(result["confidence"])
                    if len(probabilities) != 10 or not 0 <= digit <= 9:
                        raise ValueError("The prediction API returned an invalid response.")
                    st.session_state["prediction"] = {
                        "digit": digit,
                        "confidence": confidence,
                        "probabilities": probabilities,
                    }
                except requests.Timeout:
                    st.error("The prediction request timed out. Please try again.")
                except requests.HTTPError as error:
                    try:
                        detail = error.response.json().get("detail", str(error))
                    except ValueError:
                        detail = str(error)
                    st.error(f"Prediction failed: {detail}")
                except (requests.RequestException, KeyError, TypeError, ValueError) as error:
                    st.error(f"Could not get a prediction: {error}")

        prediction = st.session_state.get("prediction")
        if prediction is not None:
            st.markdown(
                f'<div class="result-caption">MY BEST GUESS IS</div>'
                f'<div class="result-number">{prediction["digit"]}</div>',
                unsafe_allow_html=True,
            )
            st.metric("Confidence", f'{prediction["confidence"]:.1%}')
            st.markdown("**All possibilities**")
            for start in range(0, 10, 2):
                probability_columns = st.columns(2)
                for column, current_digit in zip(probability_columns, range(start, start + 2)):
                    probability = float(prediction["probabilities"][current_digit])
                    column.caption(f"Digit {current_digit} · {probability:.1%}")
                    column.progress(max(0.0, min(1.0, probability)))
        elif not predict_clicked:
            st.markdown(
                '<div class="empty-state"><div class="empty-icon">?</div>'
                '<p>Your answer is one sketch away.</p>'
                '<p>Write a number from 0 to 9, then hit recognize.</p></div>',
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="footer">✳ &nbsp; A small demo powered by your CNN'
    '<span style="float:right">Draw. Predict. Repeat.</span></div>',
    unsafe_allow_html=True,
)
