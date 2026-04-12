import streamlit as st
import tempfile
import os

from main import run_pipeline

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("Deepfake Video Detection System")
st.write("Upload a video to detect whether it is REAL or FAKE")

# Upload video
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("Run Detection"):

        with st.spinner("Processing video..."):

            result = run_pipeline(
                video_path=video_path,
                use_transformer=False  # keep stable
            )

        if result:

            st.success(f"Final Prediction: {result['final_label']}")

            st.metric("Confidence", f"{result['confidence']:.2f}")
            st.metric("CNN Score", f"{result['cnn_score']:.2f}")
            st.metric("HR Quality", f"{result['hr_quality_score']:.2f}")
            st.metric("Micro-expression", f"{result['micro_expression_score']:.2f}")

        # cleanup
        os.remove(video_path)