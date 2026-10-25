import streamlit as st
import pandas as pd

from bayesian import compare_ads
from data_loader import download_marketing_ab_testing

from PIL import Image
from clip_encoder import AdImageEncoder
from layout_features import extract_layout_features
from layout_features import AdLayoutDetector

import joblib

@st.cache_resource
def load_clip():
    return AdImageEncoder()

@st.cache_resource
def load_layout_model():
    return joblib.load("layout_model.joblib")

@st.cache_resource
def load_layout_detector():
    return AdLayoutDetector()

st.set_page_config(
    page_title="A/B Testing for Ads",
)

st.title("Bayesian A/B Testing")
st.write("This is a simple Bayesian A/B testing tool. It will help you compare two versions of an ad and determine which one is better.")

st.divider()
image_a = None
image_b = None

col1, col2 = st.columns(2)
with col1:
    st.subheader("Ad A")
    image_a =st.file_uploader("Ad A Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

with col2:
    st.subheader("Ad B")
    image_b = st.file_uploader("Ad B Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

st.divider()
st.subheader("Compare Ads")

def predict_layout_ctr(model, layout):
    features = extract_layout_features(layout)
    features = features.reshape(1, -1)

    return float(model.predict_proba(features)[0, 1])

if image_a is not None and image_b is not None:
    image_a_pil = Image.open(image_a).convert("RGB")
    image_b_pil = Image.open(image_b).convert("RGB")

    if st.button("Detect Layout"):
        layout_detector = load_layout_detector()
        layout_a, features_a = layout_detector.extract(image_a_pil)
        layout_b, features_b = layout_detector.extract(image_b_pil)

        st.session_state["layout_a"] = layout_a
        st.session_state["layout_b"] = layout_b
        st.session_state["features_a"] = features_a
        st.session_state["features_b"] = features_b

    if "layout_a" in st.session_state and "layout_b" in st.session_state:
        layout_detector = load_layout_detector()
        layout_a = st.session_state["layout_a"]
        layout_b = st.session_state["layout_b"]
        features_a = st.session_state["features_a"]
        features_b = st.session_state["features_b"]

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                layout_detector.draw_boxes(image_a_pil, layout_a),
                caption="Detected Layout A"
            )
            st.json(layout_a)

        with col2:
            st.image(
                layout_detector.draw_boxes(image_b_pil, layout_b),
                caption="Detected Layout B"
            )
            st.json(layout_b)

        st.write("Feature A shape:", features_a.shape)
        st.write("Feature B shape:", features_b.shape)

        impressions = st.number_input(
            "Impressions per ad",
            min_value=1,
            value=10_000,
            step=100,
        )

        if st.button("Analyze Images"):
            if "layout_a" not in st.session_state or "layout_b" not in st.session_state:
                st.warning("Detect the layouts before analyzing the images.")
                st.stop()

            layout_a = st.session_state["layout_a"]
            layout_b = st.session_state["layout_b"]

            encoder = load_clip()
            layout_model = load_layout_model()

            result_clip = encoder.compare(image_a_pil, image_b_pil)

            ctr_a = predict_layout_ctr(layout_model, layout_a)
            ctr_b = predict_layout_ctr(layout_model, layout_b)

            clicks_a = min(impressions, int(round(ctr_a * impressions)))
            clicks_b = min(impressions, int(round(ctr_b * impressions)))

            result = compare_ads(
                impressions_a=impressions,
                clicks_a=clicks_a,
                impressions_b=impressions,
                clicks_b=clicks_b,
            )

            st.metric(
                "Visual Similarity",
                f"{result_clip['similarity']:.3f}"
            )

            metric_a, metric_b, metric_win = st.columns(3)
            metric_a.metric("Layout CTR A", f"{ctr_a:.3%}")
            metric_b.metric("Layout CTR B", f"{ctr_b:.3%}")
            metric_win.metric("P(B beats A)", f"{result['prob_b_wins']:.1%}")

            low_a, high_a = result["credible_interval_a"]
            low_b, high_b = result["credible_interval_b"]
            low_diff, high_diff = result["difference_interval"]

            st.write(
                f"95% CTR interval A: {low_a:.3%} to {high_a:.3%}"
            )
            st.write(
                f"95% CTR interval B: {low_b:.3%} to {high_b:.3%}"
            )
            st.write(
                f"95% uplift interval (B − A): {low_diff:+.3%} to {high_diff:+.3%}"
            )
            st.write(
                f"Expected loss if you choose A: {result['expected_loss_choose_a']:.3%}"
            )
            st.write(
                f"Expected loss if you choose B: {result['expected_loss_choose_b']:.3%}"
            )
