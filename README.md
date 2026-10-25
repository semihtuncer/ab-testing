## A/B Testing on Ads

Compare two ad images in a Streamlit app.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

## Files

- `main.py`: Streamlit wrapper for UI
- `bayesian.py`: Beta-Binomial comparison of two ads. Returns the probability that B beats A, credible intervals, and the expected loss of choosing either ad.
- `layout_features.py`: Detects logo, product, headline, CTA, and price with Grounding DINO, and turns each box into center, size, and presence features.
- `train_layout.py`: Fits a logistic regression from those layout features to click-through rate and saves `layout_model.joblib`.
- `data_loader.py`: Downloads the marketing A/B test summary and the ad-layout JSON annotations.
- `clip_encoder.py`: Encodes ad images with CLIP ViT-B/32 and scores how visually similar two ads are.

## Flaws

- The layout dataset has bounding boxes only. It has no clicks or impressions, so `train_layout.py` cannot learn a real click through rate from it.
- The Bayesian result treats `round(predicted CTR × impressions)` as if those clicks were observed. Raising the impression count makes the probability look more certain without any new evidence.
- Grounding DINO is used zero shot. The layout files do not include the ad images, so the detector is not fine tuned on this data.
- CLIP similarity measures how alike the pictures look. It does not measure which ad gets more clicks. No appropriate dataset.