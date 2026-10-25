
import json
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import pipeline

ELEMENTS = ["logo", "product", "headline", "cta", "price"]

def extract_layout_features(layout):
    if isinstance(layout, str):
        layout = json.loads(layout)

    features = []

    for element in ELEMENTS:
        if element not in layout or layout[element] is None:
            features.extend([0, 0, 0, 0, 0])
            continue

        x1, y1, x2, y2 = layout[element]

        if not (0 <= x1 < x2 <= 1 and
                0 <= y1 < y2 <= 1):
            raise ValueError(f"Invalid bbox for {element}")

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        features.extend([
            center_x,
            center_y,
            width,
            height,
            1.0 
        ])

    return np.array(features, dtype=np.float32)

class AdLayoutDetector:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1

        self.detector = pipeline(task="zero-shot-object-detection", model="IDEA-Research/grounding-dino-base", device=device)

        self.labels = {
            "logo": "brand logo",
            "product": "product",
            "headline": "advertising headline text",
            "cta": "call to action button",
            "price": "price tag"
        }

    def detect(self, image: Image.Image):
        image = image.convert("RGB")
        width, height = image.size

        predictions = self.detector(image, candidate_labels=list(self.labels.values()), threshold=0.25)

        layout = {}
        confidence = {}

        for prediction in predictions:
            label = prediction["label"]

            element_type = next(
                (
                    key for key, prompt in self.labels.items()
                    if prompt == label
                ),
                None
            )

            if element_type is None:
                continue

            score = float(prediction["score"])

            # Keep highest confidence box per element
            if score <= confidence.get(element_type, -1):
                continue

            box = prediction["box"]

            x1 = max(0, min(1, box["xmin"] / width))
            y1 = max(0, min(1, box["ymin"] / height))
            x2 = max(0, min(1, box["xmax"] / width))
            y2 = max(0, min(1, box["ymax"] / height))

            if x2 <= x1 or y2 <= y1:
                continue

            layout[element_type] = [x1, y1, x2, y2]
            confidence[element_type] = score

        return layout

    def draw_boxes(self, image, layout):
        annotated = image.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)

        width, height = annotated.size

        for label, bbox in layout.items():
            x1, y1, x2, y2 = bbox

            coords = [
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height)
            ]

            draw.rectangle(
                coords, outline="red", width=3
            )

            draw.text(
                (coords[0], coords[1]),
                label,
                fill="red"
            )

        return annotated

    def extract(self, image):
        layout = self.detect(image)
        features = extract_layout_features(layout)

        return layout, features
