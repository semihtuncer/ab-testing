
import torch
import open_clip
import numpy as np

from PIL import Image

class AdImageEncoder:
    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu")

        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai"
            )
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, image: Image.Image):
        image = image.convert("RGB")
        tensor = (self.preprocess(image).unsqueeze(0).to(self.device))

        with torch.inference_mode():
            features = self.model.encode_image(tensor)
            features = features / features.norm(
                dim=-1, keepdim=True
            )

        return features.cpu().numpy()[0]

    def compare(self, image_a, image_b):
        embedding_a = self.encode(image_a)
        embedding_b = self.encode(image_b)

        similarity = float(
            np.dot(embedding_a, embedding_b)
        )

        return {
            "embedding_a": embedding_a,
            "embedding_b": embedding_b,
            "similarity": similarity
        }

    def classify_image(self, image, descriptions):
        image = image.convert("RGB")

        image_tensor = (self.preprocess(image).unsqueeze(0).to(self.device))

        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        text_tokens = tokenizer(descriptions).to(self.device)

        with torch.inference_mode():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True
            )

            similarities = (image_features @ text_features.T).squeeze(0)

            scores = (100 * similarities).softmax(dim=0)

        return {
            description: float(score)
            for description, score in zip(
                descriptions, scores.cpu().tolist() 
            )
        }
