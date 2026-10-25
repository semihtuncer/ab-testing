
import kagglehub
import pandas as pd
import json

from pathlib import Path
from huggingface_hub import snapshot_download

def download_marketing_ab_testing():
    dataset_name = "faviovaz/marketing-ab-testing"
    data_dir = Path("./data/ab_testing")
    path = data_dir / "marketing_AB.csv"

    if not data_dir.exists():
        print("Downloading dataset...")
        path = kagglehub.dataset_download(dataset_name, output_dir=str(data_dir))

    df = pd.read_csv(path)

    df["test group"] = df["test group"].str.lower()
    df["converted"] = df["converted"].astype(int)

    summary = df.groupby("test group")["converted"].agg(
        impressions="count",
        clicks="sum"
    )

    return summary

def download_ad_layouts(output_dir="data/ad_layouts"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = snapshot_download(repo_id="builditwithgk/adresyze-ad-layouts", repo_type="dataset", allow_patterns=["*.json"], local_dir=str(output_dir))

    records = []

    for file in Path(dataset_path).glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        layout = {}

        for element in data.get("elements", []):
            element_type = element["type"].lower()

            if element_type in ["logo", "product", "headline", "cta", "price"]:
                if element_type not in layout:
                    layout[element_type] = element["bbox"]

        records.append({
            "ad_id": Path(data["image_file"]).stem,
            "image_file": data["image_file"],
            "layout_json": json.dumps(layout)
            })

    df = pd.DataFrame(records)

    output_csv = output_dir / "layouts.csv"
    df.to_csv(output_csv, index=False)

    print(f"Downloaded {len(df)} layouts")
    print(f"Saved to {output_csv}")

    return df