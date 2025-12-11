"""
Script to download SQuAD 2.0 dataset.
"""

import os
import urllib.request

SQUAD_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}


def download_squad(output_dir: str = "data"):
    """Download SQuAD 2.0 train and dev sets."""
    os.makedirs(output_dir, exist_ok=True)

    for split, url in SQUAD_URLS.items():
        filename = f"{split}-v2.0.json"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"{filename} already exists, skipping...")
            continue

        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")

    print("Download complete!")


if __name__ == "__main__":
    download_squad()
