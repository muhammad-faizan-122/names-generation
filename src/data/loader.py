import os
import requests
from typing import List


class DatasetLoader:
    """Handles dataset downloading and reading."""

    def __init__(self, url: str, local_path: str, max_retries: int = 3):
        self.url = url
        self.local_path = local_path
        self.max_retries = max_retries
        self.download_dataset()

    def download_dataset(self) -> None:
        """Downloads dataset if not available locally."""
        if os.path.exists(self.local_path):
            print(f"File already exists: {self.local_path}")
            return

        print(f"Downloading dataset from {self.url}...")
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.url, stream=True, timeout=10)
                response.raise_for_status()  # Raise an error for bad responses

                os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
                with open(self.local_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)

                print(f"Download complete: {self.local_path}")
                return
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt+1}/{self.max_retries} failed: {e}")

        print("Download failed after multiple attempts.")

    def read_dataset(self) -> List[str]:
        """Reads names from the dataset file."""
        try:
            with open(self.local_path, "r", encoding="utf-8") as file:
                names = [line.strip() for line in file]
            print(f"Total names loaded: {len(names)}")
            return names
        except FileNotFoundError:
            print(f"Error: File not found at {self.local_path}")
            return []
