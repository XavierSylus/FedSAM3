#!/usr/bin/env python3
"""
Generate semantic text features for BraTS 2020 samples using DeepSeek API and BERT embeddings.

Dependencies:
pip install openai transformers torch nibabel numpy tqdm

Usage:
export DEEPSEEK_API_KEY="your_api_key_here"
python scripts/generate_semantic_text_features.py --data_root /path/to/BraTS2020 --force_overwrite
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib
from tqdm import tqdm
from openai import OpenAI

# Lazy import torch and transformers to speed up startup
import torch
from transformers import AutoTokenizer, AutoModel


class TextFeatureGenerator:
    def __init__(self, api_key: str, max_retries: int = 3):
        """Initialize the generator with DeepSeek API and BERT model."""
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.max_retries = max_retries

        # Initialize BERT model for feature extraction
        print("Loading BERT model (bert-base-uncased)...")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()

        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model.to(self.device)
        print(f"Using device: {self.device}")

    def calculate_tumor_size_category(self, mask_path: Path) -> Tuple[str, int]:
        """Calculate tumor size category from segmentation mask."""
        try:
            # Load the segmentation mask
            mask_img = nib.load(str(mask_path))
            mask_data = mask_img.get_fdata()

            # Count non-zero voxels (tumor region)
            tumor_voxels = np.count_nonzero(mask_data)

            # Define thresholds (adjust based on your data characteristics)
            if tumor_voxels < 5000:
                category = "small"
            elif tumor_voxels < 20000:
                category = "medium"
            else:
                category = "large"

            return category, tumor_voxels
        except Exception as e:
            print(f"Warning: Failed to read mask {mask_path}: {e}")
            return "medium", 0  # Default fallback

    def generate_medical_text(self, tumor_size: str, voxel_count: int) -> str:
        """Call DeepSeek API to generate medical report text with retry mechanism."""
        prompt = (
            f"Please write a concise, one-sentence MRI radiologist report for a brain tumor patient. "
            f"The tumor size is {tumor_size}. No conversational filler, just the medical sentence."
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a medical radiologist writing concise MRI reports."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=100
                )

                medical_text = response.choices[0].message.content.strip()
                return medical_text

            except Exception as e:
                wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                if attempt < self.max_retries - 1:
                    print(f"  API call failed (attempt {attempt+1}/{self.max_retries}): {e}")
                    print(f"  Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"  API call failed after {self.max_retries} attempts: {e}")
                    # Fallback to rule-based text
                    return f"MRI scan reveals a {tumor_size} brain tumor with approximately {voxel_count} voxels."

        return f"MRI scan reveals a {tumor_size} brain tumor."

    def text_to_features(self, text: str) -> np.ndarray:
        """Convert text to 768-dim BERT features (CLS token) with L2 normalization."""
        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use [CLS] token representation (first token)
            cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # L2 normalization
        features = cls_features.squeeze()
        features = features / (np.linalg.norm(features) + 1e-8)

        assert features.shape == (768,), f"Expected shape (768,), got {features.shape}"
        return features.astype(np.float32)

    def is_random_feature(self, npy_path: Path) -> bool:
        """Check if the feature file contains random noise (heuristic check)."""
        try:
            if not npy_path.exists():
                return True

            features = np.load(npy_path)

            # Check if features look like random noise
            # Random noise typically has:
            # 1. Values distributed around 0 with std ~ 1/sqrt(768)
            # 2. No clear structure
            mean_val = np.abs(np.mean(features))
            std_val = np.std(features)

            # If mean is very close to 0 and std is close to 1/sqrt(768) ≈ 0.036
            # it's likely random noise
            if mean_val < 0.01 and 0.02 < std_val < 0.05:
                return True

            # Another check: if norm is very close to 1, it might be normalized random
            norm = np.linalg.norm(features)
            if 0.99 < norm < 1.01 and mean_val < 0.05:
                # Could be normalized random noise
                return True

            return False
        except:
            return True


def find_samples_to_process(data_root: Path) -> list:
    """Find all BraTS samples that need text feature generation."""
    samples = []

    # Search for segmentation files (support both .nii and .nii.gz)
    seg_patterns = ["*_seg.nii.gz", "*_seg.nii"]
    seg_files = []
    for pattern in seg_patterns:
        seg_files.extend(data_root.rglob(pattern))

    for seg_file in seg_files:
        sample_dir = seg_file.parent
        # Handle both .nii.gz and .nii extensions
        if seg_file.name.endswith("_seg.nii.gz"):
            sample_name = seg_file.name.replace("_seg.nii.gz", "")
        else:
            sample_name = seg_file.name.replace("_seg.nii", "")

        # Construct expected text feature path
        text_feature_path = sample_dir / f"{sample_name}_flair_text.npy"

        samples.append({
            'seg_file': seg_file,
            'text_feature_path': text_feature_path,
            'sample_name': sample_name
        })

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic text features for BraTS 2020 samples"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='./data/BraTS2020',
        help='Root directory containing BraTS2020 data'
    )
    parser.add_argument(
        '--force_overwrite',
        action='store_true',
        help='Force overwrite all existing features (skip random detection)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (for testing)'
    )

    args = parser.parse_args()

    # Validate API key
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY environment variable is not set!")
        print("Please run: export DEEPSEEK_API_KEY='your_api_key_here'")
        sys.exit(1)

    # Validate data root
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: Data root directory does not exist: {data_root}")
        sys.exit(1)

    # Find samples to process
    print(f"Scanning for samples in: {data_root}")
    samples = find_samples_to_process(data_root)

    if not samples:
        print("ERROR: No samples found! Check your data_root path.")
        sys.exit(1)

    print(f"Found {len(samples)} samples")

    # Limit samples if specified
    if args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Processing first {args.max_samples} samples")

    # Initialize generator
    generator = TextFeatureGenerator(api_key=api_key)

    # Process samples
    processed_count = 0
    skipped_count = 0
    error_count = 0

    print("\nProcessing samples...")
    for sample_info in tqdm(samples, desc="Generating features", unit="sample"):
        seg_file = sample_info['seg_file']
        text_feature_path = sample_info['text_feature_path']
        sample_name = sample_info['sample_name']

        # Check if we should skip this sample
        if not args.force_overwrite and text_feature_path.exists():
            if not generator.is_random_feature(text_feature_path):
                skipped_count += 1
                continue

        try:
            # Step 1: Calculate tumor size
            tumor_size, voxel_count = generator.calculate_tumor_size_category(seg_file)

            # Step 2: Generate medical text via DeepSeek API
            medical_text = generator.generate_medical_text(tumor_size, voxel_count)

            # Step 3: Convert to 768-dim features
            features = generator.text_to_features(medical_text)

            # Step 4: Save features
            text_feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(text_feature_path, features)

            processed_count += 1

        except Exception as e:
            print(f"\nERROR processing {sample_name}: {e}")
            error_count += 1
            continue

    # Summary
    print("\n" + "="*60)
    print("Processing complete!")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped:   {skipped_count}")
    print(f"  Errors:    {error_count}")
    print("="*60)


if __name__ == "__main__":
    main()
