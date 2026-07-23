"""
FedSAM3-Cream dataset validator.

Checks three layers of dataset state:
1. Top-level client metadata under data/federated_split/<client>/dataset.json
2. Global split manifests: train_split.json / val_split.json / test_split.json
3. Active on-disk split directories used by the training pipeline
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DatasetValidator:
    RESERVED_DIRS = {"train", "val", "test", "__pycache__"}
    SPLITS = ("train", "val", "test")

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def log_issue(self, message: str):
        self.issues.append(message)
        print(f"[ISSUE] {message}")

    def log_warning(self, message: str):
        self.warnings.append(message)
        print(f"[WARNING] {message}")

    def log_info(self, message: str):
        self.info.append(message)
        print(f"[INFO] {message}")

    def validate_all(self, check_leakage: bool = True, verbose: bool = False) -> bool:
        print("=" * 80)
        print("FedSAM3-Cream dataset validation")
        print("=" * 80)

        if not self.data_root.exists():
            self.log_issue(f"Data root does not exist: {self.data_root}")
            return False

        client_datasets = self.load_all_client_datasets()
        split_manifests = self.load_split_manifests()
        split_layout = self.inspect_split_directories(verbose=verbose)

        if not client_datasets:
            self.log_issue("No client metadata dataset.json files found")
            return False

        self.validate_dataset_completeness(client_datasets, verbose=verbose)
        self.report_sample_statistics(client_datasets, split_manifests, split_layout)

        if check_leakage:
            self.check_data_leakage(split_manifests, split_layout)

        self.validate_modality_consistency(client_datasets, split_layout)
        self.generate_summary_report()

        return len(self.issues) == 0

    def load_all_client_datasets(self) -> Dict[str, Dict]:
        print("\n1. Load client metadata...")
        client_datasets: Dict[str, Dict] = {}

        for client_dir in sorted(self.data_root.iterdir()):
            if not client_dir.is_dir():
                continue
            if client_dir.name in self.RESERVED_DIRS or client_dir.name.endswith("_backup"):
                continue

            dataset_file = client_dir / "dataset.json"
            if not dataset_file.exists():
                continue

            try:
                with open(dataset_file, "r", encoding="utf-8") as handle:
                    raw = json.load(handle)
            except Exception as exc:
                self.log_issue(f"{client_dir.name} dataset.json read failed: {exc}")
                continue

            storage_root = self._infer_train_client_root(client_dir.name)
            entries = raw.get("data", [])
            client_datasets[client_dir.name] = {
                "client_id": client_dir.name,
                "metadata_path": dataset_file,
                "storage_root": storage_root,
                "modality": raw.get("modality"),
                "description": raw.get("description"),
                "raw": raw,
                "entries": [
                    self._normalize_metadata_entry(storage_root, idx, entry)
                    for idx, entry in enumerate(entries)
                ],
            }
            self.log_info(f"Loaded client metadata: {client_dir.name}")

        return client_datasets

    def _infer_train_client_root(self, metadata_client_id: str) -> Optional[Path]:
        match = re.match(r"client_?(\d+)", metadata_client_id)
        if not match:
            return None
        return self.data_root / "train" / f"client_{match.group(1)}" / "private"

    def _normalize_metadata_entry(
        self,
        storage_root: Optional[Path],
        idx: int,
        entry: Dict,
    ) -> Dict:
        image_value = entry.get("image", "empty")
        if isinstance(image_value, list):
            image_paths = [
                storage_root / rel_path
                for rel_path in image_value
                if storage_root is not None
                and isinstance(rel_path, str)
                and rel_path
                and rel_path != "empty"
            ]
        elif (
            storage_root is not None
            and isinstance(image_value, str)
            and image_value
            and image_value != "empty"
        ):
            image_paths = [storage_root / image_value]
        else:
            image_paths = []

        label_value = entry.get("label", "empty")
        label_path = None
        if (
            storage_root is not None
            and isinstance(label_value, str)
            and label_value
            and label_value != "empty"
        ):
            label_path = storage_root / label_value

        text_value = entry.get("text_feature", entry.get("text", "empty"))
        text_path = None
        if (
            storage_root is not None
            and isinstance(text_value, str)
            and text_value
            and text_value != "empty"
        ):
            text_path = storage_root / text_value

        case_id = self._derive_case_id(image_paths, label_path, text_path, idx)
        return {
            "case_id": case_id,
            "fold": entry.get("fold"),
            "image_paths": image_paths,
            "label_path": label_path,
            "text_path": text_path,
        }

    def _derive_case_id(
        self,
        image_paths: List[Path],
        label_path: Optional[Path],
        text_path: Optional[Path],
        idx: int,
    ) -> str:
        for candidate in image_paths:
            if candidate.parent.name.startswith("BraTS"):
                return candidate.parent.name
        for candidate in (label_path, text_path):
            if candidate is not None and candidate.parent.name.startswith("BraTS"):
                return candidate.parent.name
        return f"sample_{idx:04d}"

    def load_split_manifests(self) -> Dict[str, Dict]:
        print("\n2. Load split manifests...")
        manifests: Dict[str, Dict] = {}

        for split in self.SPLITS:
            manifest_path = self.data_root / f"{split}_split.json"
            if not manifest_path.exists():
                self.log_warning(f"Missing split manifest: {manifest_path.name}")
                manifests[split] = {"entries": [], "case_ids": set()}
                continue

            try:
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    raw = json.load(handle)
            except Exception as exc:
                self.log_issue(f"{manifest_path.name} read failed: {exc}")
                manifests[split] = {"entries": [], "case_ids": set()}
                continue

            entries = raw.get("data", [])
            case_ids = {self._derive_case_id_from_manifest_entry(entry) for entry in entries}
            manifests[split] = {"entries": entries, "case_ids": case_ids}
            self.log_info(f"Loaded split manifest: {manifest_path.name} ({len(entries)} samples)")

        return manifests

    def _derive_case_id_from_manifest_entry(self, entry: Dict) -> str:
        image_value = entry.get("image", [])
        if isinstance(image_value, list) and image_value:
            return Path(image_value[0]).parts[0]
        if isinstance(image_value, str) and image_value and image_value != "empty":
            return Path(image_value).parts[0]

        label_value = entry.get("label")
        if isinstance(label_value, str) and label_value and label_value != "empty":
            return Path(label_value).parts[0]

        text_value = entry.get("text_feature", entry.get("text"))
        if isinstance(text_value, str) and text_value and text_value != "empty":
            return Path(text_value).parts[0]

        return "unknown"

    def inspect_split_directories(self, verbose: bool = False) -> Dict[str, Dict[str, Set[str]]]:
        print("\n3. Inspect active split directories...")
        split_layout: Dict[str, Dict[str, Set[str]]] = {split: {} for split in self.SPLITS}

        for split in ("train", "val"):
            split_root = self.data_root / split
            if not split_root.exists():
                self.log_issue(f"Missing directory: {split_root}")
                continue

            for client_dir in sorted(split_root.iterdir()):
                if not client_dir.is_dir():
                    continue
                private_dir = client_dir / "private"
                case_ids = self._collect_case_ids(private_dir)
                split_layout[split][client_dir.name] = case_ids
                if verbose:
                    print(f"  {split}/{client_dir.name}: {len(case_ids)} cases")

        test_global = self.data_root / "test" / "global"
        split_layout["test"]["global"] = self._collect_case_ids(test_global)
        if verbose:
            print(f"  test/global: {len(split_layout['test']['global'])} cases")

        return split_layout

    def _collect_case_ids(self, root_dir: Path) -> Set[str]:
        if not root_dir.exists():
            self.log_warning(f"Directory does not exist: {root_dir}")
            return set()

        return {
            item.name
            for item in root_dir.iterdir()
            if item.is_dir() and item.name.startswith("BraTS")
        }

    def validate_dataset_completeness(self, client_datasets: Dict[str, Dict], verbose: bool = False):
        print("\n4. Validate metadata contract and file completeness...")

        for client_id, client_data in client_datasets.items():
            raw = client_data["raw"]
            modality = client_data.get("modality")
            entries = client_data["entries"]
            storage_root = client_data.get("storage_root")

            if "modality" not in raw:
                self.log_issue(f"{client_id} missing top-level field: modality")
            if "data" not in raw:
                self.log_issue(f"{client_id} missing top-level field: data")
                continue
            if not isinstance(raw["data"], list):
                self.log_issue(f"{client_id} top-level field data must be a list")
                continue
            if not entries:
                self.log_issue(f"{client_id} data list is empty")
                continue
            if storage_root is None:
                self.log_issue(f"{client_id} cannot be mapped to active train/client_x/private")
                continue
            if not storage_root.exists():
                self.log_issue(f"{client_id} mapped data directory does not exist: {storage_root}")
                continue

            for entry in entries:
                case_id = entry["case_id"]
                image_paths = entry["image_paths"]
                label_path = entry["label_path"]
                text_path = entry["text_path"]

                if modality in {"image_only", "multimodal"} and not image_paths:
                    self.log_issue(f"{client_id}:{case_id} missing image paths")
                if modality in {"image_only", "multimodal"} and label_path is None:
                    self.log_issue(f"{client_id}:{case_id} missing label path")
                if modality in {"text_only", "multimodal"} and text_path is None:
                    self.log_issue(f"{client_id}:{case_id} missing text feature path")

                for image_path in image_paths:
                    if not image_path.exists():
                        self.log_issue(f"{client_id}:{case_id} image file missing: {image_path}")
                if label_path is not None and not label_path.exists():
                    self.log_issue(f"{client_id}:{case_id} label file missing: {label_path}")
                if text_path is not None and not text_path.exists():
                    self.log_issue(f"{client_id}:{case_id} text feature file missing: {text_path}")

            if verbose:
                print(f"  {client_id}: modality={modality}, metadata samples={len(entries)}")

    def report_sample_statistics(
        self,
        client_datasets: Dict[str, Dict],
        split_manifests: Dict[str, Dict],
        split_layout: Dict[str, Dict[str, Set[str]]],
    ):
        print("\n5. Sample statistics...")

        print("\n  Client metadata counts:")
        print(f"  {'client':<28} {'modality':<14} {'metadata':<12}")
        print(f"  {'-' * 28} {'-' * 14} {'-' * 12}")
        for client_id, client_data in sorted(client_datasets.items()):
            print(
                f"  {client_id:<28}"
                f"{str(client_data.get('modality', 'unknown')):<14}"
                f"{len(client_data['entries']):<12}"
            )

        print("\n  Split manifest counts:")
        for split in self.SPLITS:
            print(f"  {split:<5}: {len(split_manifests[split]['entries'])}")

        print("\n  Active split case counts:")
        print(f"  {'client':<12} {'train':<8} {'val':<8}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 8}")
        client_ids = sorted(set(split_layout["train"].keys()) | set(split_layout["val"].keys()))
        for client_id in client_ids:
            train_count = len(split_layout["train"].get(client_id, set()))
            val_count = len(split_layout["val"].get(client_id, set()))
            print(f"  {client_id:<12} {train_count:<8} {val_count:<8}")
            if client_id != "client_1" and val_count == 0:
                self.log_issue(f"{client_id} has empty validation directory")

        test_count = len(split_layout["test"].get("global", set()))
        print(f"\n  test/global: {test_count} cases")

    def check_data_leakage(
        self,
        split_manifests: Dict[str, Dict],
        split_layout: Dict[str, Dict[str, Set[str]]],
    ):
        print("\n6. Check data leakage...")

        train_ids = split_manifests["train"]["case_ids"]
        val_ids = split_manifests["val"]["case_ids"]
        test_ids = split_manifests["test"]["case_ids"]

        overlap_train_val = train_ids & val_ids
        overlap_train_test = train_ids & test_ids
        overlap_val_test = val_ids & test_ids

        if overlap_train_val:
            self.log_issue(f"train/val overlap detected: {sorted(list(overlap_train_val))[:5]}")
        if overlap_train_test:
            self.log_issue(f"train/test overlap detected: {sorted(list(overlap_train_test))[:5]}")
        if overlap_val_test:
            self.log_issue(f"val/test overlap detected: {sorted(list(overlap_val_test))[:5]}")

        if not (overlap_train_val or overlap_train_test or overlap_val_test):
            print("  [OK] No cross-split overlap in split manifests")

        for split_name in ("train", "val"):
            per_client = split_layout[split_name]
            client_ids = sorted(per_client.keys())
            has_overlap = False
            for index, left_id in enumerate(client_ids):
                for right_id in client_ids[index + 1:]:
                    overlap = per_client[left_id] & per_client[right_id]
                    if overlap:
                        has_overlap = True
                        self.log_warning(
                            f"{split_name} overlap between {left_id} and {right_id}: {len(overlap)} cases"
                        )
            if not has_overlap:
                print(f"  [OK] No within-{split_name} overlap between clients")

    def validate_modality_consistency(
        self,
        client_datasets: Dict[str, Dict],
        split_layout: Dict[str, Dict[str, Set[str]]],
    ):
        print("\n7. Validate modality consistency...")

        for client_id, client_data in sorted(client_datasets.items()):
            modality = client_data.get("modality", "unknown")
            entries = client_data["entries"]
            total = len(entries)
            image_count = sum(1 for entry in entries if entry["image_paths"])
            text_count = sum(1 for entry in entries if entry["text_path"] is not None)
            label_count = sum(1 for entry in entries if entry["label_path"] is not None)

            print(f"\n  {client_id} ({modality})")
            print(f"    image samples: {image_count}/{total}")
            print(f"    text samples:  {text_count}/{total}")
            print(f"    label samples: {label_count}/{total}")

            if modality == "text_only":
                if text_count != total:
                    self.log_issue(f"{client_id} is text_only but not all samples have text features")
                if image_count > 0:
                    self.log_warning(f"{client_id} is text_only but metadata contains image paths")
            elif modality == "image_only":
                if image_count != total:
                    self.log_issue(f"{client_id} is image_only but not all samples have images")
                if text_count > 0:
                    self.log_warning(f"{client_id} is image_only but metadata contains text features")
            elif modality == "multimodal":
                if image_count != total:
                    self.log_issue(f"{client_id} is multimodal but not all samples have images")
                if text_count != total:
                    self.log_issue(f"{client_id} is multimodal but not all samples have text features")
            else:
                self.log_warning(f"{client_id} has unknown modality: {modality}")

        if "client_3" in split_layout["val"]:
            missing_text_cases = self._find_validation_multimodal_cases_missing_text()
            if missing_text_cases:
                self.log_warning(
                    "val/client_3/private has cases without *_text.npy; this is acceptable for Phase A validation "
                    f"because validation no longer requires text. Examples: {missing_text_cases[:3]}"
                )

    def _find_validation_multimodal_cases_missing_text(self) -> List[str]:
        client_root = self.data_root / "val" / "client_3" / "private"
        if not client_root.exists():
            return []

        missing: List[str] = []
        for case_dir in sorted(d for d in client_root.iterdir() if d.is_dir() and d.name.startswith("BraTS")):
            if not any(case_dir.glob("*_text.npy")):
                missing.append(case_dir.name)
        return missing

    def generate_summary_report(self):
        print("\n" + "=" * 80)
        print("Validation summary")
        print("=" * 80)
        print(f"Issues:   {len(self.issues)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Info:     {len(self.info)}")

        if self.issues:
            print("\nIssues:")
            for index, issue in enumerate(self.issues, start=1):
                print(f"  {index}. {issue}")

        if self.warnings:
            print("\nWarnings:")
            for index, warning in enumerate(self.warnings, start=1):
                print(f"  {index}. {warning}")

        print("\n" + "=" * 80)
        print("Dice sanity checklist")
        print("=" * 80)
        print("1. Confirm the validator reports no structural errors.")
        print("2. Confirm train and val use the same normalization and mask encoding.")
        print("3. Confirm val/client_2 and val/client_3 case counts are reasonable.")
        print("4. Confirm validation case IDs do not overlap with train/test.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate FedSAM3-Cream dataset layout")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/federated_split",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--check-leakage",
        action="store_true",
        help="Check overlap across train/val/test splits",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed counts",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    validator = DatasetValidator(args.data_root)
    ok = validator.validate_all(
        check_leakage=args.check_leakage,
        verbose=args.verbose,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
