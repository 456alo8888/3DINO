from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


_CODE_ROOT = Path(__file__).resolve().parents[4]
if str(_CODE_ROOT) not in sys.path:
    sys.path.append(str(_CODE_ROOT))

from datasets.SOOP_dataset import SOOPTraceTabularDataset  # noqa: E402


DEFAULT_TARGET_COLS = ("nihss", "gs_rankin_6isdeath")


class SOOPOutcomeDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        target_col: str = "gs_rankin_6isdeath",
        transform=None,
        include_tabular: bool = False,
        drop_missing_labels: bool = False,
        return_subject_id: bool = False,
        target_cols: Sequence[str] = DEFAULT_TARGET_COLS,
    ) -> None:
        self.transform = transform
        self.include_tabular = include_tabular
        self.return_subject_id = return_subject_id

        self.base_dataset = SOOPTraceTabularDataset(
            split_csv=split_csv,
            target_cols=target_cols,
            transform=None,
            allow_missing_labels=True,
            return_subject_id=return_subject_id,
        )

        if target_col not in self.base_dataset.target_cols:
            raise ValueError(
                f"Target column '{target_col}' not in available target columns: {self.base_dataset.target_cols}"
            )

        self.target_col = target_col
        self.target_index = self.base_dataset.target_cols.index(target_col)

        label_mask = self.base_dataset.label_mask[:, self.target_index]
        if drop_missing_labels:
            self.valid_indices = np.where(label_mask > 0.0)[0].tolist()
        else:
            self.valid_indices = list(range(len(self.base_dataset)))

    def __len__(self) -> int:
        return len(self.valid_indices)

    @staticmethod
    def _validate_image_shape(image: torch.Tensor) -> None:
        if image.ndim != 4:
            raise ValueError(f"SOOP image tensor must be 4D (C,H,W,D), got shape={tuple(image.shape)}")
        if image.shape[0] != 1:
            raise ValueError(
                "SOOP image tensor must be single-channel channel-first (1,H,W,D), "
                f"got shape={tuple(image.shape)}"
            )

    def __getitem__(self, index: int):
        base_index = self.valid_indices[index]
        sample = self.base_dataset[base_index]

        image = sample["image"]
        label = sample["target"][self.target_index]
        label_mask = sample["label_mask"][self.target_index]

        transform_input = {"image": image, "label": label}
        if self.transform is not None:
            transformed = self.transform(transform_input)
            if isinstance(transformed, dict):
                image = transformed.get("image", image)
                label = transformed.get("label", label)
            else:
                image = transformed

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        image = image.to(torch.float32)
        self._validate_image_shape(image)

        if isinstance(label, torch.Tensor):
            label_value = label.item()
        else:
            label_value = float(label)

        output = {
            "image": image,
            "label": torch.tensor(label_value, dtype=torch.float32),
            "label_mask": torch.tensor(float(label_mask), dtype=torch.float32),
        }

        if self.include_tabular:
            tabular = sample["tabular"]
            if isinstance(tabular, np.ndarray):
                tabular = torch.from_numpy(tabular)
            output["tabular"] = tabular.to(torch.float32)

        if self.return_subject_id and "subject_id" in sample:
            output["subject_id"] = sample["subject_id"]

        return output
