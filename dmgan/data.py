"""CUB/AttnGAN metadata dataset and batch interface."""

from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(slots=True)
class CaptionSample:
    images: list[torch.Tensor]
    caption: torch.Tensor
    caption_length: int
    class_id: int
    key: str


def build_word_mask(caption_lengths: torch.Tensor, word_count: int) -> torch.Tensor:
    positions = torch.arange(word_count, device=caption_lengths.device)[None, :]
    return positions >= caption_lengths[:, None]


class CUBCaptionDataset(Dataset[CaptionSample]):
    """Read official DM-GAN metadata plus CUB-200-2011 images.

    Expected layout:
      root/CUB_200_2011/images/<key>.jpg
      root/CUB_200_2011/images.txt
      root/CUB_200_2011/bounding_boxes.txt
      root/train/filenames.pickle
      root/test/filenames.pickle
      root/captions.pickle
      root/class_info.pickle (or split/class_info.pickle)
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        words_num: int = 18,
        branch_sizes: tuple[int, ...] = (64, 128, 256),
        training: bool = True,
    ) -> None:
        super().__init__()
        from torchvision import transforms

        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        self.root = Path(root)
        self.split = split
        self.words_num = words_num
        self.branch_sizes = branch_sizes
        self.training = training
        with (self.root / split / "filenames.pickle").open("rb") as stream:
            self.keys = pickle.load(stream, encoding="latin1")
        with (self.root / "captions.pickle").open("rb") as stream:
            train_caps, test_caps, self.ixtoword, self.wordtoix = pickle.load(stream, encoding="latin1")
        self.captions = train_caps if split == "train" else test_caps
        self.captions_per_image = len(self.captions) // len(self.keys)
        class_path = self.root / split / "class_info.pickle"
        if not class_path.exists():
            class_path = self.root / "class_info.pickle"
        if class_path.exists():
            with class_path.open("rb") as stream:
                self.class_ids = np.asarray(pickle.load(stream, encoding="latin1"))
        else:
            self.class_ids = np.arange(len(self.keys))
        self.boxes = self._load_boxes()
        augmentation: list[object] = [transforms.Resize(int(branch_sizes[-1] * 76 / 64))]
        if training:
            augmentation.extend([transforms.RandomCrop(branch_sizes[-1]), transforms.RandomHorizontalFlip()])
        else:
            augmentation.append(transforms.CenterCrop(branch_sizes[-1]))
        augmentation.extend([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        self.transform = transforms.Compose(augmentation)

    def _load_boxes(self) -> dict[str, tuple[float, float, float, float]]:
        cub = self.root / "CUB_200_2011"
        images_file = cub / "images.txt"
        boxes_file = cub / "bounding_boxes.txt"
        if not images_file.exists() or not boxes_file.exists():
            return {}
        names = {int(line.split()[0]): line.split(maxsplit=1)[1].rsplit(".", 1)[0] for line in images_file.read_text().splitlines()}
        boxes: dict[str, tuple[float, float, float, float]] = {}
        for line in boxes_file.read_text().splitlines():
            values = line.split()
            boxes[names[int(values[0])]] = tuple(float(value) for value in values[1:5])
        return boxes

    def __len__(self) -> int:
        return len(self.keys)

    def _crop_to_box(self, image: Image.Image, key: str) -> Image.Image:
        if key not in self.boxes:
            return image
        x, y, width, height = self.boxes[key]
        radius = int(max(width, height) * 0.75)
        center_x = int(x + width / 2)
        center_y = int(y + height / 2)
        return image.crop(
            (
                max(0, center_x - radius),
                max(0, center_y - radius),
                min(image.width, center_x + radius),
                min(image.height, center_y + radius),
            )
        )

    def _caption(self, index: int) -> tuple[torch.Tensor, int]:
        source = np.asarray(self.captions[index], dtype=np.int64)
        if len(source) > self.words_num:
            selected = sorted(random.sample(range(len(source)), self.words_num))
            source = source[selected]
        length = len(source)
        caption = torch.zeros(self.words_num, dtype=torch.long)
        caption[:length] = torch.from_numpy(source)
        return caption, length

    def __getitem__(self, index: int) -> CaptionSample:
        key = self.keys[index]
        image_path = self.root / "CUB_200_2011" / "images" / f"{key}.jpg"
        with Image.open(image_path) as source:
            image = self._crop_to_box(source.convert("RGB"), key)
            largest = self.transform(image)
        images = [
            torch.nn.functional.interpolate(
                largest.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
            ).squeeze(0)
            for size in self.branch_sizes
        ]
        caption_index = index * self.captions_per_image + random.randrange(self.captions_per_image)
        caption, caption_length = self._caption(caption_index)
        return CaptionSample(images, caption, caption_length, int(self.class_ids[index]), key)


def collate_caption_samples(samples: list[CaptionSample]) -> dict[str, object]:
    lengths = torch.tensor([sample.caption_length for sample in samples], dtype=torch.long)
    order = torch.argsort(lengths, descending=True)
    images = [torch.stack([samples[index].images[scale] for index in order]) for scale in range(3)]
    captions = torch.stack([samples[index].caption for index in order])
    class_ids = torch.tensor([samples[index].class_id for index in order], dtype=torch.long)
    keys = [samples[index].key for index in order]
    return {"images": images, "captions": captions, "caption_lengths": lengths[order], "class_ids": class_ids, "keys": keys}
