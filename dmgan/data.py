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

from .part_aware import token_part_targets


@dataclass(slots=True)
class CaptionSample:
    images: list[torch.Tensor]
    caption: torch.Tensor
    caption_length: int
    class_id: int
    key: str
    part_coordinates: torch.Tensor | None = None
    part_visible: torch.Tensor | None = None
    token_part_targets: torch.Tensor | None = None


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
        include_parts: bool = False,
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
        self.include_parts = include_parts
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
        self.part_locations = self._load_part_locations() if include_parts else {}
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
        names = {
            int(line.split()[0]): line.split(maxsplit=1)[1].rsplit(".", 1)[0]
            for line in images_file.read_text().splitlines()
        }
        boxes: dict[str, tuple[float, float, float, float]] = {}
        for line in boxes_file.read_text().splitlines():
            values = line.split()
            boxes[names[int(values[0])]] = tuple(float(value) for value in values[1:5])
        return boxes

    def _load_part_locations(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        cub = self.root / "CUB_200_2011"
        images_file = cub / "images.txt"
        parts_file = cub / "parts" / "part_locs.txt"
        if not images_file.exists() or not parts_file.exists():
            raise FileNotFoundError("CUB part annotations require images.txt and parts/part_locs.txt")
        names = {
            int(line.split()[0]): line.split(maxsplit=1)[1].rsplit(".", 1)[0]
            for line in images_file.read_text().splitlines()
        }
        coordinates = {image_id: torch.zeros(15, 2) for image_id in names}
        visible = {image_id: torch.zeros(15, dtype=torch.bool) for image_id in names}
        for line in parts_file.read_text().splitlines():
            image_id_text, part_id_text, x_text, y_text, visible_text = line.split()
            image_id = int(image_id_text)
            part_index = int(part_id_text) - 1
            coordinates[image_id][part_index] = torch.tensor([float(x_text), float(y_text)])
            visible[image_id][part_index] = bool(int(visible_text))
        return {names[image_id]: (coordinates[image_id], visible[image_id]) for image_id in names}

    def __len__(self) -> int:
        return len(self.keys)

    def _crop_bounds(self, image: Image.Image, key: str) -> tuple[int, int, int, int]:
        if key not in self.boxes:
            return 0, 0, image.width, image.height
        x, y, width, height = self.boxes[key]
        radius = int(max(width, height) * 0.75)
        center_x = int(x + width / 2)
        center_y = int(y + height / 2)
        return (
            max(0, center_x - radius),
            max(0, center_y - radius),
            min(image.width, center_x + radius),
            min(image.height, center_y + radius),
        )

    def _crop_to_box(self, image: Image.Image, key: str) -> Image.Image:
        return image.crop(self._crop_bounds(image, key))

    def _transform_with_parts(
        self,
        image: Image.Image,
        key: str,
        *,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torchvision import transforms
        from torchvision.transforms import functional as transform_functional

        output_size = self.branch_sizes[-1]
        resize_size = int(output_size * 76 / 64)
        coordinates, visible = self.part_locations[key]
        coordinates = coordinates.clone()
        visible = visible.clone()

        left, top, right, bottom = self._crop_bounds(image, key)
        cropped = image.crop((left, top, right, bottom))
        coordinates[:, 0] -= left
        coordinates[:, 1] -= top
        visible &= (
            (coordinates[:, 0] >= 0)
            & (coordinates[:, 0] < cropped.width)
            & (coordinates[:, 1] >= 0)
            & (coordinates[:, 1] < cropped.height)
        )

        resized = transform_functional.resize(cropped, resize_size, antialias=True)
        coordinates[:, 0] *= resized.width / max(cropped.width, 1)
        coordinates[:, 1] *= resized.height / max(cropped.height, 1)

        if training:
            crop_top, crop_left, crop_height, crop_width = transforms.RandomCrop.get_params(
                resized, (output_size, output_size)
            )
        else:
            crop_height = crop_width = output_size
            crop_top = round((resized.height - output_size) / 2.0)
            crop_left = round((resized.width - output_size) / 2.0)
        transformed = transform_functional.crop(resized, crop_top, crop_left, crop_height, crop_width)
        coordinates[:, 0] -= crop_left
        coordinates[:, 1] -= crop_top
        visible &= (
            (coordinates[:, 0] >= 0)
            & (coordinates[:, 0] < crop_width)
            & (coordinates[:, 1] >= 0)
            & (coordinates[:, 1] < crop_height)
        )

        if training and bool(torch.rand(()) < 0.5):
            transformed = transform_functional.hflip(transformed)
            coordinates[:, 0] = (crop_width - 1) - coordinates[:, 0]

        coordinates[:, 0] /= max(crop_width - 1, 1)
        coordinates[:, 1] /= max(crop_height - 1, 1)
        coordinates.clamp_(0.0, 1.0)
        tensor = transform_functional.to_tensor(transformed)
        tensor = transform_functional.normalize(tensor, (0.5,) * 3, (0.5,) * 3)
        return tensor, coordinates, visible

    def part_coordinates_for_key(self, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic center-crop coordinates for held-out evaluation."""
        if not self.include_parts:
            raise RuntimeError("Dataset was created with include_parts=False")
        image_path = self.root / "CUB_200_2011" / "images" / f"{key}.jpg"
        with Image.open(image_path) as source:
            _, coordinates, visible = self._transform_with_parts(source.convert("RGB"), key, training=False)
        return coordinates, visible

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
            image = source.convert("RGB")
            if self.include_parts:
                largest, part_coordinates, part_visible = self._transform_with_parts(
                    image, key, training=self.training
                )
            else:
                largest = self.transform(self._crop_to_box(image, key))
                part_coordinates = None
                part_visible = None
        images = [
            torch.nn.functional.interpolate(
                largest.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
            ).squeeze(0)
            for size in self.branch_sizes
        ]
        caption_index = index * self.captions_per_image + random.randrange(self.captions_per_image)
        caption, caption_length = self._caption(caption_index)
        targets = token_part_targets(caption, self.ixtoword) if self.include_parts else None
        return CaptionSample(
            images,
            caption,
            caption_length,
            int(self.class_ids[index]),
            key,
            part_coordinates,
            part_visible,
            targets,
        )


def collate_caption_samples(samples: list[CaptionSample]) -> dict[str, object]:
    lengths = torch.tensor([sample.caption_length for sample in samples], dtype=torch.long)
    order = torch.argsort(lengths, descending=True)
    images = [torch.stack([samples[index].images[scale] for index in order]) for scale in range(3)]
    captions = torch.stack([samples[index].caption for index in order])
    class_ids = torch.tensor([samples[index].class_id for index in order], dtype=torch.long)
    keys = [samples[index].key for index in order]
    batch: dict[str, object] = {
        "images": images,
        "captions": captions,
        "caption_lengths": lengths[order],
        "class_ids": class_ids,
        "keys": keys,
    }
    optional_fields = ("part_coordinates", "part_visible", "token_part_targets")
    for field in optional_fields:
        values = [getattr(samples[index], field) for index in order]
        if all(value is not None for value in values):
            batch[field] = torch.stack(values)  # type: ignore[arg-type]
    return batch
