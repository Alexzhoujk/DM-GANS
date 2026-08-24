"""DAMSM encoders and checkpoint loading helpers.

The module names mirror the official implementation so pretrained CUB DAMSM
state dictionaries can be loaded without rewriting their keys.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DAMSMTextEncoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        input_dim: int = 300,
        hidden_dim: int = 256,
        layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = True,
        rnn_type: str = "LSTM",
    ) -> None:
        super().__init__()
        self.n_steps = 18
        self.ntoken = vocabulary_size
        self.ninput = input_dim
        self.drop_prob = dropout
        self.nlayers = layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.nhidden = hidden_dim // self.num_directions
        self.rnn_type = rnn_type.upper()
        self.encoder = nn.Embedding(vocabulary_size, input_dim)
        self.drop = nn.Dropout(dropout)
        rnn_class = nn.LSTM if self.rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_class(
            input_dim,
            self.nhidden,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.encoder.weight.data.uniform_(-0.1, 0.1)

    def init_hidden(self, batch_size: int, device: torch.device | None = None):
        weight = next(self.parameters())
        device = device or weight.device
        shape = (self.nlayers * self.num_directions, batch_size, self.nhidden)
        hidden = torch.zeros(shape, device=device, dtype=weight.dtype)
        if self.rnn_type == "LSTM":
            return hidden, torch.zeros_like(hidden)
        return hidden

    def forward(
        self, captions: torch.Tensor, caption_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.drop(self.encoder(captions))
        lengths = caption_lengths.detach().cpu().tolist()
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=True)
        output, hidden = self.rnn(packed, self.init_hidden(captions.size(0), captions.device))
        output = pad_packed_sequence(output, batch_first=True, total_length=captions.size(1))[0]
        word_embeddings = output.transpose(1, 2).contiguous()
        final_hidden = hidden[0] if self.rnn_type == "LSTM" else hidden
        sentence_embedding = final_hidden.transpose(0, 1).contiguous().view(captions.size(0), -1)
        return word_embeddings, sentence_embedding


class DAMSMImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        from torchvision import models

        self.nef = embedding_dim
        model = models.inception_v3(weights=None, aux_logits=True, init_weights=False)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c
        self.emb_features = nn.Conv2d(768, embedding_dim, 1, bias=False)
        self.emb_cnn_code = nn.Linear(2048, embedding_dim)
        self.emb_features.weight.data.uniform_(-0.1, 0.1)
        self.emb_cnn_code.weight.data.uniform_(-0.1, 0.1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = F.interpolate(image, size=(299, 299), mode="bilinear", align_corners=True)
        image = self.Conv2d_1a_3x3(image)
        image = self.Conv2d_2a_3x3(image)
        image = self.Conv2d_2b_3x3(image)
        image = F.max_pool2d(image, 3, 2)
        image = self.Conv2d_3b_1x1(image)
        image = self.Conv2d_4a_3x3(image)
        image = F.max_pool2d(image, 3, 2)
        image = self.Mixed_5b(image)
        image = self.Mixed_5c(image)
        image = self.Mixed_5d(image)
        image = self.Mixed_6a(image)
        image = self.Mixed_6b(image)
        image = self.Mixed_6c(image)
        image = self.Mixed_6d(image)
        regions = self.Mixed_6e(image)
        image = self.Mixed_7a(regions)
        image = self.Mixed_7b(image)
        image = self.Mixed_7c(image)
        image = F.adaptive_avg_pool2d(image, 1).flatten(1)
        return self.emb_features(regions), self.emb_cnn_code(image)


def load_frozen_checkpoint(module: nn.Module, path: str | Path, *, strict: bool = True) -> nn.Module:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    module.load_state_dict(state, strict=strict)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class TinyMatchingImageEncoder(nn.Module):
    """Fast test-only substitute that preserves the DAMSM encoder interface."""

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, embedding_dim, 3, 1, 1),
        )
        self.global_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        regions = self.features(image)
        regions = F.adaptive_avg_pool2d(regions, (8, 8))
        global_code = self.global_projection(regions.mean(dim=(2, 3)))
        return regions, global_code
