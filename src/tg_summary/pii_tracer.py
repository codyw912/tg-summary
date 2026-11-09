"""PII-Tracer: a PII detector trained on conversations, run locally (optional: `uv sync --extra pii`).

PII-Tracer (perplexity-ai/PII-Tracer, MIT, https://arxiv.org/abs/2609.22200) is a 0.6B Qwen3 encoder run
bidirectionally, with a token head that tags BIOES labels over 9 PII types, decoded by a constrained Viterbi.
This module loads the weights from a pinned revision of the Hugging Face repo (downloaded once into the
Hugging Face cache, about 1.2 GB) and re-implements the model's small inference code instead of running the
repo's own code through `trust_remote_code`. The decoder is plain numpy, so other runtimes can share it.

Text is fed in windows of whole lines, because the model reads at most 4,096 tokens and is weaker on very short
inputs. Spans are widened to whole words (a span covering "ave" of "dave" must not leave "d" visible).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tg_summary.config import PII_TRACER_LABELS as PII_TYPES  # in the checkpoint's order

REPO = "perplexity-ai/PII-Tracer"
REVISION = "d25c16f2e57e321f6d2527715c01df9112f956f5"  # verified against the repo's reference code
MODEL_MAX_TOKENS = 4096  # the most the model reads at once
WINDOW_TOKENS = 3500  # what `windows` aims for, below MODEL_MAX_TOKENS

LABELS = ["O"] + [f"{tag}-{t}" for t in PII_TYPES for tag in "BIES"]  # 37, in the checkpoint's order

# Our placeholder kind for each PII-Tracer label.
KINDS = {
    "private_person": "PERSON",
    "private_email": "EMAIL",
    "private_phone": "PHONE",
    "private_address": "ADDRESS",
    "private_url": "URL",
    "private_date": "DATE",
    "account_number": "ACCOUNT",
    "secret": "SECRET",
    "other_pii": "PII",
}


@dataclass(frozen=True, slots=True)
class Found:
    start: int
    end: int
    label: str


# --- decoding (numpy; a port of the repo's constrained BIOES Viterbi) ---------------------------------------


def _transitions(b_bias: float, e_bias: float) -> np.ndarray:
    """[37, 37] transition scores: 0 where legal, -inf where not; biases on entering B and leaving E states."""
    index = {label: i for i, label in enumerate(LABELS)}
    n = len(LABELS)
    legal = np.zeros((n, n), dtype=bool)
    ends, begins = {index["O"]}, {index["O"]}  # states a span can end on / states valid after a boundary
    for t in PII_TYPES:
        b, i, e, s = (index[f"{tag}-{t}"] for tag in "BIES")
        ends |= {e, s}
        begins |= {b, s}
        legal[b, i] = legal[b, e] = legal[i, i] = legal[i, e] = True
    for f in ends:
        for to in begins:
            legal[f, to] = True
    scores = np.where(legal, 0.0, -np.inf)
    for t in PII_TYPES:
        b, e = index[f"B-{t}"], index[f"E-{t}"]
        scores[:, b] = np.where(legal[:, b], scores[:, b] + b_bias, -np.inf)
        scores[e, :] = np.where(legal[e, :], scores[e, :] + e_bias, -np.inf)
    return scores


def decode(logits: np.ndarray, offsets: list[tuple[int, int]], text: str, b_bias=0.0, e_bias=0.0) -> list[Found]:
    """Character spans for one sequence of [T, 37] tag logits, with surrounding whitespace trimmed."""
    if len(logits) == 0:
        return []
    trans = _transitions(b_bias, e_bias)
    allowed_start = np.array([lab == "O" or lab[:2] in ("B-", "S-") for lab in LABELS])
    allowed_end = np.array([lab == "O" or lab[:2] in ("E-", "S-") for lab in LABELS])
    score = logits[0] + np.where(allowed_start, 0.0, -np.inf)
    back = np.zeros(logits.shape, dtype=np.int64)
    for t in range(1, len(logits)):
        candidates = score[:, None] + trans  # [previous, next]
        back[t] = candidates.argmax(axis=0)
        score = logits[t] + candidates.max(axis=0)
    path = [int((score + np.where(allowed_end, 0.0, -np.inf)).argmax())]
    for t in range(len(logits) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()

    found = []
    t = 0
    while t < len(path):
        label = LABELS[path[t]]
        start, end = offsets[t]
        t += 1
        if label.startswith("B-"):
            kind = label[2:]
            while t < len(path) and LABELS[path[t]] in (f"I-{kind}", f"E-{kind}"):
                end = offsets[t][1]
                t += 1
                if LABELS[path[t - 1]] == f"E-{kind}":
                    break
        elif not label.startswith("S-"):
            continue
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            found.append(Found(start, end, label[2:]))
    return found


# --- windows and spans -------------------------------------------------------------------------------------


def windows(text: str, count_tokens, limit: int = WINDOW_TOKENS) -> list[tuple[int, str]]:
    """(start offset, text) pieces covering `text` exactly, each within `limit` tokens by `count_tokens`.

    Whole lines are packed together; then every piece is recounted as one string (per-line counts can differ
    from the count of the joined text) and any piece over the limit is halved, at a line break or a space when
    there is one, until it fits. Nothing is truncated: text the model doesn't read is text nobody checked."""
    lines = text.splitlines(keepends=True)
    counts = count_tokens(lines) if lines else []
    packed: list[tuple[int, str]] = []
    start = pos = used = 0
    for line, n in zip(lines, counts, strict=True):
        if used and used + n > limit:
            packed.append((start, text[start:pos]))
            start, used = pos, 0
        pos += len(line)
        used += n
    if pos > start:
        packed.append((start, text[start:pos]))
    return [fitted for piece in packed for fitted in _fit(piece, count_tokens, limit)]


def _fit(piece: tuple[int, str], count_tokens, limit: int) -> list[tuple[int, str]]:
    start, s = piece
    if len(s) <= 1 or count_tokens([s])[0] <= limit:
        return [piece]
    half = len(s) // 2
    newline = s.rfind("\n", 0, half)
    space = s.rfind(" ", len(s) // 4, half)
    cut = newline + 1 if newline > 0 else space + 1 if space > 0 else half
    return _fit((start, s[:cut]), count_tokens, limit) + _fit((start + cut, s[cut:]), count_tokens, limit)


_WORD = re.compile(r"\w")


def whole_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a span to whole words, and over a leading "@" of a handle."""
    while start > 0 and _WORD.match(text[start - 1]):
        start -= 1
    if start > 0 and text[start - 1] == "@":
        start -= 1
    while end < len(text) and _WORD.match(text[end]):
        end += 1
    return start, end


# --- the model (PyTorch) -----------------------------------------------------------------------------------


class PiiTracer:
    """The model, its tokenizer and the device it runs on. Loads lazily: the first call downloads the weights."""

    def __init__(self, device: str = "auto"):
        self._device_setting = device
        self._model: Any = None

    def _load(self) -> None:
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from tokenizers import Tokenizer
            from transformers import Qwen3Config, Qwen3Model
        except ImportError as e:
            raise RuntimeError("the pii_tracer detector needs the optional dependencies: uv sync --extra pii") from e

        print("loading PII-Tracer (the first run downloads about 1.2 GB)", file=sys.stderr, flush=True)
        path = {
            name: hf_hub_download(REPO, name, revision=REVISION)
            for name in ("config.json", "tokenizer.json", "model.safetensors")
        }
        config = json.loads(Path(path["config.json"]).read_text(encoding="utf-8"))
        backbone_config = Qwen3Config.from_dict(config["backbone"])
        backbone_config.is_causal = False  # an encoder: every token sees the whole window
        backbone_config._attn_implementation = "sdpa"
        weights = load_file(path["model.safetensors"])
        backbone = Qwen3Model(backbone_config)
        # Weights in bf16 like the checkpoint; the rotary position buffers stay fp32, as transformers keeps them.
        # Casting those too shifts every logit (by up to 0.25 against the reference code).
        buffers = {name: b.clone() for name, b in backbone.named_buffers()}
        backbone = backbone.to(torch.bfloat16)
        for name, b in buffers.items():
            owner, _, attr = name.rpartition(".")
            setattr(backbone.get_submodule(owner), attr, b)
        missing, unexpected = backbone.load_state_dict(
            {k.removeprefix("backbone."): v for k, v in weights.items() if k.startswith("backbone.")}, strict=False
        )
        if unexpected or any("inv_freq" not in k for k in missing):
            raise RuntimeError(f"PII-Tracer weights don't match the model: {missing[:3]} {unexpected[:3]}")

        device = self._device_setting
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._device = device
        self._model = backbone.to(device).eval()
        self._head_w = weights["token_cls_head.weight"].float().to(device)
        self._head_b = weights["token_cls_head.bias"].float().to(device)
        self._biases = (float(weights["viterbi.b_bias"]), float(weights["viterbi.e_bias"]))
        self._tokenizer = Tokenizer.from_file(path["tokenizer.json"])
        # The file truncates at 4,096 tokens. Counts must be exact and nothing may be cut silently: text the model
        # doesn't read is text nobody checked.
        self._tokenizer.no_truncation()

    def count_tokens(self, lines: list[str]) -> list[int]:
        if self._model is None:
            self._load()
        return [len(e.ids) for e in self._tokenizer.encode_batch(lines, add_special_tokens=False)]

    def find(self, text: str) -> list[Found]:
        """The model's PII spans in `text` (any length), as it predicts them: they may cut through a word."""
        if self._model is None:
            self._load()
        torch = self._torch
        found = []
        for offset, piece in windows(text, self.count_tokens):
            enc = self._tokenizer.encode(piece, add_special_tokens=False)
            if not enc.ids:
                continue
            if len(enc.ids) > MODEL_MAX_TOKENS:  # windows() keeps pieces under WINDOW_TOKENS; never feed more
                raise RuntimeError(f"PII-Tracer window of {len(enc.ids)} tokens exceeds {MODEL_MAX_TOKENS}")
            ids = torch.tensor([enc.ids], device=self._device)
            with torch.inference_mode():
                hidden = self._model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state
                logits = (hidden[0].float() @ self._head_w.T + self._head_b).cpu().numpy()
            found.extend(
                Found(offset + f.start, offset + f.end, f.label)
                for f in decode(logits, enc.offsets, piece, *self._biases)
            )
        return found


def spans_for(text: str, found: Iterable[Found], labels: Iterable[str]) -> list[tuple[int, int, str]]:
    """(start, end, kind) for the configured labels, widened to whole words: a span the model cut short ("ave"
    of "dave") must not leave the rest of the word visible."""
    wanted = set(labels)
    return [(*whole_words(text, f.start, f.end), KINDS[f.label]) for f in found if f.label in wanted]
