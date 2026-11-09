"""PII-Tracer's decoding, windowing, span handling and runtimes, with the real model replaced: no download."""

from __future__ import annotations

import numpy as np
import pytest

from tg_summary import pii_tracer
from tg_summary.config import RedactionSettings
from tg_summary.pii_tracer import LABELS, Found, decode, whole_words, windows
from tg_summary.redact import RequestMap, build_redactor
from tg_summary.store import Store


def _logits(*tags: str) -> np.ndarray:
    """One row per token, strongly preferring the given tag."""
    rows = np.full((len(tags), len(LABELS)), -5.0)
    for t, tag in enumerate(tags):
        rows[t, LABELS.index(tag)] = 5.0
    return rows


def test_decoder_joins_bioes_tags_into_trimmed_character_spans():
    text = "Hi Dave Smith, mail d@x.io"
    offsets = [(0, 2), (2, 7), (7, 13), (13, 14), (14, 19), (19, 26)]  # the tokenizer folds leading spaces in
    tags = ["O", "B-private_person", "E-private_person", "O", "O", "S-private_email"]
    assert decode(_logits(*tags), offsets, text) == [
        Found(3, 13, "private_person"),  # " Dave" starts with a space; the span doesn't
        Found(20, 26, "private_email"),
    ]


def test_decoder_never_emits_a_span_from_an_illegal_tag_sequence():
    # Token by token the logits say I-, I-: no B- opens it, so a constrained decode can't produce that span.
    text = "ab cd"
    spans = decode(_logits("I-private_person", "I-private_person"), [(0, 2), (2, 5)], text)
    assert all(text[s.start : s.end] != "ab cd" or s.label != "private_person" for s in spans)
    assert all(s.start < s.end for s in spans)


def test_windows_keep_whole_lines_and_cover_the_text_exactly():
    text = "one two\nthree\nfour five six seven eight\n" + "x" * 50

    def count(lines):
        return [len(line) // 2 for line in lines]

    pieces = windows(text, count, limit=12)
    assert "".join(p for _, p in pieces) == text
    assert all(text[start : start + len(p)] == p for start, p in pieces)
    assert pieces[0][1] == "one two\nthree\n"  # 4 + 3 tokens fit; the next line wouldn't
    assert all(count([p])[0] <= 12 for _, p in pieces)  # the 25-token line was split by characters


def test_windows_stay_within_the_limit_when_characters_cost_different_token_counts():
    # Dense text (CJK, code, emoji) costs more tokens per character, so cutting a long line into equal character
    # lengths can still overflow the model's window; whatever overflows would go unread.
    text = "aaaa" + "界" * 8

    def count(lines):
        return [sum(4 if ch == "界" else 1 for ch in line) for line in lines]

    pieces = windows(text, count, limit=12)
    assert "".join(p for _, p in pieces) == text
    assert all(text[start : start + len(p)] == p for start, p in pieces)
    assert max(count([p])[0] for _, p in pieces) <= 12


@pytest.mark.parametrize(
    ("text", "span", "whole"),
    [("dave: lmao", (1, 4), "dave"), ("ping @samwise now", (6, 13), "@samwise"), ("key sk-4f9a", (6, 11), "sk-4f9a")],
)
def test_spans_widen_to_whole_words_and_handles(text, span, whole):
    start, end = whole_words(text, *span)
    assert text[start:end] == whole


def test_pii_tracer_spans_are_redacted_whole_and_dates_are_kept_by_default(tmp_path, monkeypatch):
    text = "dave: meetup 14 March at 221B Baker Street"
    found = [
        Found(1, 4, "private_person"),  # the model cut "dave" short; redaction must not leave the "d"
        Found(13, 21, "private_date"),
        Found(25, 42, "private_address"),
    ]
    monkeypatch.setattr(pii_tracer.PiiTracer, "find", lambda self, t: found)
    settings = RedactionSettings(detectors=["pii_tracer"])
    store = Store(tmp_path / "chat.sqlite")
    r = build_redactor(store.vault_assign, store.vault_values, [], settings)
    assert r.redact(text, RequestMap()) == "[PERSON_1]: meetup 14 March at [ADDRESS_1]"


def test_unknown_detectors_and_labels_are_rejected():
    with pytest.raises(ValueError, match=r"redaction\.detectors"):
        RedactionSettings(detectors=["gliner"])
    with pytest.raises(ValueError, match=r"redaction\.pii_tracer_labels"):
        RedactionSettings(pii_tracer_labels=["person"])


def test_mlx_and_pytorch_encoders_compute_the_same_model(tmp_path):
    # A tiny random Qwen3 encoder in fp32, run by both runtimes: any difference in how either builds the model
    # (RoPE layout, q/k norms, grouped-query attention, the bidirectional mask) shows up far above fp32 noise.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("mlx.core")
    from safetensors.numpy import save_file

    from tg_summary.pii_tracer import _MlxEncoder, _TorchEncoder

    hidden, layers, heads, kv_heads, head_dim, ffn = 32, 2, 4, 2, 8, 64
    backbone = {
        "vocab_size": 50, "hidden_size": hidden, "intermediate_size": ffn, "num_hidden_layers": layers,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads, "head_dim": head_dim,
        "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "max_position_embeddings": 128, "hidden_act": "silu", "attention_bias": False, "tie_word_embeddings": True,
    }  # fmt: skip
    rng = np.random.default_rng(0)

    def w(*shape):
        return rng.normal(0, 0.3, shape).astype(np.float32)

    weights = {"backbone.embed_tokens.weight": w(50, hidden), "backbone.norm.weight": 1 + w(hidden)}
    for i in range(layers):
        p = f"backbone.layers.{i}."
        weights |= {
            p + "input_layernorm.weight": 1 + w(hidden), p + "post_attention_layernorm.weight": 1 + w(hidden),
            p + "self_attn.q_proj.weight": w(heads * head_dim, hidden),
            p + "self_attn.k_proj.weight": w(kv_heads * head_dim, hidden),
            p + "self_attn.v_proj.weight": w(kv_heads * head_dim, hidden),
            p + "self_attn.o_proj.weight": w(hidden, heads * head_dim),
            p + "self_attn.q_norm.weight": 1 + w(head_dim), p + "self_attn.k_norm.weight": 1 + w(head_dim),
            p + "mlp.gate_proj.weight": w(ffn, hidden), p + "mlp.up_proj.weight": w(ffn, hidden),
            p + "mlp.down_proj.weight": w(hidden, ffn),
        }  # fmt: skip
    weights |= {
        "token_cls_head.weight": w(len(LABELS), hidden), "token_cls_head.bias": w(len(LABELS)),
        "viterbi.b_bias": np.zeros(1, np.float32), "viterbi.e_bias": np.zeros(1, np.float32),
    }  # fmt: skip
    path = str(tmp_path / "tiny.safetensors")
    save_file(weights, path)

    ids = [int(i) for i in rng.integers(0, 50, 40)]
    reference = _TorchEncoder(path, backbone, "cpu", dtype="float32").logits(ids)
    mlx = _MlxEncoder(path, backbone, "auto").logits(ids)
    assert reference.shape == mlx.shape == (40, len(LABELS))
    assert np.abs(reference - mlx).max() < 1e-3
