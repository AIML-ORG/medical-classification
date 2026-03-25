"""
Smoke test: QAT prepare -> short forward -> convert -> save state_dict ->
rebuild graph (prepare+convert) -> load_state_dict -> forward.

Second phase: same pipeline with best_model_3class config + tokenizer, then load_trained_model().

Run from repo root: python3 cursor_scripts/verify_qat_bert_pipeline.py
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
import tempfile

import torch
from transformers import BertConfig, BertForSequenceClassification

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAIN_PATH = os.path.join(_REPO_ROOT, "v2", "train_ocr_three_class_bert.py")
_BEST_DIR = os.path.join(_REPO_ROOT, "best_model_3class")


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_ocr_three_class_bert", _TRAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, _REPO_ROOT)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_qat_cycle(
    mod,
    cfg: BertConfig,
    device: torch.device,
    inp: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    model = BertForSequenceClassification(cfg).to(device)
    model.train()
    mod.apply_qat_prepare_model(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(2):
        opt.zero_grad()
        out = model(input_ids=inp, attention_mask=mask, labels=labels)
        out.loss.backward()
        opt.step()
    model.eval()
    mod.quantize_(model, mod.QATConfig(step="convert"), filter_fn=mod._is_fake_qat_for_convert)
    return model.state_dict()


def main() -> None:
    mod = _load_train_module()
    device = torch.device("cpu")
    mod._set_torch_quantized_engine_for_cpu()

    cfg_small = BertConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        num_labels=3,
    )
    inp_s = torch.randint(0, 128, (2, 16), device=device)
    mask_s = torch.ones(2, 16, dtype=torch.long, device=device)
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    sd_small = _run_qat_cycle(mod, cfg_small, device, inp_s, mask_s, labels)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        path = tmp.name
    try:
        torch.save(sd_small, path)
        reloaded = mod.build_qat_converted_bert_for_load(cfg_small)
        reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        reloaded.eval()
        with torch.no_grad():
            out2 = reloaded(input_ids=inp_s, attention_mask=mask_s)
        assert out2.logits.shape == (2, 3), out2.logits.shape
        print(
            "verify_qat_bert_pipeline (tiny): OK | logits",
            tuple(out2.logits.shape),
            "| bytes",
            os.path.getsize(path),
        )
    finally:
        os.unlink(path)

    if not os.path.isdir(_BEST_DIR) or not os.path.isfile(os.path.join(_BEST_DIR, "config.json")):
        print("skip load_trained_model e2e: best_model_3class/config.json missing")
        return

    cfg_full = BertConfig.from_pretrained(_BEST_DIR)
    vs = min(cfg_full.vocab_size, 30522)
    inp_f = torch.randint(0, vs, (1, 32), device=device)
    mask_f = torch.ones(1, 32, dtype=torch.long, device=device)
    sd_full = _run_qat_cycle(mod, cfg_full, device, inp_f, mask_f, torch.tensor([0], device=device))

    tmp_dir = tempfile.mkdtemp(prefix="qat_bert_load_")
    try:
        for name in os.listdir(_BEST_DIR):
            if name in ("model_qat_int8.pt", "model_int8_dynamic.pt"):
                continue
            s = os.path.join(_BEST_DIR, name)
            d = os.path.join(tmp_dir, name)
            if os.path.isfile(s):
                shutil.copy2(s, d)
        torch.save(sd_full, os.path.join(tmp_dir, mod.QAT_INT8_WEIGHTS_FILENAME))
        m2, _tok, dev = mod.load_trained_model(tmp_dir, torch.device("cpu"))
        assert m2 is not None and dev is not None
        m2.eval()
        with torch.no_grad():
            o3 = m2(input_ids=inp_f, attention_mask=mask_f)
        assert o3.logits.shape == (1, 3)
        print("load_trained_model(QAT path, full config): OK | logits", tuple(o3.logits.shape))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
