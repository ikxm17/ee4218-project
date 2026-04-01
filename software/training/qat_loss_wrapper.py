"""
qat_loss_wrapper.py
===================
Provides build_qat_criterion() and DetectHeadHooks for QAT with the
onnx2torch-converted TinyissimoYOLO model.

Problem
-------
v8DetectionLoss requires:
  1. model.args, model.model[-1]  — not present on onnx2torch model
  2. Raw pre-DFL feature tensors [B, no, H, W] — baked inside the ONNX graph

The onnx2torch model exposes these intermediate tensors at:
  "model/model/16/cv2/0/cv2/0/2/Conv"  — box distribution  [B, reg_max*4, H, W]
  "model/model/16/cv3/0/cv3/0/2/Conv"  — class logits      [B, nc, H, W]

Solution
--------
1. _FakeDetect + _FakeModel supply the missing model attributes.
2. DetectHeadHooks registers forward hooks on the two output convs to
   capture raw feats during the forward pass.
3. build_qat_criterion() returns both the loss and the hook manager.

Usage in run_qat training loop
-------------------------------
    criterion, hooks = build_qat_criterion(model, DEVICE)

    for i, batch in enumerate(train_loader):
        imgs = batch["img"].to(DEVICE)
        ...
        with hooks:                           # registers hooks, auto-removes
            _ = model(imgs)                   # populates hooks.feat
            loss, loss_items = criterion(
                (None, hooks.feats), batch
            )
        loss.backward()
        ...

Known constants (tinyissimo-v1-small.yaml + args.yaml)
-------------------------------------------------------
    nc      = 3
    reg_max = 16   (confirm with dfl/conv/Conv.in_channels)
    stride  = [32] (single scale, 5 maxpool → 256//32 = 8x8 grid)
    box     = 7.5, cls = 0.5, dfl = 1.5   (from args.yaml)
"""

from __future__ import annotations

import types
import torch
import torch.nn as nn
from typing import Optional

from ultralytics.utils.loss import v8DetectionLoss

# ── Layer names where raw Detect head outputs live ────────────────────────────
_CV2_LAYER = "model/model/16/cv2/0/cv2/0/2/Conv"   # box distribution
_CV3_LAYER = "model/model/16/cv3/0/cv3/0/2/Conv"   # class logits

# ── Known constants ───────────────────────────────────────────────────────────
_NC      = 3
_REG_MAX = 16
_NO      = _REG_MAX * 4 + _NC   # 67
_STRIDE  = [32.0]


# ── Hook manager ──────────────────────────────────────────────────────────────

class DetectHeadHooks:
    """
    Context manager that registers forward hooks on the two raw-output
    convolutions inside the baked-in Detect head.

    Captures:
        cv2_out : [B, reg_max*4, H, W]  box distribution logits
        cv3_out : [B, nc, H, W]         class logits

    Concatenates them into feats = [[B, no, H, W]] — the single-scale
    list that v8DetectionLoss.__call__ expects in preds[1].

    Usage:
        with hooks:
            _ = model(imgs)
        loss, items = criterion((None, hooks.feats), batch)
    """
    def __init__(self, model: nn.Module):
        self._model   = model
        self._handles = []
        self.cv2_out  = None
        self.cv3_out  = None

    @property
    def feats(self):
        """Returns [[B, no, H, W]] list for v8DetectionLoss."""
        if self.cv2_out is None or self.cv3_out is None:
            raise RuntimeError(
                "DetectHeadHooks.feats accessed before a forward pass. "
                "Use inside a 'with hooks:' block."
            )
        # Concatenate along channel dim → [B, reg_max*4 + nc, H, W]
        feat = torch.cat([self.cv2_out, self.cv3_out], dim=1)
        return [feat]

    def __enter__(self):
        mods = dict(self._model.named_modules())

        if _CV2_LAYER not in mods:
            raise KeyError(
                f"cv2 layer '{_CV2_LAYER}' not found in model. "
                f"Run: [print(n) for n,_ in model.named_modules()] "
                f"to find the correct name."
            )
        if _CV3_LAYER not in mods:
            raise KeyError(
                f"cv3 layer '{_CV3_LAYER}' not found in model."
            )

        def _hook_cv2(m, inp, out):
            self.cv2_out = out

        def _hook_cv3(m, inp, out):
            self.cv3_out = out

        self._handles.append(
            mods[_CV2_LAYER].register_forward_hook(_hook_cv2)
        )
        self._handles.append(
            mods[_CV3_LAYER].register_forward_hook(_hook_cv3)
        )
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.cv2_out = None
        self.cv3_out = None


# ── Fake model wrapper ────────────────────────────────────────────────────────

class _FakeDetect:
    """Supplies model.model[-1] attributes required by v8DetectionLoss."""
    def __init__(self, nc, no, reg_max, stride, device):
        self.nc      = nc
        self.no      = no
        self.reg_max = reg_max
        self.stride  = torch.tensor(stride, dtype=torch.float32,
                                    device=device)


class _FakeModel(nn.Module):
    """
    Wraps the onnx2torch model to satisfy v8DetectionLoss.__init__:
        model.args       → SimpleNamespace with box/cls/dfl gains
        model.model[-1]  → _FakeDetect with stride/nc/no/reg_max
    """
    def __init__(self, real_model, fake_detect, fake_args):
        super().__init__()
        self._real  = real_model
        self.model  = [fake_detect]
        self.args   = fake_args

    def parameters(self, recurse=True):
        return self._real.parameters(recurse)

    def forward(self, x):
        return self._real(x)


# ── Public factory ────────────────────────────────────────────────────────────

def build_qat_criterion(
    model:    nn.Module,
    device:   torch.device,
    nc:       int   = _NC,
    reg_max:  int   = _REG_MAX,
    stride:   list  = None,
    box_gain: float = 7.5,
    cls_gain: float = 0.5,
    dfl_gain: float = 1.5,
):
    """
    Build v8DetectionLoss and DetectHeadHooks for the onnx2torch model.

    Automatically reads reg_max from the DFL conv layer to confirm the
    value matches the supplied argument — raises if they differ.

    Args:
        model    : quantized onnx2torch model (after build_quantized_model)
        device   : CUDA device
        nc       : number of classes (default 3)
        reg_max  : DFL bins — will be verified against dfl/conv layer
        stride   : grid strides (default [32.0])
        box_gain : box loss weight from args.yaml
        cls_gain : cls loss weight from args.yaml
        dfl_gain : dfl loss weight from args.yaml

    Returns:
        (criterion, hooks)
            criterion : v8DetectionLoss instance
            hooks     : DetectHeadHooks context manager
    """
    if stride is None:
        stride = _STRIDE

    # ── Verify reg_max against the actual DFL conv ────────────────────────────
    mods = dict(model.named_modules())
    dfl_layer_name = "model/model/16/dfl/conv/Conv"
    if dfl_layer_name in mods:
        actual_reg_max = mods[dfl_layer_name].in_channels
        if actual_reg_max != reg_max:
            print(
                f"[build_qat_criterion] WARNING: supplied reg_max={reg_max} "
                f"but dfl/conv/Conv.in_channels={actual_reg_max}. "
                f"Using actual value {actual_reg_max}."
            )
            reg_max = actual_reg_max
    else:
        print(f"[build_qat_criterion] dfl/conv/Conv not found — "
              f"using reg_max={reg_max} as supplied.")

    no = reg_max * 4 + nc

    # ── Verify nc against cv3 output channels ─────────────────────────────────
    if _CV3_LAYER in mods:
        actual_nc = mods[_CV3_LAYER].out_channels
        if actual_nc != nc:
            print(
                f"[build_qat_criterion] WARNING: supplied nc={nc} but "
                f"cv3 Conv.out_channels={actual_nc}. Using {actual_nc}."
            )
            nc = actual_nc
            no = reg_max * 4 + nc

    # ── Build fake wrappers ───────────────────────────────────────────────────
    fake_detect = _FakeDetect(nc, no, reg_max, stride, device)
    fake_args   = types.SimpleNamespace(
        box=box_gain, cls=cls_gain, dfl=dfl_gain
    )
    fake_model  = _FakeModel(model, fake_detect, fake_args)

    criterion = v8DetectionLoss(fake_model)
    # hooks     = DetectHeadHooks(model)

    print(f"[build_qat_criterion] Loss ready: "
          f"nc={nc}  reg_max={reg_max}  no={no}  "
          f"stride={stride}  "
          f"box={box_gain} cls={cls_gain} dfl={dfl_gain}")

    return criterion