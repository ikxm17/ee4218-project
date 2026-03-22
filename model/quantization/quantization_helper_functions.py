import torch
import torch.nn as nn
import numpy as np
import os
import onnx
from onnx2torch import convert

from pytorch_quantization import quant_modules, tensor_quant
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

from pytorch_quantization import calib
from tqdm import tqdm

from evaluate_map import *
from qat_loss_wrapper import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Load and fuse ──────────────────────────────────────────────────────────────

def load_fused_model(onnx_path: str) -> nn.Module:
    """Convert ONNX → PyTorch and fuse Conv+BN pairs."""
    onnx_model = onnx.load(onnx_path)
    model = convert(onnx_model)

    # Fuse Conv + BN by absorbing BN parameters into conv weights/bias.
    # TinyissimoYOLO has no skip connections so a sequential walk is safe.
    fused_pairs = []
    mods = list(model.named_modules())
    for i in range(len(mods) - 1):
        name_a, mod_a = mods[i]
        name_b, mod_b = mods[i + 1]
        if isinstance(mod_a, nn.Conv2d) and isinstance(mod_b, nn.BatchNorm2d):
            fused_pairs.append((name_a, name_b))

    for conv_name, bn_name in fused_pairs:
        conv = dict(model.named_modules())[conv_name]
        bn   = dict(model.named_modules())[bn_name]
        _absorb_bn_into_conv(conv, bn)
        # Replace BN with Identity so the module graph stays intact
        _set_module(model, bn_name, nn.Identity())

    return model.to(DEVICE)


def _absorb_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    std   = torch.sqrt(bn.running_var + bn.eps)
    scale = bn.weight / std                        # shape: [C_out]

    conv.weight.data *= scale[:, None, None, None]

    if conv.bias is None:
        conv.bias = nn.Parameter(torch.zeros(conv.out_channels,
                                             device=conv.weight.device))
    conv.bias.data = (conv.bias.data - bn.running_mean) * scale + bn.bias


def _set_module(model: nn.Module, name: str, new_mod: nn.Module):
    """Set a submodule by dotted name."""
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


# ── Quantization descriptors ───────────────────────────────────────────────────
# For your HLS/HDL: symmetric INT8, per-channel weights, per-tensor activations.
# zero_point is always 0 under symmetric — confirmed by the library.

WEIGHT_DESC = QuantDescriptor(
    num_bits=8,
    axis=(0,),            # per output-channel (axis 0 of weight tensor)
    unsigned=False,       # signed INT8: range [-128, 127]
    narrow_range=False,   # use full [-128, 127] not [-127, 127]
)

ACT_DESC = QuantDescriptor(
    num_bits=8,
    axis=None,            # per-tensor (single scale for whole activation map)
    unsigned=False,
    narrow_range=False,
    calib_method='histogram',  # histogram calibration on GPU
)

# Push descriptors globally before wrapping modules
quant_nn.QuantConv2d.set_default_quant_desc_input(ACT_DESC)
quant_nn.QuantConv2d.set_default_quant_desc_weight(WEIGHT_DESC)


def build_quantized_model(onnx_path: str) -> nn.Module:
    """
    Load fused FP32 model and replace all Conv2d with QuantConv2d.
    The QuantConv2d wrapper inserts TensorQuantizer on both
    input activation and weight, using the descriptors above.
    """
    model = load_fused_model(onnx_path)

    # Layers to exclude from quantization
    EXCLUDE = {
    # "model/model/16/cv2/0/cv2/0/0/conv/Conv",
    # "model/model/16/cv2/0/cv2/0/1/conv/Conv",
    # "model/model/16/cv2/0/cv2/0/2/Conv",
    # "model/model/16/cv3/0/cv3/0/0/conv/Conv",
    # "model/model/16/cv3/0/cv3/0/1/conv/Conv",
    # "model/model/16/cv3/0/cv3/0/2/Conv",
    "model/model/16/dfl/conv/Conv",
    }

    # Replace Conv2d → QuantConv2d while preserving weights
    for name, mod in list(model.named_modules()):
        if type(mod) is nn.Conv2d:
            if name in EXCLUDE:
                print(f"[build_quantized_model] Skipping quantization: {name}")
                continue
            q_conv = quant_nn.QuantConv2d(
                in_channels=mod.in_channels,
                out_channels=mod.out_channels,
                kernel_size=mod.kernel_size,
                stride=mod.stride,
                padding=mod.padding,
                dilation=mod.dilation,
                groups=mod.groups,
                bias=mod.bias is not None,
            )
            q_conv.weight.data.copy_(mod.weight.data)
            if mod.bias is not None:
                q_conv.bias.data.copy_(mod.bias.data)
            _set_module(model, name, q_conv)

    return model.to(DEVICE)

# ── Calibrate ───────────────────────────────────────────────────
def run_calibration(
    model: nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    num_batches: int = 200,
    percentile: float = 99.99,
) -> nn.Module:
    """
    Phase 1: collect activation histograms on GPU.
    Phase 2: compute amax from histograms using percentile method.

    percentile=99.99 works well for YOLO activations with LeakyReLU,
    which produce occasional large outliers that MinMax over-amplifies.
    """
    model.eval()

    # ── Phase 1: collect histograms ───────────────────────────────────────────
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, TensorQuantizer):
                if mod._calibrator is not None:
                    mod.disable_quant()
                    mod.enable_calib()
                else:
                    mod.disable()

        print("Collecting calibration statistics on GPU...")
        for i, batch in enumerate(tqdm(calib_loader)):
            if i >= num_batches:
                break
            images = batch["img"].to(DEVICE, non_blocking=True)
            model(images)

    # ── Phase 2: compute amax ─────────────────────────────────────────────────
    print(f"\nComputing amax values (percentile={percentile})...")
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, TensorQuantizer):
                if mod._calibrator is not None:

                    # HistogramCalibrator: supports percentile method
                    # MaxCalibrator:       takes no keyword arguments
                    if isinstance(mod._calibrator,
                                  calib.HistogramCalibrator):
                        mod.load_calib_amax(method='percentile',
                                            percentile=percentile)
                    else:
                        mod.load_calib_amax()

                    # Move amax to the correct device after loading
                    # (pytorch-quantization loads amax on CPU by default)
                    if mod.amax is not None:
                        # mod.amax = mod.amax.to(DEVICE)
                        # attemt to preserve internal buffer to ensure contiguous memory
                        mod._amax.data = mod._amax.data.to(DEVICE)

                    mod.disable_calib()
                    mod.enable_quant()
                else:
                    mod.enable()

    print("Calibration complete.")
    return model


# ── Post Calibration Clipping Check  ───────────────────────────────────────────────────
def measure_clipping_rate(
    model:       nn.Module,
    val_loader:  torch.utils.data.DataLoader,
    num_batches: int = 50,
):
    """
    Estimates activation clipping rate per QuantConv2d layer by running
    a dedicated single-image DataLoader and reading quantizer amax values
    directly — no hooks, no module swapping, no graph modification.

    For each layer, collects the input activation tensor by temporarily
    enabling only the input_quantizer's calibrator in observation mode,
    then compares the observed max against the calibrated amax.

    Safe for fx.GraphModule (onnx2torch output) because the model graph
    is never structurally modified.
    """
    from torch.utils.data import DataLoader
    from ultralytics.data.dataset import YOLODataset

    # ── Snapshot calibrated amax values as plain floats ───────────────────────
    amax_map = {}
    for name, mod in model.named_modules():
        if isinstance(mod, quant_nn.QuantConv2d):
            raw = mod.input_quantizer.amax
            if raw is not None:
                amax_map[name] = float(raw.detach().abs().max().cpu())

    if not amax_map:
        print("[measure_clipping_rate] No calibrated layers found — "
              "run calibration first.")
        return

    # ── Per-layer running stats ───────────────────────────────────────────────
    observed_max = {n: 0.0 for n in amax_map}   # max abs activation seen
    clip_counts  = {n: 0   for n in amax_map}
    total_counts = {n: 0   for n in amax_map}

    # ── Register plain input hooks that only do tensor.abs().max() ───────────
    # These hooks do NOT call into the quantizer or touch any CUDA buffer —
    # they only read the floating-point input tensor that arrives at the conv.
    handles = []

    def _make_obs_hook(layer_name, amax_val):
        def _hook(module, args, output):
            # args[0] is the input tensor to the QuantConv2d
            x = args[0].detach()
            clip_counts[layer_name]  += int((x.abs() > amax_val).sum().item())
            total_counts[layer_name] += int(x.numel())
            cur_max = float(x.abs().max().item())
            if cur_max > observed_max[layer_name]:
                observed_max[layer_name] = cur_max
        return _hook

    for name, mod in model.named_modules():
        if name in amax_map:
            h = mod.register_forward_hook(_make_obs_hook(name, amax_map[name]))
            handles.append(h)

    # ── Build batch_size=1 loader to avoid static reshape issue ──────────────
    def _collate_f32(batch):
        result = YOLODataset.collate_fn(batch)
        result["img"] = result["img"].float() / 255.0
        return result

    single_loader = DataLoader(
        val_loader.dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = _collate_f32,
        pin_memory  = False,
    )

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(single_loader):
            if i >= num_batches:
                break
            try:
                model(batch["img"].to(DEVICE))
            except Exception as e:
                print(f"[measure_clipping_rate] Forward pass {i} failed: {e}")
                break

    # ── Remove hooks ──────────────────────────────────────────────────────────
    for h in handles:
        h.remove()

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'Layer':<45} {'amax':>8} {'obs_max':>10} "
          f"{'Clip rate':>12}   {'Status'}")
    print("-" * 85)
    for name in amax_map:
        tot      = total_counts[name]
        clip     = clip_counts[name]
        rate     = (clip / tot * 100) if tot > 0 else 0.0
        obs_max  = observed_max[name]
        cal_amax = amax_map[name]
        status   = "OK" if rate < 0.1 else ("WARN" if rate < 1.0 else "HIGH")
        print(f"{name:<45} {cal_amax:>8.4f} {obs_max:>10.4f} "
              f"{rate:>10.4f}%   {status}")
    print()

# ── PTQ  ───────────────────────────────────────────────────
def run_ptq(
    onnx_path: str,
    calib_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    save_path: str = "tinyissimo_ptq.pth",
    args_yaml: str = "args.yaml"
) -> nn.Module:

    print("Building quantized model...")
    model = build_quantized_model(onnx_path)

    print("Running PTQ calibration...")
    model = run_calibration(model, calib_loader, num_batches=200)

    # TODO: Fix measure_clipping_rate for debugging if needed
    # print("Checking on Clipping...")
    # measure_clipping_rate(model, val_loader, num_batches=50)

    # do quick audit
    audit_quantizers(model)

    # Evaluate immediately after calibration
    metrics = evaluate_map(
    model      = model,
    val_loader = val_loader,
    args_yaml  = args_yaml,   # conf resolved automatically → 0.001
    )
    print(f"PTQ mAP@0.5      : {metrics['map50']:.4f}")
    print(f"PTQ mAP@0.5:0.95 : {metrics['map']:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"PTQ model saved to {save_path}")
    return model

# ── QAT  ───────────────────────────────────────────────────
def run_qat(
    onnx_path:    str,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    calib_loader: torch.utils.data.DataLoader,
    num_epochs:   int   = 10,
    base_lr:      float = 1e-5,
    save_path:    str   = "tinyissimo_qat.pth",
    args_yaml:    str   = "args.yaml",
) -> nn.Module:

    print("Building and calibrating model for QAT initialisation...")
    model   = build_quantized_model(onnx_path)
    n_calib = min(100, len(calib_loader))
    model   = run_calibration(model, calib_loader, num_batches=n_calib)

    # build_qat_criterion constructs v8DetectionLoss via a fake Detect wrapper
    # because the onnx2torch model lacks model.args and model.model[-1]
    criterion = build_qat_criterion(model, DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr           = base_lr,
        momentum     = 0.9,
        weight_decay = 1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )

    for epoch in range(num_epochs):
        model.train()

        if epoch == 2:
            print("Freezing quantizer amax values...")
            for name, mod in model.named_modules():
                if isinstance(mod, TensorQuantizer):
                    mod.disable_calib()
                    mod.enable_quant()

        if epoch == 4:
            for mod in model.modules():
                if isinstance(mod, nn.BatchNorm2d):
                    mod.eval()

        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            imgs = batch["img"].to(DEVICE, non_blocking=True)
            batch["bboxes"]    = batch["bboxes"].to(DEVICE, non_blocking=True)
            batch["cls"]       = batch["cls"].to(DEVICE, non_blocking=True)
            batch["batch_idx"] = batch["batch_idx"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # model.train() takes the Detect training path: return x
            # raw[1] = x = list of raw feature tensors [B, no, H, W]
            # This is exactly what v8DetectionLoss.__call__ expects in preds[1]
            # Detect.forward() confirmed: return (y, x) in inference,
            # return x in training — so model(imgs) returns x directly
            # when model.train() is active
            outputs = model(imgs)

            # In train mode Detect returns x (list of tensors), not (y, x)
            # v8DetectionLoss expects (pred_or_None, feat_list)
            if isinstance(outputs, (list, tuple)) and not isinstance(outputs[0], torch.Tensor):
                # train mode: outputs is the raw feat list directly
                feats = outputs
            else:
                # inference mode fallback: outputs = (y, x)
                feats = outputs[1]

            loss, loss_items = criterion((None, feats), batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        metrics = evaluate_map(
            model      = model,
            val_loader = val_loader,
            args_yaml  = args_yaml,
        )

        box_l, cls_l, dfl_l = loss_items.detach().cpu()
        print(
            f"Epoch [{epoch+1:3d}/{num_epochs}]  "
            f"loss: {running_loss/len(train_loader):.4f}  "
            f"(box={box_l:.3f} cls={cls_l:.3f} dfl={dfl_l:.3f})  "
            f"mAP@0.5: {metrics['map50']:.4f}  "
            f"mAP@0.5:0.95: {metrics['map']:.4f}  "
            f"lr: {scheduler.get_last_lr()[0]:.2e}"
        )

    torch.save(model.state_dict(), save_path)
    print(f"QAT model saved to {save_path}")
    return model
 

# ── Scale Verification  ───────────────────────────────────────────────────
def audit_quantizers(model: nn.Module):
    print(f"\n{'Quantizer':<55} {'amax':<12} {'scale':<12} {'status'}")
    print("-" * 90)

    issues = []
    for name, mod in model.named_modules():
        # Specifically target the quantizers inside QuantConv2d
        from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
        if not isinstance(mod, TensorQuantizer):
            continue

        amax = mod.amax
        if amax is None:
            print(f"{name:<55} {'NONE':<12} {'–':<12} FAIL (Not Calibrated)")
            issues.append(name)
            continue

        # Convert to a standard float for display
        amax_val = amax.detach().cpu().abs().max().item()
        scale_val = amax_val / 127.0

        ok = True
        status = "OK"
        
        if amax_val == 0:
            status = "ZERO!"
            ok = False
        elif not np.isfinite(amax_val):
            status = "INF/NaN"
            ok = False
        
        print(f"{name:<55} {amax_val:<12.5f} {scale_val:<12.8f} {status}")
        if not ok:
            issues.append(f"{name} ({status})")

    if issues:
        print(f"\n🚨 Found {len(issues)} problematic quantizers. cuDNN will crash if amax is 0.")
    else:
        print("\n✅ All quantizers have valid amax values.")
    # Report any Conv2d layers that were intentionally left unquantized
    print("\nUnquantized Conv2d layers (intentional):")
    for name, mod in model.named_modules():
        if type(mod) is nn.Conv2d:
            print(f"  {name}  (kept FP32)")

def layer_snr_check(model_fp32: nn.Module,
                    model_quant: nn.Module,
                    test_input: torch.Tensor):
    """
    Layer-wise SNR between FP32 and quantized activations.
    Both models run on GPU; hooks capture outputs.
    """
    fp32_acts  = {}
    quant_acts = {}

    def hook_fp32(name):
        def h(m, i, o):
            fp32_acts[name] = o.detach().float()
        return h

    def hook_quant(name):
        def h(m, i, o):
            t = o.dequantize() if o.is_quantized else o
            quant_acts[name] = t.detach().float()
        return h

    handles = []
    for name, mod in model_fp32.named_modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(hook_fp32(name)))
    for name, mod in model_quant.named_modules():
        if isinstance(mod, quant_nn.QuantConv2d):
            handles.append(mod.register_forward_hook(hook_quant(name)))

    test_input = test_input.to(DEVICE)
    with torch.no_grad():
        model_fp32(test_input)
        model_quant(test_input)

    for h in handles:
        h.remove()

    print(f"\n{'Layer':<40} {'SNR (dB)':<12} {'Max err':<12} {'Status'}")
    print("-" * 72)

    for (nf, af), (nq, aq) in zip(fp32_acts.items(), quant_acts.items()):
        if af.shape != aq.shape:
            print(f"{nf:<40} shape mismatch"); continue
        sig  = (af ** 2).mean()
        noise = ((af - aq) ** 2).mean()
        snr  = 10 * torch.log10(sig / (noise + 1e-12))
        merr = (af - aq).abs().max()
        status = "OK" if snr > 25 else ("WARN" if snr > 15 else "FAIL")
        print(f"{nf:<40} {snr.item():>8.2f} dB   {merr.item():>8.5f}   {status}")

# ── Weight Extraction  ───────────────────────────────────────────────────
def extract_weights(model: nn.Module,
                    save_dir: str = "weights/") -> dict:
    """
    Extract from every QuantConv2d:
      W_int8      int8 numpy array, shape [C_out, C_in, kH, kW]
      b_int32     int32 numpy array, shape [C_out]
      s_w         per-channel weight scales, shape [C_out]
      s_x_in      input activation scale (scalar)
      M           requant multiplier per channel, shape [C_out]
      m0          fixed-point mantissa (int32), shape [C_out]
      n_shift     right-shift count (int32), shape [C_out]
    """
    os.makedirs(save_dir, exist_ok=True)
    all_params = {}

    # Collect activation scales: each QuantConv2d stores its input
    # activation amax on the input_quantizer, and its weight amax
    # on the weight_quantizer.
    # Output activation scale = input amax of the *next* layer.
    # Build a list of (name, module) for conv layers in forward order.
    conv_layers = [(n, m) for n, m in model.named_modules()
                   if isinstance(m, quant_nn.QuantConv2d)]

    for layer_idx, (name, mod) in enumerate(conv_layers):

        # ── Weight quantization ───────────────────────────────────────────────
        w_fp32  = mod.weight.detach().cpu().float()          # [Co, Ci, kH, kW]
        # w_amax  = mod.weight_quantizer.amax.detach().cpu()   # [Co] per-channel

        w_amax_raw = mod.weight_quantizer.amax.detach().cpu()
 
        # Debug: print raw shape before any reshape
        # print(f"  [{layer_idx:02d}] {name}")
        # print(f"       w_fp32 shape    : {tuple(w_fp32.shape)}")
        # print(f"       w_amax raw shape: {tuple(w_amax_raw.shape)}")
 
        # Validate: amax must have exactly Co elements regardless of shape
        Co_expected = w_fp32.shape[0]
        if w_amax_raw.numel() != Co_expected:
            raise ValueError(
                f"Layer {name}: weight_quantizer.amax has {w_amax_raw.numel()} "
                f"elements but Conv2d has Co={Co_expected} output channels. "
                f"Calibration may not have run for this layer."
            )
 
        # Flatten to 1D [Co] — handles [Co], [Co,1], [Co,1,1], [Co,1,1,1]
        w_amax = w_amax_raw.reshape(-1)   # guaranteed [Co]
        assert w_amax.ndim == 1 and w_amax.shape[0] == Co_expected, \
            f"Layer {name}: w_amax reshape failed — got {w_amax.shape}"


        # scale per output channel
        s_w = (w_amax / 127.0).numpy().astype(np.float32)   # shape [Co]

        # s_w_tensor must be strictly 1D before adding broadcast dims.
        # If s_w still had extra dims here, [:, None, None, None] would
        # produce a 7D+ tensor instead of [Co, 1, 1, 1].
        s_w_tensor = torch.tensor(s_w, dtype=torch.float32)  # [Co] — 1D
        assert s_w_tensor.ndim == 1, \
            f"Layer {name}: s_w_tensor is not 1D — got shape {s_w_tensor.shape}"

        # Quantize weights to INT8
        W_int8 = torch.clamp(
            torch.round(w_fp32 / s_w_tensor[:, None, None, None]),
            -128, 127,
        ).numpy().astype(np.int8)

        # Validate output shape — must be 4D [Co, Ci, kH, kW]
        if W_int8.ndim != 4:
            raise ValueError(
                f"Layer {name}: W_int8 has unexpected shape {W_int8.shape} "
                f"(expected 4D [Co,Ci,kH,kW]). "
                f"w_fp32={tuple(w_fp32.shape)}, s_w={tuple(s_w_tensor.shape)}"
            )

        # ── Bias (keep as int32 for accumulator) ─────────────────────────────
        if mod.bias is not None:
            b_fp32  = mod.bias.detach().cpu().float().numpy()
            # Bias scale = s_x_in * s_w (one per output channel)
            # We quantize bias to int32 using this combined scale
            # so the accumulator can add it directly to the INT32 accumulation
            b_int32 = None  # computed below once s_x_in is known
        else:
            b_fp32  = np.zeros(mod.out_channels, dtype=np.float32)
            b_int32 = None

        # ── Input activation scale ────────────────────────────────────────────
        in_amax = mod.input_quantizer.amax.detach().cpu().item()
        s_x_in  = float(in_amax / 127.0)

        # Quantize bias now that we have s_x_in
        s_bias  = s_x_in * s_w                               # shape [Co]
        b_int32 = np.round(b_fp32 / s_bias).astype(np.int32)

        # ── Output activation scale ───────────────────────────────────────────
        # = input scale of the next conv layer, or use this layer's
        #   output quantizer if it's the last layer
        if layer_idx + 1 < len(conv_layers):
            next_mod = conv_layers[layer_idx + 1][1]
            out_amax = next_mod.input_quantizer.amax.detach().cpu().item()
        else:
            # Last layer — use a nominal output scale of 1/127
            out_amax = 1.0
        s_x_out = float(out_amax / 127.0)

        # ── Requantization multiplier ─────────────────────────────────────────
        # M[c] = (s_x_in * s_w[c]) / s_x_out  — one per output channel
        M = (s_x_in * s_w) / s_x_out           # shape [Co], float32

        # Decompose M into a 16-bit normalised dyadic rational: M ≈ m0 × 2^(-n)
        # n is chosen so that m0 ∈ [32768, 65536) — leading bit always 1 (normalised)
        # This gives 15 fractional bits of precision, keeping RT_err < 0.002%
        # for all M values in this model (M range: 0.003 to 0.021)
        #
        # Hardware requantization (HDL/HLS):
        #   scaled   = acc_int32 × m0          (INT32 × INT32 → INT64, two DSP48E2)
        #   out_int8 = clamp(scaled >> n, -128, 127)
        # Software (ARM): native INT64 multiply
        # HLS:            ap_int<64> intermediate
        # HDL:            64-bit wire, two DSP48E2 cascaded

        n  = np.maximum(0, (15 - np.floor(np.log2(M + 1e-30)))).astype(np.int32)
        m0 = np.round(M * (2.0 ** n)).astype(np.int32)

        # Verify round-trip accuracy
        M_rt    = m0.astype(np.float64) / (2.0 ** n)
        rt_err  = np.abs(M - M_rt) / (M + 1e-12) * 100
        max_err = rt_err.max()

        print(f"Layer {layer_idx:02d} {name:<30} "
              f"s_x_in={s_x_in:.6f}  "
              f"s_x_out={s_x_out:.6f}  "
              f"M=[{M.min():.5f},{M.max():.5f}]  "
              f"RT_err={max_err:.4f}%  "
              f"{'OK' if max_err < 0.1 else 'WARN'}")
        
        # Perform check for possible overflows
        if np.any(M >= 1.0):
            bad_channels = np.where(M >= 1.0)[0]
            raise ValueError(
                f"Layer {name}: M >= 1.0 for output channels {bad_channels}. "
                f"This will overflow the INT32 requantization stage in hardware. "
                f"Recalibrate with more representative data or increase s_x_out."
            )

        params = {
            'W_int8':  W_int8,
            'b_int32': b_int32,
            's_w':     s_w,
            's_x_in':  np.float32(s_x_in),
            's_x_out': np.float32(s_x_out),
            'M':       M,
            'm0':      m0,
            'n_shift': n,
        }
        all_params[name] = params

        # ── Save per-layer numpy files ─────────────────────────────────
        # Sanitise layer name: replace '/' and '.' with '_' so the name
        # is safe to use as a filename component
        safe_name = name.replace('/', '_').replace('.', '_')
        prefix    = f"{save_dir}/layer{layer_idx:02d}_{safe_name}"

        np.save(f"{prefix}_W_int8.npy",  W_int8)
        np.save(f"{prefix}_b_int32.npy", b_int32)
        np.save(f"{prefix}_s_w.npy",     s_w)
        np.save(f"{prefix}_scales.npy",
                np.array([s_x_in, s_x_out], dtype=np.float32))
        np.save(f"{prefix}_M.npy",       M)
        np.save(f"{prefix}_m0_n.npy",
                np.stack([m0, n]).astype(np.int32))

    return all_params


def pack_binary(all_params: dict, out_path: str = "tinyissimo_int8.bin"):
    """
    Pack all layers into a single binary for SD card / SCP deployment.

    Per-layer layout:
      [int32]  layer_index
      [int32]  C_out, C_in, kH, kW
      [int8 × C_out×C_in×kH×kW]  W_int8   (OIHW row-major)
      [int32 × C_out]             b_int32
      [int32 × C_out]             m0
      [int32 × C_out]             n_shift
      [float32]                   s_x_in
      [float32]                   s_x_out
    """
    with open(out_path, 'wb') as f:
        for idx, (name, p) in enumerate(all_params.items()):
            W  = p['W_int8']
            # Debug — print shape of every layer before unpacking
            print(f"  [{idx:02d}] {name}  W.shape={W.shape}  ndim={W.ndim}")
            # Squeeze any leading dimensions of size 1
            # onnx2torch may add a batch dim: [1, Co, Ci, kH, kW] → [Co, Ci, kH, kW]
            while W.ndim > 4 and W.shape[0] == 1:
                W = W.squeeze(0)
            # onnx2torch may store 1×1 conv weights as [Co, Ci]
            # reshape to [Co, Ci, 1, 1] for consistent binary layout
            if W.ndim == 2:
                W = W[:, :, np.newaxis, np.newaxis]

            Co, Ci, kH, kW = W.shape
            f.write(np.array([idx, Co, Ci, kH, kW],
                              dtype=np.int32).tobytes())
            f.write(W.tobytes())
            f.write(p['b_int32'].tobytes())
            f.write(p['m0'].tobytes())
            f.write(p['n_shift'].tobytes())
            f.write(p['s_x_in'].tobytes())
            f.write(p['s_x_out'].tobytes())

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nPacked {len(all_params)} layers → {out_path}  ({size_kb:.1f} KB)")


def weights_to_coe(W_int8: np.ndarray, filename: str):
    """Vivado BRAM .coe initialisation file from int8 weight array."""
    flat = W_int8.flatten().view(np.uint8)
    with open(filename, 'w') as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        f.write(',\n'.join(f'{b:02x}' for b in flat))
        f.write(';\n')

