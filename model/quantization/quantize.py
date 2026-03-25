from quantization_helper_functions import *
from dataloader import *
import argparse

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--args-yaml", type=str, default="results/exp2/args.yaml",
                        help="Path to args.yaml from a TinyissimoYOLO training run")
    parser.add_argument("--n-calib",   type=int, default=256)
    parser.add_argument("--n-qat",   type=int, default=50)
    parser.add_argument("--onnx-file", type=str, default="tinyissimo_yolo.onnx",
                         help="Path to input onnx file from training")
    parser.add_argument("--onnx-output", type=str, default="tinyissimo_ptq.onnx",
                         help="Path to output onnx file after PTQ")
    parser.add_argument("--tflite-output", type=str, default="tinyissimo.tflite",
                         help="Path to output tflite file after onnx intermediate conversion")
    args = parser.parse_args()

    ONNX_PATH = args.onnx_file
    ONNX_OUTPUT = args.onnx_output
    TFLITE_OUTPUT = args.tflite_output

    # Dataloaders — replace with your actual dataset
    calib_loader = build_ptq_calibration_loader(
        args_yaml=args.args_yaml,
        n_calib=args.n_calib,
        batch_size=64
    )
    train_loader, val_loader = build_qat_loaders(
        args_yaml=args.args_yaml,
        batch_size=64
    )

    # ── Sanity Check with unquantized FP32 model ───────────────────────────────
    model_fp32 = load_fused_model(ONNX_PATH)
    metrics_fp32  = evaluate_map(model_fp32, val_loader, args_yaml="args.yaml")
    print(f"FP32  mAP@0.5:     {metrics_fp32['map50']:.4f}")
    print(f"FP32  mAP@0.5:0.95: {metrics_fp32['map']:.4f}")

    # ── PTQ ───────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RUNNING PTQ")
    print("="*60)
    model_ptq = run_ptq(ONNX_PATH, calib_loader, val_loader,
                        save_path="tinyissimo_ptq.pth", 
                        args_yaml=args.args_yaml)

    # ── Verify PTQ scales ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PTQ SCALE AUDIT")
    print("="*60)
    audit_quantizers(model_ptq)

    test_input = next(iter(val_loader))["img"][:1].to(DEVICE)
    layer_snr_check(model_fp32, model_ptq, test_input)

    # ── Convert PTQ model to tflite ───────────────────────────────────────────
    # export
    export_to_onnx(model_ptq, ONNX_OUTPUT, test_input)
    # check that scales are captured too
    check_onnx_scales(ONNX_OUTPUT)
    # check that calculations area consistent pre and post export
    check_int8_onnx_parity(model_ptq, ONNX_OUTPUT, test_input)
    # export from onnx to tflite
    print(f"Use command: onnx2tf -i {ONNX_OUTPUT} -o output_dir -oiqt -qt per-channel -iqd uint8 -oqd int8 -otl -qnm 0.0 -qns 1.0")

    # ── Extract PTQ weights ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXTRACTING PTQ WEIGHTS")
    print("="*60)
    ptq_params = extract_weights(model_ptq, save_dir="weights_ptq/")
    pack_binary(ptq_params, "tinyissimo_ptq_int8.bin")

    # Generate .coe files for Vivado BRAM initialisation
    for name, p in ptq_params.items():
        safe = name.replace('.', '_').replace('/', '_')
        weights_to_coe(p['W_int8'], f"weights_ptq/{safe}.coe")
    

    # ── QAT (only if PTQ mAP drop > 1%) ──────────────────────────────────────
    print("\nSkipping QAT")
    return
    print("\n" + "="*60)
    print("RUNNING QAT")
    print("="*60)
    # 10% of epoches of initial training for QAT
    model_qat = run_qat(ONNX_PATH, train_loader, val_loader, calib_loader,
                        num_epochs=args.n_qat, base_lr=1e-5,
                        save_path="tinyissimo_qat.pth",
                        args_yaml=args.args_yaml)

    print("\n" + "="*60)
    print("EXTRACTING QAT WEIGHTS")
    print("="*60)
    qat_params = extract_weights(model_qat, save_dir="weights_qat/")
    pack_binary(qat_params, "tinyissimo_qat_int8.bin")

    for name, p in qat_params.items():
        safe = name.replace('.', '_').replace('/', '_')
        weights_to_coe(p['W_int8'], f"weights_qat/{safe}.coe")


if __name__ == "__main__":
    main()