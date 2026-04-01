import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="model/tinyissimo_ptq_full_integer_quant.tflite",
                        help="Path to TFLite model")
    parser.add_argument("--input-image", type=str, default="data/input_image.jpg",
                        help="Path to input image")
    parser.add_argument("--result-file", type=str, default="view_tensors.txt")
    args = parser.parse_args()

    # 1. Setup Interpreter with Tensor Preservation
    interpreter = tflite.Interpreter(
        model_path=args.model_path, 
        experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    in_idx = input_details['index']

    # 2. Preprocess Image
    img = Image.open(args.input_image).convert('RGB').resize((256, 256))
    input_data = np.expand_dims(np.array(img), axis=0)

    # 3. Inference
    interpreter.set_tensor(in_idx, input_data)
    interpreter.invoke()

    # 4. Extract and Log Tensors
    tensor_details = interpreter.get_tensor_details()
    # Manually verify the first element of the Mul operation
    a_raw = interpreter.get_tensor(52)[0,0,0,0] # 89
    b_raw = interpreter.get_tensor(53)[0,0,0,0] # 50
    out_raw = interpreter.get_tensor(54)[0,0,0,0]

    # Calculate what it SHOULD be
    a_real = 0.3484917 * (a_raw - 3)
    b_real = 0.00390625 * (b_raw - (-128))
    rescale_factor = (0.3484917 * 0.00390625) / 0.1700168
    expected_q = round((a_raw - 3) * (b_raw + 128) * rescale_factor) - 126

    print(f"Input A Real: {a_real:.4f}")
    print(f"Input B Real: {b_real:.4f}")
    print(f"Expected Out (int8): {expected_q}")
    print(f"Actual Out (int8): {out_raw}")
    details_53 = interpreter._get_tensor_details(53,0)
    details_54 = interpreter._get_tensor_details(54,0)

    print(f"Index 53 Data Address: {details_53['data_address']}")
    print(f"Index 54 Data Address: {details_54['data_address']}")
    return
    with open(args.result_file, "w") as f:
        for tensor in tensor_details:
            idx = tensor['index']
            name = tensor['name']
            scale, zp = tensor['quantization']
            
            try:
                val = interpreter.get_tensor(idx)
                
                # Header info
                f.write(f"\n{'='*80}\n")
                f.write(f"INDEX: {idx} | NAME: {name}\n")
                f.write(f"SHAPE: {val.shape} | DTYPE: {val.dtype}\n")
                f.write(f"QUANT: Scale={scale}, ZeroPoint={zp}\n")
                
                # Write a small slice to avoid massive file sizes, 
                # or use sys.maxsize if you truly need every pixel
                with np.printoptions(threshold=1000, precision=4, suppress=True):
                    f.write(f"RAW INT8 VALUES:\n{val}\n")
                    
                    if scale != 0: # Show dequantized "Real" values for debugging
                        real_val = scale * (val.astype(np.float32) - zp)
                        f.write(f"DEQUANTIZED VALUES (Approx):\n{real_val}\n")
                        
            except ValueError:
                f.write(f"INDEX: {idx} | NAME: {name} | (Buffer not accessible)\n")

    print(f"Done! Intermediate tensors written to {args.result_file}")

if __name__ == "__main__":
    main()