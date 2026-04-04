import numpy as np
import tflite_runtime.interpreter as tflite
import os

def to_hex(val, width):
    """Converts integer to hex string with specified bit width (Two's Complement)."""
    if val < 0:
        val = (1 << width) + val
    return f"{int(val):0{width//4}x}"

def extract_all_hdl_params(tflite_path, output_dir="tflite_quant_params"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    interpreter = tflite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()

    for tensor in details:
        name = tensor['name'].replace("/", "_").replace(":", "_")
        data = interpreter.get_tensor(tensor['index'])
        q = tensor['quantization_parameters']
        
        # Get Zero-Point(s)
        # Note: TFLite supports per-channel ZP, but weights are almost always per-tensor 0.
        zps = q['zero_points']
        
        # 1. WEIGHTS (INT8) + Weight Zero Point
        if len(tensor['shape']) == 4: 
            flat_w = data.flatten()
            write_coe(f"{output_dir}/{name}_w.coe", flat_w, 8)
            # Store the weight ZP (usually a single value)
            write_coe(f"{output_dir}/{name}_w_zp.coe", zps, 8)

            # 2. REQUANTIZATION: Multipliers (m0) and Shifts (n)
            scales = q['scales']
            m0_list, n_list = [], []
            for s in scales:
                m0, n = np.frexp(s)
                m0_int = int(round(m0 * (2**31)))
                m0_list.append(m0_int)
                n_list.append(abs(n))
            
            write_coe(f"{output_dir}/{name}_m0.coe", m0_list, 32)
            write_coe(f"{output_dir}/{name}_n.coe", n_list, 8)

        # 3. BIASES (INT32)
        elif len(tensor['shape']) == 1 and data.dtype == np.int32:
            write_coe(f"{output_dir}/{name}_b.coe", data, 32)
            
        # 4. ACTIVATION/INPUT TENSORS (To get Input/Output ZP)
        else:
            if len(zps) > 0:
                write_coe(f"{output_dir}/{name}_act_zp.coe", zps, 8)

def write_coe(path, data, width):
    with open(path, "w") as f:
        f.write("memory_initialization_radix=16;\nmemory_initialization_vector=\n")
        f.writelines([to_hex(x, width) + (",\n" if i < len(data)-1 else ";") for i, x in enumerate(data)])

extract_all_hdl_params("tl/tinyissimo_ptq_full_integer_quant.tflite")