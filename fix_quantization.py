import onnx
import numpy as np
from onnx import numpy_helper

def fix_model(model_path, output_path):
    model = onnx.load(model_path)
    print(f"Loaded {model_path}")
    
    # Create a map of initializers for easy access
    initializers = {init.name: init for init in model.graph.initializer}
    
    converted_count = 0
    
    for node in model.graph.node:
        if node.op_type == "ConvInteger":
            # Inputs: [x, w, x_zp, w_zp]
            if len(node.input) < 4:
                continue
                
            w_name = node.input[1]
            w_zp_name = node.input[3]
            
            if w_name in initializers and w_zp_name in initializers:
                w_init = initializers[w_name]
                w_zp_init = initializers[w_zp_name]
                
                # Check if INT8 (DataType 3)
                if w_init.data_type == 3: # TensorProto.INT8
                    print(f"Converting node {node.name} weights to UINT8...")
                    
                    # Convert Weights
                    w_data = numpy_helper.to_array(w_init)
                    w_data_u8 = (w_data.astype(np.int16) + 128).astype(np.uint8)
                    
                    new_w_init = numpy_helper.from_array(w_data_u8, name=w_name)
                    
                    # Convert Zero Point
                    w_zp_data = numpy_helper.to_array(w_zp_init)
                    w_zp_data_u8 = (w_zp_data.astype(np.int16) + 128).astype(np.uint8)
                    
                    new_w_zp_init = numpy_helper.from_array(w_zp_data_u8, name=w_zp_name)
                    
                    # Update Initializers in the list (replace old ones)
                    model.graph.initializer.remove(w_init)
                    model.graph.initializer.extend([new_w_init])
                    
                    model.graph.initializer.remove(w_zp_init)
                    model.graph.initializer.extend([new_w_zp_init])
                    
                    converted_count += 1

    if converted_count > 0:
        print(f"Converted {converted_count} ConvInteger nodes.")
        onnx.save(model, output_path)
        print(f"Saved fixed model to {output_path}")
    else:
        print("No INT8 ConvInteger nodes found to convert.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="license-plate-finetune-v1l-int8-dynamic.onnx")
    parser.add_argument("--output", type=str, default="fixed_model.onnx")
    args = parser.parse_args()
    
    fix_model(args.model, args.output)
