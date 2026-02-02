import onnx
import argparse

def inspect_node(model_path, node_name):
    try:
        model = onnx.load(model_path)
        print(f"Searching for node: {node_name}")
        
        target_node = None
        for node in model.graph.node:
            if node.name == node_name:
                target_node = node
                break
        
        if target_node:
            print(f"Found Node: {target_node.name}")
            print(f"OpType: {target_node.op_type}")
            print(f"Inputs: {target_node.input}")
            print(f"Outputs: {target_node.output}")
            
            # Print input types if possible (requires looking up initializers or value info)
            print("\nInput details:")
            for inp_name in target_node.input:
                # Check initializers
                init = next((x for x in model.graph.initializer if x.name == inp_name), None)
                if init:
                    print(f"  Input '{inp_name}' is Initializer: DataType={init.data_type}, Shape={init.dims}")
                else:
                    # Check value info
                    vi = next((x for x in model.graph.value_info if x.name == inp_name), None)
                    if vi:
                        print(f"  Input '{inp_name}' is ValueInfo: {vi.type}")
                    else:
                        print(f"  Input '{inp_name}' is Graph Input or Intermediate")

        else:
            print("Node not found.")

    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="license-plate-finetune-v1l-int8-dynamic.onnx", help="Path to ONNX model")
    parser.add_argument("--node", type=str, default="/model.0/conv/Conv_quant", help="Name of node to inspect")
    args = parser.parse_args()
    
    inspect_node(args.model, args.node)
