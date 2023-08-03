import onnx_graphsurgeon as gs
import onnx
import numpy as np

# load sd_clip_transformer.onnx 
graph = gs.import_onnx(onnx.load("sd_clip_transformer.onnx"))

# modify the shape of output to (1, 77, 768)
output_tensor = graph.outputs[0]    
output_tensor.shape = ["B", 77, 768]

output_tensor = graph.outputs[1]
output_tensor.shape = ["B", 768]

# find a node named /text/ArgMax
# node = graph.find_node_by_name("/text/ArgMax")
# # add a cast node before this node, converting from int64 to int32
# cast_node = gs.Node()

# save the modified model to sd_clip_transformer_onnx_reshape.onnx
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "sd_clip_transformer_onnx_reshape.onnx")
