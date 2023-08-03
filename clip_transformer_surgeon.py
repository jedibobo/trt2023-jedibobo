from collections import OrderedDict
from copy import deepcopy
import numpy as np
import onnx
import onnx_graphsurgeon as gs

# onnxFilePath = "/home/"
#onnxFilePath = "./" # local host
sourceOnnx = "./sd_clip_subgraph.onnx"
destinationOnnx = "./sd_clip_subgraph_layernorm.onnx"

bConvertToStaticNetwork = False
bDebug = True

nWili = 0
bSimplifyOutput = True
bNot = False
nNot = 0
bNotV2 = True
nNotV2 = 0
bMaskPlugin = False
nMaskPlugin = 0
bLayerNormPlugin = True
nLayerNormPlugin = 0
bConstantFold = True
nConstantFold = 0
bHongwei = True
nHongwei = 0
bExpand = True
nExpand = 0
b2DMM = True
n2DMM = 0
bAttentionMM = False
nAttentionMM = 0

graph = gs.import_onnx(onnx.shape_inference.infer_shapes(onnx.load(sourceOnnx)))

# Round 2: Layer Normalization
if bLayerNormPlugin:
    for node in graph.nodes:
        if node.op == 'ReduceMean' and \
            node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
            node.o().o(0).op =='Pow' and node.o().o(1).op =='Div' and \
            node.o().o(0).o().op == 'ReduceMean' and \
            node.o().o(0).o().o().op == 'Add' and \
            node.o().o(0).o().o().o().op == 'Sqrt' and \
            node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1) and \
            node.o().o(0).o().o().o().o().o().op == 'Mul' and \
            node.o().o(0).o().o().o().o().o().o().op == 'Add':

            inputTensor = node.inputs[0]

            lastMultipyNode = node.o().o(0).o().o().o().o().o()
            index = ['weight' in i.name for i in lastMultipyNode.inputs].index(True)
            b = np.array(deepcopy(lastMultipyNode.inputs[index].values.tolist()), dtype=np.float32)
            constantB = gs.Constant("LayerNormB-" + str(nLayerNormPlugin), np.ascontiguousarray(b.reshape(-1)))  # MUST use np.ascontiguousarray, or TRT will regard the shape of this Constant as (0) !!!

            lastAddNode = node.o().o(0).o().o().o().o().o().o()
            index = ['bias' in i.name for i in lastAddNode.inputs].index(True)
            a = np.array(deepcopy(lastAddNode.inputs[index].values.tolist()), dtype=np.float32)
            constantA = gs.Constant("LayerNormA-" + str(nLayerNormPlugin), np.ascontiguousarray(a.reshape(-1)))

            inputList = [inputTensor, constantB, constantA]
            layerNormV = gs.Variable("LayerNormV-" + str(nLayerNormPlugin), np.dtype(np.float32), None)
            layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin), inputs=inputList, outputs=[layerNormV])
            graph.nodes.append(layerNormN)

            if lastAddNode.outputs[0] in graph.outputs:  # the last LayerNorm provide one of the graph's output, and do not unsqueeze to 4 dimension
                # oldLastAdd -> graph.outputs[0] ===> LayerNorm -> Squeeze -> graph.outputs[0]
                layerNormN.outputs[0].name = 'encoder_out'
                index = graph.outputs.index(lastAddNode.outputs[0])
                graph.outputs[index] = layerNormN.outputs[0]
            else:  # other LayerNorm contain the subsequent Squeeze operation
                for n in graph.nodes:
                    if lastAddNode.outputs[0] in n.inputs:
                        index = n.inputs.index(lastAddNode.outputs[0])
                        n.inputs[index] = layerNormN.outputs[0]

                lastAddNode.outputs = []

            nLayerNormPlugin += 1
            continue

# if b2DMM:
#     for node in graph.nodes:
#         if node.op == 'MatMul' and node.name != 'MatMul_61' and \
#             node.o().op == 'Add' and \
#             node.o().o().op == 'Sigmoid' and \
#             node.o().o().o().op == 'Mul' and \
#             node.o().o().o().o().op == 'MatMul' and \
#             node.o().o().o().o().o().op == 'Add' and \
#             node.o().o().o().o().o().o().op == 'Mul':

#             reshape1V = gs.Variable("wiliReshape1V-" + str(n2DMM), np.dtype(np.float32), ['B*t4', 256])
#             reshape1N = gs.Node("Reshape", "wiliReshape1N-" + str(n2DMM), inputs=[node.inputs[0], bt4Comma256Tensor], outputs=[reshape1V])
#             graph.nodes.append(reshape1N)
#             n2DMM += 1

#             node.inputs[0] = reshape1V

#             lastNode = node.o().o().o().o().o().o()  # Mul[0.5]

#             reshape2V = gs.Variable("wiliReshape2V-" + str(n2DMM), np.dtype(np.float32), ['B', 't4', 256])
#             reshape2N = gs.Node("Reshape", "wiliReshape2N-" + str(n2DMM), inputs=[lastNode.inputs[0], bCommat4Comma64Tensor], outputs=[reshape2V])
#             graph.nodes.append(reshape2N)
#             n2DMM += 1

#             lastNode.inputs[0] = reshape2V
# if bAttentionMM:
#     for node in graph.nodes:
#         if node.op == 'LayerNorm' and node.name == int(node.name[11:]) % 5 == 1:
#             qM = node.o(1).inputs[1].values
#             qB = node.o(1).o().inputs[0].values
#             kM = node.o(2).inputs[1].values
#             kB = node.o(2).o().inputs[0].values
#             vM = node.o(3).inputs[1].values
#             vB = node.o(3).o().inputs[0].values
#             bigFactor = np.concatenate([qM,kM,vM],axis=1)
#             bigBias = np.concatenate([qB,kB,vB],axis=0)

#             bigFactorTensor = gs.Constant("bigFactorTensor" + str(nAttentionMM), np.ascontiguousarray(bigFactor))
#             bigBiasTensor = gs.Constant("bigBiasTensor" + str(nAttentionMM), np.ascontiguousarray(bigBias))
#             nAttentionMM += 1

#             qReshapeN = node.o(1).o().o()
#             kReshapeN = node.o(2).o().o()
#             vReshapeN = node.o(3).o().o()

#             matMulV = gs.Variable("wiliMatMul1V-" + str(nAttentionMM), np.dtype(np.float32), ['B*t4', 256*3])
#             matMulN = gs.Node("MatMul", "wiliMatMulN-" + str(nAttentionMM), inputs=[node.outputs[0], bigFactorTensor], outputs=[matMulV])
#             graph.nodes.append(matMulN)
#             nAttentionMM += 1

#             addV = gs.Variable("wiliAddV-" + str(nAttentionMM), np.dtype(np.float32), ['B*t4', 256*3])
#             addN = gs.Node("Add", "wiliAddN-" + str(nAttentionMM), inputs=[matMulV, bigBiasTensor], outputs=[addV])
#             graph.nodes.append(addN)
#             nAttentionMM += 1

#             split0V = gs.Variable("wiliSplit0V-" + str(nAttentionMM), np.dtype(np.float32), ['B*t4', 256])
#             split1V = gs.Variable("wiliSplit1V-" + str(nAttentionMM), np.dtype(np.float32), ['B*t4', 256])
#             split2V = gs.Variable("wiliSplit2V-" + str(nAttentionMM), np.dtype(np.float32), ['B*t4', 256])
#             splitN = gs.Node("Split", "wiliSplitN-" + str(nAttentionMM), inputs=[addV], outputs=[split0V,split1V,split2V], attrs=OrderedDict([('axis', 1)]))
#             graph.nodes.append(splitN)
#             nAttentionMM += 1

#             qReshapeN.inputs[0] = split0V
#             qReshapeN.inputs[1] = bCommat4Comma4Comma64Tensor
#             kReshapeN.inputs[0] = split1V
#             kReshapeN.inputs[1] = bCommat4Comma4Comma64Tensor
#             vReshapeN.inputs[0] = split2V
#             vReshapeN.inputs[1] = bCommat4Comma4Comma64Tensor

graph.cleanup()
onnx.save(gs.export_onnx(graph), destinationOnnx)

print("finish encoder onnx-graphsurgeon!")
print("%4d Not" %nNot)
print("%4d NotV2" %nNotV2)
print("%4d mask" %nMaskPlugin)
print("%4d LayerNormPlugin" %nLayerNormPlugin)
print("%4d ShapeOperation" %nConstantFold)
print("%4d Hongwei" %nHongwei)
print("%4d Expand" %nExpand)
print("%4d 2DMM" %n2DMM)
print("%4d Wili" %nWili)
print("%4d AttentionMM" %nAttentionMM)