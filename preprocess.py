from canny2image_TRT import hackathon
import os
if not os.path.exists('unet-onnx'):
    os.makedirs('unet-onnx')

hk = hackathon()
hk.initialize()