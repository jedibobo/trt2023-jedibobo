echo "preprocess"

rm -rf *.onnx 
rm -rf *.engine
rm -rf unet-onnx/*

/usr/bin/python3 preprocess.py