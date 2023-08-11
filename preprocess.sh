echo "preprocess"

rm -rf *.onnx 
rm -rf *.engine
rm -rf unet-onnx/*
rm -rf engine/*

/usr/bin/python3 preprocess.py