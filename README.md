# EMface: Detecting hard faces by exploring receptive field pyramids
This is an official Pytorch implementation of paper entitled "EMface: Detecting hard faces by exploring receptive field pyramids".
# Requirement
Python 3.5
Pytorch 0.4+
opencv
numpy
# Prepare data
download wider face dataset, and modify the datapath in data/config.py.
# Train
python train_Res_RFP.py
# Evaluation
download the [pretrained model](https://drive.google.com/file/d/1Uv5JpjrVW06iVjrjjF3HwCpjwVUqy9IZ/view?usp=sharing), and test on wider face.
