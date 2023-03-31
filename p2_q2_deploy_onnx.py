import numpy as np
import onnxruntime
from tqdm import tqdm
import os
from PIL import Image
import argparse
import time

# TODO: create argument parser object

# TODO: add one argument for selecting VGG or MobileNet-v1 models

# TODO: Modify the rest of the code to use those arguments correspondingly

# Argument parser
parser = argparse.ArgumentParser(
    description='ECE361E HW3 - Deploy ONNX code')
# Define the model to be deployed
parser.add_argument('--model', type=str, default='vgg11',
                    help='Model to be deployed')
args = parser.parse_args()

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
deployed_model = args.model

if deployed_model == 'vgg11':
    PATH = './models/vgg11_rpi_pt.onnx'
elif deployed_model == 'vgg16':
    PATH = './models/vgg16_rpi_pt.onnx'
else:
    PATH = './models/mobilenet_rpi_pt.onnx'

onnx_model_name = PATH  # TODO: insert ONNX model name, essentially the path to the onnx model

# Create Inference session using ONNX runtime
sess = onnxruntime.InferenceSession(onnx_model_name)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)

# Mean and standard deviation 
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
acc_counter = 0
start_inference = time.time()
for filename in tqdm(os.listdir("/home/student/HW3_files/test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("/home/student/HW3_files/test_deployment", filename)).resize((32, 32)) as img:
        print("Image shape:", np.float32(img).shape)

        # normalize image
        input_image = (np.float32(img) / 255. - mean) / std
        
        # Add the Batch axis in the data Tensor (C, H, W)
        input_image = np.expand_dims(np.float32(input_image), axis=0)

        # change the order from (B, H, W, C) to (B, C, H, W)
        input_image = input_image.transpose([0, 3, 1, 2])
        
        print("Input Image shape:", input_image.shape)

        # Run inference and get the prediction for the input image
        pred_onnx = sess.run(None, {input_name: input_image})[0]

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        # TODO: compute test accuracy of the model

        # get real label
        image_label = filename.split(sep='_')[1].split(sep='.')[0]
        print("inference of image: " + filename)
        print("Image Label is: " + image_label + " and the predicted class is: " + pred_class)

        if(pred_class == image_label):
            acc_counter += 1

finish_inference = time.time()

total_inference_time = finish_inference - start_inference

print("Table Content")
print("Total time of the inference: %.4f seconds" %(total_inference_time))
print("Accuracy of the inference: %.4f percent" %(acc_counter/100))
# Mc1 Idle mem is 256 (used)
# Mc1 VGG11 mem is 315
# Mc1 VGG16 mem is 337
# Rpi Idle mem 85 - available 759
# Rpi VGG11 mem is 151
# Rpi VGG16 mem is 171



## Time
# Mc1 VGG11 - 535 segundos
# Mc1 VGG16 - 992 segundos
