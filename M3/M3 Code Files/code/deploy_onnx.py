import numpy as np
import onnxruntime
from tqdm import tqdm
import os
from PIL import Image
import argparse
import subprocess
import time
# TODO: create argument parser object

# TODO: add one argument for selecting VGG or MobileNet-v1 models

# TODO: Modify the rest of the code to use those arguments correspondingly

parser = argparse.ArgumentParser(
    description='ECE361E Project - Deploy ONNX code')
# Define the model to be deployed
parser.add_argument('--model', type=str, default='mobilenet_v1',
                    help='Model to be deployed')
args = parser.parse_args()
deployed_model = args.model

if args.model == 'vgg16':
     onnx_model_name = "vgg16_model_pt.onnx"
elif args.model == 'mobilenet_v1':
     onnx_model_name = "mobilenet_v1_model_pt.onnx"
else:
     raise ValueError("Invalid model name: " + args.model)

# rest of the code

#if deployed_model == 'mobilenet_v1':
    #PATH = './models/mobilenet_v1_rpi_pt.onnx'

onnx_model_name = "mobilenet_v1_model_pt.onnx"  # TODO: insert ONNX model name, essentially the path to the onnx model

# Create Inference session using ONNX runtime
sess = onnxruntime.InferenceSession(onnx_model_name)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
#print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
#print("Input shape :", input_shape)

# Mean and standard deviation 
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

num_correct = 0
total_images = 0
total_time = 0
# Start get_power_temp_rpi.py script in the background
power_temp = subprocess.Popen(["python3", "get_power_temp_rpi.py", "--csv_loc", "mobilenet_readings.csv"])
# The test_deployment folder contains all 10.000 images from the testing dataset of CIFAR10 in .png format
for filename in tqdm(os.listdir("/home/student/Project_files/test_deployment")):
    # Take each image, one by one, and make inference
    with Image.open(os.path.join("/home/student/Project_files/test_deployment", filename)).resize((32, 32)) as img:
        #print("Image shape:", np.float32(img).shape)

        # normalize image
        input_image = (np.float32(img) / 255. - mean) / std
        
        # Add the Batch axis in the data Tensor (C, H, W)
        input_image = np.expand_dims(np.float32(input_image), axis=0)

        # change the order from (B, H, W, C) to (B, C, H, W)
        input_image = input_image.transpose([0, 3, 1, 2])
        
        #print("Input Image shape:", input_image.shape)
        start_time = time.time()
        # Run inference and get the prediction for the input image
        pred_onnx = sess.run(None, {input_name: input_image})[0]
        end_time = time.time()
        total_time += (end_time - start_time)
        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        # TODO: compute test accuracy of the model 
        class1 = filename.split("_")[1]
        #print(class1.split(".")[0])
        if pred_class == class1.split(".")[0]:
            
            num_correct += 1
        total_images += 1

accuracy = num_correct / total_images
print("Test accuracy: {:.2%}".format(accuracy))
power_temp.kill()
print("Total Inference Time: ", total_time)
