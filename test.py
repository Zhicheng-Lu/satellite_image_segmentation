import os
from extra_functions import make_prediction, read_image, evaluate_accuracy
import numpy as np
import cv2

import tensorflow as tf
from tf_unet.model import Model

import datetime

### PARAM LIST ###

# input data param ----------
mask 	 	=	-1		# Decide which class/classes to use. [0 - 9] for 1 class, -1 for 11 classes (5 for crop)
input_channels 	= 	3		# Number of input channels
output_classes 	= 	11		# Number of output classes
img_rows 	=	112		# Image patch size (row)
img_cols 	=	112		# Image patch size (column)
# ---------------------------

# u-net initialization params (later try to import this from pre-saved param file)
loss_function 	= 	"jaccard"	# cross_entropy, jaccard, DICE, cross_jac (cross_entropy+jaccard) 
cropping 	=	16 		# crop the label
blocks 		=	5 		# Number of "blocks"/"layer groups" for downsampling. That for upsampling is this number minus 1
layers 		=	2 		# Number of conv layers within one "block"/"layer group"
features_root 	= 	32 		# Number of filters in starting layer
filter_size 	=	3
pool_size 	=	2
padding 	=	"SAME"		# Padding method: "SAME". "Valid" currently not support
regularizer	= 	"None"		# Regularisation on weights: "None", ("L2" not yet implemented)
activation 	=	"elu" 		# Activation function for conv layers "elu", "relu", "Leaky_relu"
batch_norm	= 	True		# batch normalisation
upsampling 	= 	1  		# 0: bilinear, 1: nearest neighbour, 2: bicubic, 3: area, 4: deconv
summaries 	=	True
# ---------------------------

# Testing Params
gpu_id 		= 	"0"	# GPU ID (Currently not implemented and assumed to be 0
model_path 	= 	"./Pretrained_11_Classes"
#model_path 	= 	"./Pretrained_Crop_Class"
#model_path	=	"./training_data"
output_path	= 	"./Results"
test_ids 	=	['6110_3_1']
epsilon	= 	1e-12	# Small constant for calculate IoU
# ---------------------------

### END OF PARAM LIST ###

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

# create output directory if not exists
output_folder = os.path.join(output_path, os.path.basename(model_path))
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# initialize model (u-net)
model = Model(channels=input_channels, 
    n_class=output_classes,
    img_rows=img_rows,
    img_cols=img_cols,
    is_train=False,
    cost=loss_function,
    cropping = cropping,
    batch_norm=batch_norm,
    cost_kwargs={"regularizer": regularizer},
    blocks=blocks,
    layers=layers,
    features_root=features_root,
    filter_size=filter_size,
    pool_size=pool_size,
    pad=padding,
    summaries=summaries)

for image_id in test_ids:

    # Read in the testing image
    image = read_image(image_id, bands=input_channels)

    # Make prediction ( Also for flip and swap_axis)
    predicted_mask = make_prediction(model, model_path, image, input_size=(img_rows, img_cols),
                                     crop=cropping,
                                     num_masks=output_classes, num_channels=input_channels)

    image_v = flip_axis(image, 1)
    predicted_mask_v = make_prediction(model, model_path, image_v, input_size=(img_rows, img_cols),
                                       crop=cropping,
                                       num_masks=output_classes,
                                       num_channels=input_channels)

    image_h = flip_axis(image, 2)
    predicted_mask_h = make_prediction(model, model_path, image_h, input_size=(img_rows, img_cols),
                                       crop=cropping,
                                       num_masks=output_classes,
                                       num_channels=input_channels)

    image_s = image.swapaxes(1, 2)
    predicted_mask_s = make_prediction(model, model_path, image_s, input_size=(img_rows, img_cols),
                                       crop=cropping,
                                       num_masks=output_classes,
                                       num_channels=input_channels)

    # combine the predictions in different image angles
    new_mask = np.power(predicted_mask * flip_axis(predicted_mask_v, 1) * flip_axis(predicted_mask_h, 2) * predicted_mask_s.swapaxes(1, 2), 0.25)

    # Evaluate Accuracy
    evaluate_accuracy(output_folder, new_mask, image_id, None, mask, epsilon)


