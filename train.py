''' Parameter tuning framework for Satellite Image Segmentation project 

For Downer Group/Envista

Code reference
- Valdimir Iglovikov		 	https://github.com/ternaus/kaggle_dstl_submission
- Jakeret 				https://github.com/jakeret/tf_unet
- Rogerxujiang				https://github.com/rogerxujiang/dstl_unet/tree/master/data
'''

import numpy as np
import tensorflow as tf

from tf_unet.model import Model
from tf_unet.trainer import Trainer
from batch_generator import Batch_generator
from batch_generator_ram import Batch_generator_ram
from extra_functions import read_image, read_GT_Masks
import os
import psutil

import datetime

### PARAM LIST ###

# input data param ----------
data_path = 		"../data"
data_file = 		"train_3.h5"
image_ids =     ['6010_1_2','6010_4_2','6010_4_4','6040_1_0','6040_1_3','6040_2_2','6040_4_4','6060_2_3','6070_2_3',
                 '6090_2_0','6100_1_3','6100_2_2','6100_2_3','6110_1_2','6110_4_0','6120_2_0','6120_2_2',
                 '6140_1_2','6140_3_1','6150_2_3','6160_2_1','6170_0_4','6170_2_4','6170_4_1']
mask = 			-1		# Decide which class/classes to use. [0 - 9] for 1 class, -1 for 11 classes
input_channels = 	3		# Number of input channels
output_classes = 	11		# Number of output class/classes
img_rows = 		112		# Image patch size (row)
img_cols = 		112		# Image patch size (column)
# ---------------------------

# Pre-processing/augmentation
horizontal_flip = 	True
vertical_flip 	= 	True
swap_axis 	=	True
colour_space 	=	"RGB" 		# (not yet implemented)
stretch_band 	= 	True  		# (not yet implemented)
brightness 	= 	True		# (not yet implemented)
# ---------------------------

# u-net initialization params
model_type 	= 	"u-net"		# Model type: "u-net"
loss_function 	= 	"cross_entropy"	# cross_entropy, jaccard, DICE, cross_jac (cross_entropy+jaccard)
blocks 		=	5 		# Number of "blocks"/"layer groups" for downsampling. That for upsampling is this number minus 1
layers 		=	2 		# Number of conv layers within one "block"/"layer group"
cropping 	=	2*blocks*layers # crop the label
features_root 	= 	32 		# Number of filters in starting layer
filter_size 	=	3		# filter size 3x3, 5x5, 7x7, ..
pool_size 	=	2		# pool size >= 2
padding 	=	"SAME"		# Padding method: "SAME". "Valid" currently not support
regularizer	= 	"None"		# Regularisation on weights: "None", ("L2" not yet implemented)
activation 	=	"elu" 		# Activation function for conv layers "elu", "relu", "Leaky_relu"
batch_norm	= 	True		# batch normalisation
upsampling 	= 	1  		# 0: bilinear, 1: nearest neighbour, 2: bicubic, 3: deconv/transposed conv
initializer 	= 	"Normal"	# Initializer for weights: "Normal": truncated normal, "He": He initialization, "Xavier": Xavier initialization
summaries 	=	True
# ---------------------------

# trainer initialization params
batch_size 	= 	96
veri_batch_size =	64
optimizer 	= 	"momentum"  	# Momentum or adam
veri_image_id	=	'6110_3_1'	# name of image for verification

if optimizer == "momentum":
    op_param = {'learning_rate': 0.1, 
                     'decay_rate': 1,  	# for exponential decay
                     'momentum': 0.2,
                     'steps': 25}	# step for decay
elif optimizer == "adam":
    op_param = {"learning_rate": 0.001}
# -----------------------------

# training params
model_save_path = 	"./training_data/"
nb_epoch 	=	2
batch_pass 	=	2
dropout 	=	1		# Dropout is not recommended in Conv net
display_step 	=	1
restore 	=	False		# Restore weights from pre-trained model
write_graph 	=	False
prediction_path = 	None 		# The path to save images during training, None for not save
save_image_dims = 	[800, 800]	# Dimension of the images saved for tensorboard
gpu_id		= 	"0"		# "0", "1","all" 
# -----------------------------

### END OF PARAM LIST ###


assert (img_rows / pool_size ** (blocks - 1) % 1 == 0), "Rows need to be mutiples of pool_size ** (block number - 1)"
assert (img_cols / pool_size ** (blocks - 1) % 1 == 0), "Columns need to be mutiples of pool_size ** (block number - 1)"
assert mask in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "Invalid Mask Value"

if __name__ == '__main__':

    now = datetime.datetime.now()

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    # initialize model (u-net)
    model = Model(model_type=model_type,
            channels=input_channels, 
	    n_class=output_classes,
            img_rows=img_rows,
            img_cols=img_cols,
            is_train=True,
            cost=loss_function,
            cropping = cropping,
            cost_kwargs={"regularizer": regularizer},
	    blocks=blocks,
            layers=layers,
	    features_root=features_root,
            filter_size=filter_size,
            pool_size=pool_size,
            pad=padding,
            activation=activation,
            batch_norm=batch_norm,
            upsampling=upsampling,
            initializer=initializer,
            summaries=summaries)


    print('[{}] Reading train...'.format(str(datetime.datetime.now())))

    # compare total memory and available memory to decide how to load data
    total_size = 0
    for i in range(len(image_ids)):
        total_size += os.path.getsize('../data/three_band/' + image_ids[i] + '.tif')
    available_size = psutil.virtual_memory().available

    #if total_size < available_size:
    if False:
        data = Batch_generator_ram(image_ids,
                                   img_rows=img_rows,
                                   img_cols=img_cols,
                                   num_channels=input_channels,
                                   cropping = cropping,
                                   horizontal_flip=horizontal_flip, 
                                   vertical_flip=vertical_flip, 
                                   swap_axis=swap_axis, 
                                   mask=mask)
    else:
        data = Batch_generator(os.path.join(data_path, data_file), 
                               img_rows=img_rows, 
                               img_cols=img_cols,
                               num_channels=input_channels,
                               cropping = cropping,
                               horizontal_flip=horizontal_flip, 
                               vertical_flip=vertical_flip, 
                               swap_axis=swap_axis, 
                               mask=mask)
#    exit(0)
    # initialize trainer
    trainer = Trainer(model, 
                      batch_size=batch_size, 
                      verification_batch_size = veri_batch_size, 
                      norm_grads=False, 
                      optimizer=optimizer,
                      mask=mask,
                      test_x=read_image(veri_image_id, input_channels),
                      test_y=read_GT_Masks(veri_image_id, output_classes, mask),
                      save_image_dims=save_image_dims,
                      opt_kwargs=op_param)

    # start training
    trainer.train(data, model_save_path,
                         training_iters=batch_pass,
                         epochs=nb_epoch,
                         dropout=dropout,# probability to keep units
                         display_step=display_step,
                         restore=restore,
                         write_graph=write_graph,
                         prediction_path=prediction_path,
                         gpu_id=gpu_id)

    # close .h5 file
    data.close()

