# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Jul 28, 2016

author: jakeret
'''

import numpy as np
import logging
import os

import tensorflow as tf
import tensorlayer as tl


from tf_unet import util
from tf_unet.loss import (cross_entropy,jaccard_coef_loss,jaccard_coef_int_avg,jaccard_coef_int,get_loss)
from tf_unet.unet_model import unet

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Model(object):
    """
    A unet implementation

    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self, model_type="u-net", channels=3, batch_norm=False, n_class=1, img_rows=112, img_cols=112, is_train=True, cost="jaccard", cropping=0, cost_kwargs={"regularizer": None}, **kwargs):
        tf.reset_default_graph()

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.n_class = n_class
        self.crop = cropping
        self.batch_norm = batch_norm
        self.summaries = kwargs.get("summaries", True)

        self.x = tf.placeholder("float", shape=[None, img_rows, img_cols, channels], name="x")
        self.y = tf.placeholder("float", shape=[None, img_rows - 2 * cropping, img_cols - 2 * cropping, n_class], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_probability")  # dropout (keep probability)
        #self.is_train = tf.placeholder(tf.float32, name="dropout_probability")

        # create model 
        if model_type == "u-net":
            self.model = unet(self.x, self.keep_prob, channels, n_class, img_rows, img_cols, batch_norm, is_train, **kwargs)
        else:
            raise NameError("Model Type Not Defined")

        # obtain logit from tensorlayer model
        logits = self.model.outputs
        # crop the output to remove the effect of padding
        logits = logits[:,cropping:img_rows-cropping,cropping:img_cols-cropping,:]

        # calculate loss
        self.cost = get_loss(self, logits, cost, cost_kwargs)

        # (currently not used)
        #self.gradients_node = tf.gradients(self.cost, self.variables)

        # define function for tensorboard variables here
        with tf.name_scope("cross_entropy"):
            self.cross_entropy = cross_entropy(self.y, logits)

        with tf.name_scope("jaccard"):
            self.jaccard = jaccard_coef_int_avg(self.y, logits)

        #with tf.name_scope("predicted_image"):
        #    self.predicted_image = tf.placeholder("uint8", shape=[11, 3348, 3396, 1], name="pred")

        #with tf.name_scope("jaccard_background"):
        #    self.jaccard_background = jaccard_coef_int(self.y[:,:,:,-1], logits[:,:,:,-1])

        # later extend to multiclass
        with tf.name_scope("results"):
            if n_class == 1:
                self.predicter = tf.sigmoid(logits)
                self.pred = tf.round(self.predicter)
                self.correct_pred = tf.equal(self.pred, self.y)
            else:
                self.predicter = tf.nn.softmax(logits)
                self.pred = tf.argmax(self.predicter, -1)
                self.correct_pred = tf.equal(self.pred, tf.argmax(self.y, -1))

            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def predict(self, x_test, path=None, sess=None):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)

        """

        print(self.model)
        gpu_id = "0"
        close_flag = 0
        if sess == None:
            close_flag = 1
            # GPU config: put this to the main param list later
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            if gpu_id != "all":
                config.gpu_options.visible_device_list = gpu_id

            # initialise session and global variables
            sess = tf.Session(config=config)

            # Initialize variables
            tl.layers.initialize_global_variables(sess)

        # Restore model weights from previously saved model
        if path != None:
            self.restore(sess, os.path.join(path, 'model.npz'))

        batch = int(x_test.shape[0] / 128)
  
        # separate a large batch of data into smaller ones for testing
        prediction = np.empty((0, self.y.shape[1] ,self.y.shape[2]), dtype=np.float32)
        for i in range(batch+1):

            if i < batch:
                b = x_test[i*128:(i+1)*128, :, :, :]
                pred = tl.utils.predict(sess, self.model, b, self.x, self.pred, b.shape[0])
                pred = np.reshape(pred, (b.shape[0], self.y.shape[1] ,self.y.shape[2]))
                prediction = np.concatenate((prediction, pred), axis=0)
            else:
                b = x_test[i*128:, :, :, :]
                pred = tl.utils.predict(sess, self.model, b, self.x, self.pred, b.shape[0])
                pred = np.reshape(pred, (b.shape[0], self.y.shape[1] ,self.y.shape[2]))
                prediction = np.concatenate((prediction, pred), axis=0)  
        
        prediction = np.reshape(prediction, (x_test.shape[0], self.y.shape[1] ,self.y.shape[2], 1))

        if close_flag == 1:
            sess.close()

        return prediction

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """

        tl.files.save_npz(self.model.all_params, name= model_path + 'model.npz', sess=sess)
        logging.info("Model saved to file: %s" % model_path)

        return

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        tl.files.load_and_assign_npz(sess, model_path, self.model)
        #logging.info("Model restored from file: %s" % model_path)

        return

# Functions not used / maybe used later
def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step + 1)))) + (gradients[i] / (step + 1))

    return avg_gradients





