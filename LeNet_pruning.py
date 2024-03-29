"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/5/27 10:50
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python.layers import layers
import time

class LeNet_Mode():
    """ create LeNet network use tensorflow
        LeNet network structure:
        (conv 5x5 32 ,pool/2)
        (conv 5x5 64, pool/2)
        (fc 100)=>=>(fc classes)
    """
    def conv_layer(self, data, ksize, stride, name, w_biases = False,padding = "SAME"):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            w_init = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable(name= name,shape= ksize, initializer= w_init)
            biases = tf.Variable(tf.constant(0.0, shape=[ksize[3]], dtype=tf.float32), 'biases')
        if w_biases == False:
            cov = tf.nn.conv2d(input= data, filter= w, strides= stride, padding= padding)
        else:
            cov = tf.nn.conv2d(input= data,filter= w, stride= stride,padding= padding) + biases
        return cov

    def pool_layer(self, data, ksize, stride, name, padding= 'VALID'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            max_pool =  tf.nn.max_pool(value= data, ksize= ksize, strides= stride,padding= padding)
        return max_pool

    def flatten(self,data):
        [a,b,c,d] = data.get_shape().as_list()
        ft = tf.reshape(data,[-1,b*c*d])
        return ft

    def fc_layer(self,data,name,fc_dims):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            data_shape = data.get_shape().as_list()
            w_init = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable(shape=[data_shape[1],fc_dims],name= 'w',initializer=w_init)
            # w = tf.Variable(tf.truncated_normal([data_shape[1], fc_dims], stddev=0.01),'w')
            biases = tf.Variable(tf.constant(0.0, shape=[fc_dims], dtype=tf.float32), 'biases')
            fc = tf.nn.relu(tf.matmul(data,w)+ biases)
        return fc

    def finlaout_layer(self,data,name,fc_dims):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            w_init = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable(shape=[data.shape[1],fc_dims],name= 'w',initializer=w_init)
            biases = tf.Variable(tf.constant(0.0, shape=[fc_dims], dtype=tf.float32), 'biases')
            # fc = tf.nn.softmax(tf.matmul(data,w)+ biases)
            fc = tf.matmul(data,w)+biases
        return fc

    def model_bulid(self, height, width, channel,classes):
        x = tf.placeholder(dtype= tf.float32, shape = [None,height,width,channel])
        y = tf.placeholder(dtype= tf.float32 ,shape=[None,classes])

        # conv 1 ,if image Nx465x128x1 ,(conv 5x5 32 ,pool/2)
        conv1_1 = tf.nn.relu(self.conv_layer(x,ksize=[5,5,channel,32],stride=[1,1,1,1],padding="SAME",name="conv1_1")) # Nx465x128x1 ==>   Nx465x128x32
        pool1_1 = self.pool_layer(conv1_1,ksize=[1,2,2,1],stride=[1,2,2,1],name="pool1_1") # N*232x64x32

        # conv 2,(conv 5x5 32)=>(conv 5x5 64, pool/2)
        conv2_1 = tf.nn.relu(self.conv_layer(pool1_1,ksize=[5,5,32,64],stride=[1,1,1,1],padding="SAME",name="conv2_1"))
        pool2_1 = self.pool_layer(conv2_1,ksize=[1,2,2,1],stride=[1,2,2,1],name="pool2_1") # Nx116x32x128

        # Flatten
        ft = self.flatten(pool2_1)

        # Dense layer,(fc 100)=>=>(fc classes) and prune optimize
        fc_layer1 = layers.masked_fully_connected(ft, 200)
        fc_layer2 = layers.masked_fully_connected(fc_layer1, 100)
        prediction = layers.masked_fully_connected(fc_layer2, 10)

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
        #  original Dense layer
        # fc1 = self.fc_layer(ft,fc_dims=100,name="fc1")
        # finaloutput = self.finlaout_layer(fc1,fc_dims=10,name="final")

        #  pruning op
        global_step = tf.train.get_or_create_global_step()
        reset_global_step_op = tf.assign(global_step, 0)
        # Get, Print, and Edit Pruning Hyperparameters
        pruning_hparams = pruning.get_pruning_hparams()
        print("Pruning Hyper parameters:", pruning_hparams)
        # Change hyperparameters to meet our needs
        pruning_hparams.begin_pruning_step = 0
        pruning_hparams.end_pruning_step = 250
        pruning_hparams.pruning_frequency = 1
        pruning_hparams.sparsity_function_end_step = 250
        pruning_hparams.target_sparsity = .9
        # Create a pruning object using the pruning specification, sparsity seems to have priority over the hparam
        p = pruning.Pruning(pruning_hparams, global_step=global_step)
        prune_op = p.conditional_mask_update_op()

        # optimize
        LEARNING_RATE_BASE = 0.001
        LEARNING_RATE_DECAY = 0.9
        LEARNING_RATE_STEP = 300
        gloabl_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE
                                                   , gloabl_steps,
                                                   LEARNING_RATE_STEP,
                                                   LEARNING_RATE_DECAY,
                                                   staircase=True)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step)

        # prediction
        prediction_label = prediction
        correct_prediction = tf.equal(tf.argmax(prediction_label,1),tf.argmax(y,1))
        accurary = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
        correct_times_in_batch = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.int32))

        return dict(
            x=x,
            y=y,
            optimize=optimize,
            correct_prediction=prediction_label,
            correct_times_in_batch=correct_times_in_batch,
            cost=loss,
            accurary = accurary,
            prune_op = prune_op
        )

    def init_sess(self):
        init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(init)

    def train_network(self,graph,x_train,y_train):
        # Tensorfolw Adding more and more nodes to the previous graph results in a larger and larger memory footprint
        # reset graph
        # tf.reset_default_graph()
        # prune op
        self.sess.run(graph['prune_op'])
        self.sess.run(graph['optimize'], feed_dict={graph['x']:x_train, graph['y']:y_train})
        # print("cost: ",self.sess.run(graph['cost'],feed_dict={graph['x']:x_train, graph['y']:y_train}))
        # print("accurary: ",self.sess.run(graph['accurary'],feed_dict={graph['x']:x_train, graph['y']:y_train}))

    def save_model(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess,"save/model.ckpt")

    def load_data(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        g = self.model_bulid(28, 28, 1, 10)
        # Build the model first, then initialize it, just once
        start = time.time()
        self.init_sess()
        for epoch in range(30):
            for i in range(1500):
                batch_xs, batch_ys = mnist.train.next_batch(1000)
                batch_xs = np.reshape(batch_xs,[-1,28,28,1])
                # sess.run(g['prune_op'], feed_dict={g['x']:batch_xs, g['y']:batch_ys})
                self.train_network(g,batch_xs,batch_ys)
                print("Train cost accurary print:","cost: ", self.sess.run(g['cost'], feed_dict={g['x']: batch_xs, g['y']: batch_ys}), "accurary: ",
                      self.sess.run(g['accurary'], feed_dict={g['x']: batch_xs, g['y']: batch_ys}))
                if i % 30==0:
                    batch_xs_test, batch_ys_test = mnist.test.next_batch(1000)
                    batch_xs_test = np.reshape(batch_xs_test,[-1,28,28,1])
                    acc = self.sess.run(g['accurary'],feed_dict={g['x']:batch_xs_test, g['y']:batch_ys_test})
                    print("******Test cost accurary print******:","cost: ",self.sess.run(g['cost'],feed_dict={g['x']:batch_xs_test, g['y']:batch_ys_test}),"accurary: ",
                          self.sess.run(g['accurary'],feed_dict={g['x']:batch_xs_test, g['y']:batch_ys_test}))
                    print("Sparsity of layers (should be 0)", self.sess.run(tf.contrib.model_pruning.get_weight_sparsity()))
                    if acc > 0.7:
                        self.save_model()

        end = time.time()
        print(end-start,"min times")

if __name__ == '__main__':
    LeNet = LeNet_Mode()
    LeNet.load_data()
