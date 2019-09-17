"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/6/19 16:12
"""

import tensorflow as tf
import numpy as np
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def view_parameter(model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    # print("ckpt :",ckpt)
    ckpt_path = ckpt.model_checkpoint_path

    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    param_dict = reader.get_variable_to_shape_map()
    # importing graph
    reader = tf.train.NewCheckpointReader(ckpt_path)
    all_variables = reader.get_variable_to_shape_map()
    print(all_variables)
    # can be view data and data sizes
    for key, val in param_dict.items():
        try:
            print(key, val)
            data = reader.get_tensor(key)
            # print(data)
    #         print_tensors_in_checkpoint_file(ckpt_path, tensor_name=key, all_tensors=False, all_tensor_names=False)
        except:
            pass
    return all_variables,ckpt_path


def save_parameter_txt(model_dir=None):
    model_dir = "save/"
    ckpt = tf.train.get_checkpoint_state(model_dir)
    # print("ckpt :",ckpt)
    ckpt_path = ckpt.model_checkpoint_path
    # reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    # param_dict = reader.get_variable_to_shape_map()

    # importing graph
    reader = tf.train.NewCheckpointReader(ckpt_path)
    all_variables = reader.get_variable_to_shape_map()
    print(all_variables)

    pf = open('parameter.txt', 'w')
    # loop save non-None data in txt
    for key in all_variables.keys():
        # print(key,all_variables[key])
        parameter_data = reader.get_tensor(key)
        print('**************** save', key ,' succeed******************* shape:',parameter_data.shape)
        data_shape = parameter_data.shape
        pf.write(str(key))
        pf.write(',data shape:')
        pf.write(str(all_variables[key]))
        pf.write('\n')
        if len(data_shape) == 0:
            pf.write(str(parameter_data))

        # save 1-D data format
        if len(data_shape) == 1:
            pf.write('{')
            for i in range(parameter_data.shape[0]):
                pf.write(str(parameter_data[i]))
                if i < parameter_data.shape[0] - 1:
                    pf.write(',')
                else:
                    pf.write('}')
            pf.write('\n')

        # save 2-D data format
        if len(data_shape) == 2:
            pf.write('{')
            for i in range(parameter_data.shape[0]):
                for j in range(parameter_data.shape[1]):
                    pf.write(str(parameter_data[i][j]))
                    if i < parameter_data.shape[0] - 1:
                        pf.write(',')
                    else:
                        pf.write('}')
            pf.write('\n')

        # save 4-D data format
        if len(data_shape) == 4:
            pf.write('{')
            for i in range(parameter_data.shape[0]):
                for j in range(parameter_data.shape[1]):
                    for k in range(parameter_data.shape[2]):
                        for l in range(parameter_data.shape[3]):
                            pf.write(str(parameter_data[i][j][k][l]))
                            if i < parameter_data.shape[0] - 1:
                                pf.write(',')
                            else:
                                pf.write('}')
            print('\n')
    pf.close()

def weights_csc_matrix(weights_matrix, mask_matrix):
    """
    Compressed Sparse Column format using csc;
    """
    weights_matrix = np.multiply(weights_matrix,mask_matrix)
    row_data = []
    col_data = []
    weights_data = []
    # print(weights_matrix)

    for i in range(weights_matrix.shape[0]):
        for j in range(weights_matrix.shape[1]):
            if weights_matrix[i][j] > 0:
                row_data.append(i)
                col_data.append(j)
                weights_data.append(weights_matrix[i][j])
            else:
                continue
    # print(row_data,col_data,weights_data)
    return row_data,col_data,weights_data


def aim_save(aim_key,model_dir=None):
    model_dir = "save/"
    ckpt = tf.train.get_checkpoint_state(model_dir)
    ckpt_path = ckpt.model_checkpoint_path
    # importing graph
    reader = tf.train.NewCheckpointReader(ckpt_path)
    all_variables = reader.get_variable_to_shape_map()
    aim_file = aim_key+'_parameter.txt'
    pf = open(aim_file, 'w')

    # loop save non-None data in txt
    parameter_data = reader.get_tensor('fully_connected/weights')
    mask_data = reader.get_tensor('fully_connected/mask')
    print(parameter_data.shape)
    weight_r,weight_l,weight_data = weights_csc_matrix(parameter_data,mask_data)
    print(len(weight_r),len(weight_l),len(weight_data))

    pf.write('#define WEIGHT_DATA1 ')
    pf.write('{')
    for i in range(len(weight_data)):
        pf.write(str(weight_data[i]))
        if i < len(weight_data)-1:
            pf.write(',')
        else:
            pf.write('}')
    pf.write('\n')

    pf.write('#define WEIGHT_R1')
    pf.write('{')
    for i in range(len(weight_r)):
        pf.write(str(weight_r[i]))
        if i < len(weight_r)-1:
            pf.write(',')
        else:
            pf.write('}')
    pf.write('\n')

    pf.write('#define WEIGHT_L1 ')
    pf.write('{')
    for i in range(len(weight_l)):
        pf.write(str(weight_l[i]))
        if i < len(weight_data)-1:
            pf.write(',')
        else:
            pf.write('}')



if __name__ == '__main__':
    # data = [[1,1,1],[1,1,1],[1,1,3]]
    # mask = [[0,1,0],[1,0,1],[0,0,1]]
    # __csc_matrix(data,mask)
    # save_parameter_txt()

    aim_save('fully_connected_weights')
