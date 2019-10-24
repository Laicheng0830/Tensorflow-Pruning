"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Created on 2019/9/18 16:32
"""
import os
import numpy as np
import tensorflow as tf
from audio_processing import *
from deeplearning_api import *
import matplotlib.pylab as plt


def search_file(root_dir, data_type):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[-1].lower() == data_type:
                file_name = root + file
    # print(file_name)
    return file

def find_file(root_dir, data_type):
    file_path = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[-1].lower() == data_type:
                file_path.append(file)
    # print(file_path)
    return file_path

def search_key(keys,keys_key1, keys_key2):
    for key in keys:
        if keys_key1 in key.split('/') and keys_key2 in key.split('/'):
            return key
        else:
            continue

def scale_feature(data, mean, std):
    for i in range(data.shape[0]):

        if len(data.shape) == 3:
            for j in range(data.shape[1]):
                data[i, j, :] = (data[i, j, :] - mean) / std
        if len(data.shape) == 2:
            data[i, :] = (data[i, :] - mean) / std
    return data

def relu_activate(data_in):
    for i in range(len(data_in)):
        if data_in[i] < 0:
            data_in[i] = 0
    return data_in

def weight_pruning(data_in, spares_rate):
    """
    weights pruning
    :param data_in: 2-D weight
    :param spares_rate: float
    :return: 2-D weight
    """
    # [r, l] = data_in.shape
    # data_sort = np.sort(abs(data_in.reshape([r*l])))
    # data_zeros = data_sort[0:int(r*l*spares_rate)]
    # for i in range(r):
    #     for j in range(l):
    #         if abs(data_in[i][j]) <= data_zeros[-1]:
    #             data_in[i][j] = 0
    # return data_in

    [r, l] = data_in.shape
    data_sort = np.sort(abs(data_in.reshape([r*l])))
    data_min = data_sort[int(r*l*spares_rate)]
    data_in[abs(data_in) < data_min] = 0

    # count None-Zero
    none_z = np.nonzero(data_in)
    z_sizes = len(data_in[none_z])
    rate = z_sizes/(r*l)
    print(1-rate, "pruning rate")
    return data_in

def statistic_weight(modle_dir):
    parameter = return_parameter(model_dir)
    fcl_1_w = parameter['generator_model/fcl_1/w']
    fcl_1_b = parameter['generator_model/fcl_1/b']
    fcl_2_w = parameter['generator_model/fcl_2/w']
    fcl_2_b = parameter['generator_model/fcl_2/b']
    fcl_3_w = parameter['generator_model/fcl_3/w']
    fcl_3_b = parameter['generator_model/fcl_3/b']
    fcl_4_w = parameter['generator_model/fcl_4/w']
    fcl_4_b = parameter['generator_model/fcl_4/b']

    fcl_1_w_ = fcl_1_w.reshape([7*257*2048])
    fcl_2_w_ = fcl_2_w.reshape([2048*2048])
    fcl_3_w_ = fcl_3_w.reshape([2048*2048])
    fcl_4_w_ = fcl_4_w.reshape([2048*257])
    print(np.median(abs(fcl_1_w_)), np.median(abs(fcl_2_w_)),np.median(abs(fcl_3_w_)),np.median(abs(fcl_4_w_)),"all parameter mean")
    print(np.min(abs(fcl_1_w)), np.min(abs(fcl_2_w)),np.min(abs(fcl_3_w)),np.min(abs(fcl_4_w)),"mean")

    plt.figure(1)
    plt.subplot(141);plt.imshow(fcl_1_w)
    plt.subplot(142);plt.imshow(fcl_2_w)
    plt.subplot(143);plt.imshow(fcl_3_w)
    plt.subplot(144);plt.imshow(fcl_4_w)
    plt.colorbar()

    plt.figure(2)
    plt.subplot(221);plt.hist(fcl_1_w_,200)
    plt.subplot(222);plt.hist(fcl_2_w_,200)
    plt.subplot(223);plt.hist(fcl_3_w_,200)
    plt.subplot(224);plt.hist(fcl_4_w_,200)
    plt.show()

##  define sdk end

def fully_connected_1D(data, weight, biases, in_data_dim, out_data_dim, bias = True):
    """
    data [a*b*c]
    weight [a*b*c*out_data_dim]
    """
    fully_out = []
    out_temp = 0
    if bias == False:
        for i in range(out_data_dim):
            for j in range(in_data_dim):
                out_temp += data[j] * weight[i*in_data_dim+j]
            fully_out.append(out_temp)
    else:
        for i in range(out_data_dim):
            for j in range(in_data_dim):
                out_temp += data[j] * weight[i*in_data_dim+j]
            out_temp +=  biases[i]
            fully_out.append(out_temp)
    # print(fully_out)
    return fully_out


def fully_connected(data, weight, biases, in_data_dim, out_data_dim, bias = True):
    """
    fully connected 2-D
    :param data: 1-D data in
    :param weight: 2-D [data_in ,data_out]
    :param biases: 1-D data_in
    :param in_data_dim: int
    :param out_data_dim: int
    :param bias: bool
    :return:1-D data_out
    """
    fully_out = []
    # print(data.shape, weight.shape,biases.shape,in_data_dim,out_data_dim,"data weight biases i_dim out_dim")
    if bias == False:
        for i in range(out_data_dim):
            out_temp = np.sum(data * weight[:,i])
            fully_out.append(out_temp)
    else:
        for i in range(out_data_dim):
            out_temp = np.sum(data * weight[:,i]) + biases[i]
            fully_out.append(out_temp)
    return fully_out


def return_parameter(model_dir):
    ckpt = search_file(model_dir, '.meta')
    ckpt_path = model_dir + ckpt.split('.')[0]

    reader = tf.train.NewCheckpointReader(ckpt_path)
    all_variables = reader.get_variable_to_shape_map()
    # print(all_variables)
    # loop save non-None data in txt
    parameter_dict = {}
    for key in all_variables.keys():
        # print(key,all_variables[key])
        parameter_data = reader.get_tensor(key)
        if 'generator' in key.split('_'):
            # print('**************** save', key ,' succeed******************* shape:',parameter_data.shape)
            parameter_dict[key] = parameter_data
    # print(parameter_dict.keys())
    return parameter_dict


def model_prediction(data_in, parameter):
    [r, l, p, q] = data_in.shape
    data_in = np.reshape(data_in,[r, l*p])
    # plt.plot(data_in[4])
    # plt.show()
    prediction_data = np.zeros((r,p))
    ##   model parameter config
    # # fcl_1_w = np.reshape(parameter[search_key(parameter.keys(), 'fcl_1', 'w')], [1799*2048])
    # fcl_1_w = parameter[search_key(parameter.keys(), 'fcl_1', 'w')]
    # fcl_1_b = parameter[search_key(parameter.keys(), 'fcl_1', 'b')]
    # # fcl_2_w = np.reshape(parameter[search_key(parameter.keys(), 'fcl_2', 'w')], [2048*2048])
    # fcl_2_w = parameter[search_key(parameter.keys(), 'fcl_2', 'w')]
    # fcl_2_b = parameter[search_key(parameter.keys(), 'fcl_2', 'b')]
    # # fc1_3_w = np.reshape(parameter[search_key(parameter.keys(), 'fcl_3', 'w')], [2048*2048])
    # fcl_3_w = parameter[search_key(parameter.keys(), 'fcl_3', 'w')]
    # fcl_3_b = parameter[search_key(parameter.keys(), 'fcl_3', 'b')]
    # # fc1_4_w = np.reshape(parameter[search_key(parameter.keys(), 'fcl_4', 'w')], [2048*257])
    # fcl_4_w = parameter[search_key(parameter.keys(), 'fcl_4', 'w')]
    # fcl_4_b = parameter[search_key(parameter.keys(), 'fcl_4', 'b')]

    fcl_1_w = parameter['generator_model/fcl_1/w']
    fcl_1_b = parameter['generator_model/fcl_1/b']
    fcl_2_w = parameter['generator_model/fcl_2/w']
    fcl_2_b = parameter['generator_model/fcl_2/b']
    fcl_3_w = parameter['generator_model/fcl_3/w']
    fcl_3_b = parameter['generator_model/fcl_3/b']
    fcl_4_w = parameter['generator_model/fcl_4/w']
    fcl_4_b = parameter['generator_model/fcl_4/b']
    # weight pruning
    fcl_1_w = weight_pruning(fcl_1_w, 0.6)
    fcl_2_w = weight_pruning(fcl_2_w, 0.6)
    fcl_3_w = weight_pruning(fcl_3_w, 0.6)
    fcl_4_w = weight_pruning(fcl_4_w, 0.6)
    print("model parameter pruning finish")
    # print("fcl_1_w", fcl_1_w.shape,"fcl_2_w", fcl_2_w.shape, "fcl_3_w", fc1_3_w.shape,"fcl_4_w", fc1_4_w.shape,"fcl_1_b", fcl_1_b.shape)
    ##   model parameter config
    for i in range(r):
        data_inputs = np.reshape(data_in[i], [7*257])
        fcl_1 = fully_connected(data_inputs, fcl_1_w, fcl_1_b, 1799, 2048)
        fcl_1 = relu_activate(fcl_1)
        fcl_2 = fully_connected(fcl_1, fcl_2_w, fcl_2_b, 2048, 2048)
        fcl_2 = relu_activate(fcl_2)
        fcl_3 = fully_connected(fcl_2, fcl_3_w, fcl_3_b, 2048, 2048)
        fcl_3 = relu_activate(fcl_3)
        fcl_4 = fully_connected(fcl_3, fcl_4_w, fcl_4_b, 2048, 257)
        prediction_data[i] = fcl_4
        print("frame :", i)
    # plt.imshow(prediction_data.T)
    # plt.show()
    return prediction_data


def prediction_audio(audio_dir, save_dir, scaler_path, model_dir):
    data_type = '.wav'
    n_frame = 7
    n_window = 512
    n_feature = int(n_window / 2) + 1
    n_overlap = 256
    fs = 16000
    scale = True

    # model = speech_model.model(model_para, mode='load_model')
    # model.rest_graph()
    # model.init_sess()
    # model.load_model(model_dir)

    fp = h5py.File(scaler_path, 'r')
    mean = np.array(fp.get('feature_mean'))
    var = np.array(fp.get('feature_var'))
    std = np.sqrt(var)
    fp.close()
    win_func = librosa.filters.get_window('hanning', n_window)
    file_name_list = find_file(audio_dir, data_type)
    # load parameter
    parameter = return_parameter(model_dir)

    for file_name in file_name_list:
        file_path = os.path.join(audio_dir, file_name)
        print("file path :", file_path, file_name)
        audio = read_audio(file_path, fs)

        audio_magnitude, audio_angle = calculate_audio_spectrogram(audio, n_window, n_overlap, win_func, 'magnitude')
        n_pad = int((n_frame - 1) / 2)
        audio_feature = pad_with_border(audio_magnitude, n_pad)
        audio_feature_map = generate_feature_map(audio_feature, n_frame, 1)

        audio_feature_map = audio_feature_map.reshape((audio_feature_map.shape[0], n_frame, n_feature))
        audio_feature_map = calculate_log(audio_feature_map)

        if scale:
            audio_feature_map = scale_feature(audio_feature_map, mean, std)

        audio_feature_map = audio_feature_map.reshape((audio_feature_map.shape[0], n_frame, n_feature, 1))
        print(audio_feature_map.shape)

        # load parameter and prediction
        audio_feature_map_prediction = model_prediction(audio_feature_map, parameter)

        if scale:
            audio_feature_map_prediction = audio_feature_map_prediction * std + mean
            audio_feature_map_prediction = np.exp(audio_feature_map_prediction)

        audio_feature_map_prediction_correct = audio_feature_map_prediction

        predict_audio = restore_audio(audio_feature_map_prediction_correct, audio_angle, n_window, n_overlap, win_func, 'magnitude')
        predict_audio = predict_audio[int(n_window / 2):len(predict_audio) - int(n_window / 2)]
        # if (np.max(np.abs(predict_audio)) > 1):
        #     predict_audio = predict_audio / np.max(np.abs(predict_audio))
        # filter audio
        # audio_filter = butter_bandpass_filter(audio, 100, 5000, fs, order=6)
        # add audio
        # predict_audio, _, _, _, _ = add_noise_to_signal(predict_audio, audio_filter, 10)
        # predict_audio = butter_bandpass_filter(predict_audio, 100, 5000, fs, order=2)
        save_path = os.path.join(save_dir, file_name)
        soundfile.write(save_path, predict_audio, fs)
        print('save file:', save_path)


if __name__ == '__main__':
    model_dir = 'C:/Users/asus/Desktop/siamese_model/'
    # return_parameter(model_dir)
    wave_dir = 'C:/Users/asus/Desktop/mix/eval/test/'
    save_dir = 'C:/Users/asus/Desktop/mix/save_data_0.4/'
    scale_path = 'C:/Users/asus/Desktop/scale/scale.hdf5'
    prediction_audio(wave_dir, save_dir, scale_path, model_dir)
    # statistic_weight(model_dir)

    # data = np.array([1,2,3,4])
    # w = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14],[15,16,17,18,19,20,21],[22,23,24,-25,26,27,-28]])
    # biases = np.array([9,8,7,6,5,6,7])
    # fc = fully_connected(data,w,biases,4,7)
    # print(fc)
    # print(weight_pruning(w, 0.4))

