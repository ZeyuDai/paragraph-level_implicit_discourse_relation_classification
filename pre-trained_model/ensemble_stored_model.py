import sys
import os
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import BaseSequenceLabeling,BaseSequenceLabelingSplitImpExp,BiLSTMCRFSplitImpExp

from sklearn import metrics
import numpy as np
import cPickle
import copy

######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


use_cuda = torch.cuda.is_available()
if use_cuda:
    print ("Using GPU!")
else:
    print ("Using CPU!")

def feed_data_cuda(data):
    if use_cuda:
        for X in data:
            for i in range(len(X)):
                X[i] = X[i].cuda()


def load_data(weighted_class = False):
    print 'Loading Data...'
    outfile = open(os.path.join(os.getcwd(),'data/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding.pt'),'r')
    pdtb_data = torch.load(outfile)
    outfile.close()

    dev_X,dev_Y,train_X,train_Y,test_X,test_Y = pdtb_data['dev_X'],pdtb_data['dev_Y'],pdtb_data['train_X'] ,pdtb_data['train_Y'],pdtb_data['test_X'],pdtb_data['test_Y']

    dev_X_eos_list = dev_X[2]
    dev_X_label_length_list = dev_X[1]
    dev_X = dev_X[0]

    train_X_eos_list = train_X[2]
    train_X_label_length_list = train_X[1]
    train_X = train_X[0]

    test_X_eos_list = test_X[2]
    test_X_label_length_list = test_X[1]
    test_X = test_X[0]

    if weighted_class:
        apply_weighted_class(train_Y)
        apply_weighted_class(dev_Y)
        apply_weighted_class(test_Y)

    feed_data_cuda([test_X,test_Y])

    return dev_X,dev_X_label_length_list,dev_X_eos_list,dev_Y,train_X,train_X_label_length_list,train_X_eos_list,train_Y,test_X,test_X_label_length_list,test_X_eos_list,test_Y

def process_label(predict_Y,target_Y):
    assert predict_Y.shape == target_Y.shape
    list_1 = []
    list_2 = []

    for i in range(target_Y.shape[0]):
        real_label = target_Y[i,:]
        predict_label = predict_Y[i,:]

        #handle non-label case
        if np.sum(real_label) <= 0:
            continue

        #handle multilabel case
        if np.sum(real_label) >= 2:
            # predict one of the correct label
            if real_label[np.argmax(predict_label)] > 0:
                real_label = np.zeros(target_Y.shape[1])
                real_label[np.argmax(predict_label)] = 1

        list_1.append(real_label)
        list_2.append(predict_label)

    if len(list_1) > 0:
        real_Y = np.stack(list_1)
        predict_Y = np.stack(list_2)

        return predict_Y,real_Y
    else:
        return None,None

def print_evaluation_result(result):
    predict_Y,target_Y = result[0],result[1]

    print 'Confusion Metric'
    print metrics.confusion_matrix(target_Y,predict_Y)
    print 'Accuracy'
    print metrics.accuracy_score(target_Y, predict_Y)
    print 'Micro Precision/Recall/F-score'
    print metrics.precision_recall_fscore_support(target_Y, predict_Y, average='micro') 
    print 'Macro Precision/Recall/F-score'
    print metrics.precision_recall_fscore_support(target_Y, predict_Y, average='macro') 
    print 'Each-class Precision/Recall/F-score'
    print metrics.precision_recall_fscore_support(target_Y, predict_Y, average=None) 

    return metrics.precision_recall_fscore_support(target_Y, predict_Y, average='macro')[2], result

def evaluate(model_list,X,Y, discourse = 'implicit'):
    X_eos_list = X[1]
    X = X[0]

    predict_Y_list = []
    target_Y_list = []

    for model in model_list:
        model.eval()

    for i in range(len(Y)):
        sample = X[i]
        sample_eos_list = X_eos_list[i]
        target = Y[i]
        sample_predict_list = []

        for model in model_list:
            sample_preidct = model(sample, sample_eos_list, target)
            #sample_preidct = model(sample, sample_eos_list)

            if use_cuda:
                sample_preidct = sample_preidct.cpu()

            sample_preidct = sample_preidct.data.numpy()
            sample_predict_list.append(sample_preidct)
           
        predict = np.zeros(sample_preidct.shape)
        for sample_preidct in sample_predict_list:
            tmp_sample_preidct = np.zeros(sample_preidct.shape)

            for i in range(sample_preidct.shape[0]):
                index = np.argmax(sample_preidct[i,:])
                tmp_sample_preidct[i,index] = 1

            predict = predict + tmp_sample_preidct

        if discourse == 'all':
            target = target.abs()
        elif discourse == 'explicit':
            target = -target

        if use_cuda:
            target = target.cpu()
        target = target.data.numpy()

        predict,target = process_label(predict,target)

        if target is not None:
            predict = np.argmax(predict,axis = 1)
            target = np.argmax(target,axis = 1)

            predict_Y_list.append(predict)
            target_Y_list.append(target)


    predict_Y = np.concatenate(predict_Y_list,axis=0)
    target_Y = np.concatenate(target_Y_list,axis=0)

    return print_evaluation_result((predict_Y,target_Y))



batch_size_list = [128]  # fixed 128
hidden_size_list = [300] # fixed 300>100>600
dropout_list = [5]  # 3>2>0>5
l2_reg_list = [0]   # fixed 0
nb_epoch_list = [40]
encoder_sentence_embedding_type_list = ['max'] # max > last
sentence_zero_inithidden_list = [False]
decoder_type_list = ['concat_prev']
crf_decode_method_list = ['viterbi'] #'marginal' < 'viterbi'
loss_function_list = ['likelihood'] # 'likelihood','labelwise'
optimizer_type_list = ['adam']  # adam > adagrad
num_layers_list = [1] # 1 > 2

parameters_list = []
for num_layers in num_layers_list:
    for sentence_embedding_type in encoder_sentence_embedding_type_list:
        for sentence_zero_inithidden in sentence_zero_inithidden_list:
            for batch_size in batch_size_list:
                for optimizer_type in optimizer_type_list:
                    for hidden_size in hidden_size_list:
                        for nb_epoch in nb_epoch_list:
                            for decoder_type in decoder_type_list:
                                for loss_function in loss_function_list:
                                    for crf_decode_method in crf_decode_method_list:
                                        for weight_decay in l2_reg_list:
                                            for dropout in dropout_list:
                                                parameters = {}
                                                parameters['nb_epoch'] = nb_epoch
                                                parameters['sentence_embedding_type'] = sentence_embedding_type
                                                parameters['sentence_zero_inithidden']= sentence_zero_inithidden
                                                parameters['num_layers'] = num_layers
                                                parameters['batch_size'] = batch_size
                                                parameters['hidden_size'] = hidden_size
                                                parameters['crf_decode_method'] = crf_decode_method
                                                parameters['loss_function'] = loss_function
                                                parameters['decoder_type'] = decoder_type
                                                parameters['optimizer_type'] = optimizer_type
                                                parameters['dropout'] = dropout * 0.1
                                                parameters['weight_decay'] = weight_decay
                                                parameters_list.append(parameters)


if __name__ == "__main__":
    _,_,_,_,_,_,_,_,test_X,_,test_X_eos_list,test_Y = load_data()
    word_embedding_dimension = test_X[0].size(-1)
    number_class = test_Y[0].size(-1)

    parameters = parameters_list[0]
    #stored_bilstm_model_file = open(os.path.join(os.getcwd(),'result/model/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding_BaseSequenceLabeling_alpha0_eachiterationmodel_hidden300_dropout0.5_addoutputdropout.pt'),'r')
    #stored_bilstm_model_file = open(os.path.join(os.getcwd(),'result/model/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding_BaseSequenceLabeling_eachiterationmodel_hidden300_dropout0.5_addoutputdropout_exp3.pt'),'r')
    #stored_bilstm_model_file = open(os.path.join(os.getcwd(),'result/model/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding_BaseSequenceLabelingSplitImpExp_eachiterationmodel_hidden300_dropout0.5_addoutputdropout_exp2.pt'),'r')
    stored_bilstm_model_file = open(os.path.join(os.getcwd(),'result/model/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding_BiLSTMCRFSplitImpExp_rand_viterbi_eachiterationmodel_hidden300_addoutputdropout_exp2.pt'),'r')

    stored_bilstm_model_list = torch.load(stored_bilstm_model_file)
    stored_bilstm_model_list = [stored_bilstm_model_list] if type(stored_bilstm_model_list) != type([]) else stored_bilstm_model_list
    stored_bilstm_model_file.close()
    print 'Number of stored BiLSTM model seeds: ' + str(len(stored_bilstm_model_list))

    model_list = []
    for i in range(len(stored_bilstm_model_list)):
        #model = BaseSequenceLabeling(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
        #                                        sentence_zero_inithidden = parameters['sentence_zero_inithidden'], cross_attention = False, attention_function = 'dot', NTN_flag = False, 
        #                                        num_layers = parameters['num_layers'], dropout = parameters['dropout'])

        #model = BaseSequenceLabelingSplitImpExp(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
        #                                        sentence_zero_inithidden = parameters['sentence_zero_inithidden'], cross_attention = False, attention_function = 'dot', NTN_flag = False, 
        #                                        num_layers = parameters['num_layers'], dropout = parameters['dropout'])

        model = BiLSTMCRFSplitImpExp(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
                              sentence_zero_inithidden = parameters['sentence_zero_inithidden'],crf_decode_method = parameters['crf_decode_method'], loss_function = parameters['loss_function'], 
                              cross_attention = False, attention_function = 'dot', NTN_flag = False, num_layers = parameters['num_layers'], dropout = parameters['dropout'])

        if use_cuda:
            model = model.cuda()
        model.load_state_dict(stored_bilstm_model_list[i])

        model_list.append(model)

    print 'Ensemble Evaluate on Explicit/Implicit discourse relation'
    print '----------------------------------------------------'
    _, tmp_all_result = evaluate(model_list,(test_X, test_X_eos_list), test_Y, discourse = 'all')

    print 'Ensemble Evaluate on Explicit discourse relation'
    print '----------------------------------------------------'
    _ , tmp_explicit_result = evaluate(model_list, (test_X, test_X_eos_list), test_Y, discourse = 'explicit')

    print 'Ensemble Evaluate on Implicit discourse relation'
    print '----------------------------------------------------'
    best_macro_Fscore, tmp_implicit_result = evaluate(model_list, (test_X, test_X_eos_list), test_Y, discourse = 'implicit')


