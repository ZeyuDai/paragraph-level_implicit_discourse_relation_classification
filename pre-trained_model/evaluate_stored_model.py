import sys
import os

import torch
from torch.autograd import Variable
from model import BaseSequenceLabelingSplitImpExp,BaseSequenceLabeling,BaseSequenceLabeling_LSTMEncoder,BiLSTMCRFSplitImpExp

from sklearn import metrics
import numpy as np
import cPickle
import copy


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


def load_data(para_length_range = [1,10000]):
    print 'Loading Data...'
    outfile = open(os.path.join(os.getcwd(),'data/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding.pt'),'r')
    pdtb_data = torch.load(outfile)
    outfile.close()

    '''with open(os.path.join(os.getcwd(),'data/pdtb_implicit_moreexplicit_discourse_withoutAltLex_argpair_testdata_implicit_paralength_list.pkl'),'r') as f:
        pdtb_paralength_list = cPickle.load(f)
        f.close()
    print len(pdtb_paralength_list)
    print pdtb_paralength_list'''

    count = 0
    para_length_list = []
    test_X_eos_list = []
    test_X = []
    test_Y = []

    for (sample_X,sample_eos_list,y) in zip(pdtb_data['dev_X'][0], pdtb_data['dev_X'][2], pdtb_data['dev_Y']):
        #if torch.sum(y.data[0].clamp(0,1)) > 0: 
            #para_length = pdtb_paralength_list[count]
            #count += 1
        
            para_length = len(sample_eos_list)
            para_length_list.append(para_length)
            if para_length < para_length_range[0] or para_length > para_length_range[1]:
                continue
            test_X.append(sample_X)
            test_X_eos_list.append(sample_eos_list)
            test_Y.append(y)

    feed_data_cuda([test_X,test_Y])
    print len(para_length_list)
    print 'para length distribution: ' + str(np.unique(para_length_list, return_counts=True))
    return test_X,test_X_eos_list,test_Y

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

def evaluate(model,X,Y, discourse = 'implicit'):
    model.eval()
    X_eos_list = X[1]
    X = X[0]

    predict_Y_list = []
    target_Y_list = []

    for i in range(len(Y)):
        sample = X[i]
        sample_eos_list = X_eos_list[i]
        target = Y[i]

        predict = model(sample, sample_eos_list, target)
        #predict = model(sample, sample_eos_list)

        if discourse == 'all':
            target = target.abs()
        elif discourse == 'explicit':
            target = -target

        if use_cuda:
            predict = predict.cpu()
            target = target.cpu()

        predict = predict.data.numpy()
        target = target.data.numpy()
        
        predict,target = process_label(predict,target)

        if target is not None:
            predict = np.argmax(predict,axis = 1)
            target = np.argmax(target,axis = 1)

            predict_Y_list.append(predict)
            target_Y_list.append(target)

    predict_Y = np.concatenate(predict_Y_list,axis=0)
    target_Y = np.concatenate(target_Y_list,axis=0)
    model.train()

    return print_evaluation_result((predict_Y,target_Y))


def average_result(each_iteration_result_list):
    all_result_list = []
    explicit_result_list = []
    implicit_result_list = []

    for each_iteration_result in each_iteration_result_list:
        all_result, explicit_result, implicit_result = each_iteration_result[0],each_iteration_result[1],each_iteration_result[2]

        all_result_list.append(all_result)
        explicit_result_list.append(explicit_result)
        implicit_result_list.append(implicit_result)

    def average_result_list(result_list):
        predict_Y_list = []
        target_Y_list = []
        for result in result_list:
            predict_Y_list.append(result[0])
            target_Y_list.append(result[1])

        predict_Y = np.concatenate(predict_Y_list,axis=0)
        target_Y = np.concatenate(target_Y_list,axis=0)
        return (predict_Y,target_Y)
    
    return (average_result_list(all_result_list),average_result_list(explicit_result_list),average_result_list(implicit_result_list))


'''def average_result(each_iteration_result_list):
    implicit_result_list = []

    for each_iteration_result in each_iteration_result_list:
        implicit_result = each_iteration_result[0]
        implicit_result_list.append(implicit_result)

    def average_result_list(result_list):
        predict_Y_list = []
        target_Y_list = []
        for result in result_list:
            predict_Y_list.append(result[0])
            target_Y_list.append(result[1])
        predict_Y = np.concatenate(predict_Y_list,axis=0)
        target_Y = np.concatenate(target_Y_list,axis=0)
        return (predict_Y,target_Y)
    
    return [average_result_list(implicit_result_list)]'''

batch_size_list = [128]  # fixed 128
hidden_size_list = [300] # fixed 300>100>600
dropout_list = [5]  # 3>2>0>5
l2_reg_list = [0]   # fixed 0
nb_epoch_list = [50]
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
    test_X,test_X_eos_list,test_Y = load_data(para_length_range = [1,10000])
    word_embedding_dimension = test_X[0].size(-1)
    number_class = test_Y[0].size(-1)

    parameters = parameters_list[0]
    #stored_model_file = open(os.path.join(os.getcwd(),'result/model/pdtb_implicit_moreexplicit_discourse_withoutAltLex_argpair_multilabel_addposnerembedding_BaseSequenceLabelingLSTMEncoder_alpha0_eachiterationmodel_hidden50_inithiddenTrue_addoutputdropout.pt'),'r')
    #stored_model_file = open(os.path.join(os.getcwd(),'result/model/pdtb_implicit_moreexplicit_discourse_withoutAltLex_argpair_multilabel_addposnerembedding_BaseSequenceLabelingLSTMEncoder_NTNr=1_alpha0_eachiterationmodel_hidden300_inithiddenTrue_addoutputdropout.pt'),'r')

    #stored_model_file = open(os.path.join(os.getcwd(),'result/model/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding_BaseSequenceLabeling_eachiterationmodel_hidden300_dropout0.5_addoutputdropout_exp3.pt'),'r')
    #stored_model_file = open(os.path.join(os.getcwd(),'result/model/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding_BaseSequenceLabelingSplitImpExp_eachiterationmodel_hidden300_dropout0.5_addoutputdropout_exp2.pt'),'r')
    stored_model_file = open(os.path.join(os.getcwd(),'result/model/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding_BiLSTMCRFSplitImpExp_rand_viterbi_eachiterationmodel_hidden300_addoutputdropout_exp2.pt'),'r')

    stored_model_list = torch.load(stored_model_file)
    stored_model_list = [stored_model_list] if type(stored_model_list) != type([]) else stored_model_list
    stored_model_file.close()
    print 'Number of stored BiLSTM model seeds: ' + str(len(stored_model_list))

    overall_best_result = None
    overall_best_model = None
    overall_best_macro = -1
    each_iteration_result_list = []
    each_iteration_macro_Fscore_list = []

    for i in range(len(stored_model_list)):
        #model = BaseSequenceLabeling_LSTMEncoder(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
        #                       sentence_zero_inithidden = parameters['sentence_zero_inithidden'], cross_attention = False, attention_function = 'dot', NTN_flag = False, num_layers = parameters['num_layers'], dropout = parameters['dropout'])

        #model = BaseSequenceLabeling(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
        #                       sentence_zero_inithidden = parameters['sentence_zero_inithidden'], cross_attention = False, attention_function = 'dot', NTN_flag = False, num_layers = parameters['num_layers'], dropout = parameters['dropout'])
        
        
        #model = BaseSequenceLabelingSplitImpExp(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
        #                                                          sentence_zero_inithidden = parameters['sentence_zero_inithidden'], cross_attention = False, attention_function = 'dot', NTN_flag = False, 
        #                                                          num_layers = parameters['num_layers'], dropout = parameters['dropout'])
        
        model = BiLSTMCRFSplitImpExp(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
                          sentence_zero_inithidden = parameters['sentence_zero_inithidden'],crf_decode_method = parameters['crf_decode_method'], loss_function = parameters['loss_function'], 
                          cross_attention = False, attention_function = 'dot', NTN_flag = False, num_layers = parameters['num_layers'], dropout = parameters['dropout'])


        if use_cuda:
            model = model.cuda()
        model.load_state_dict(stored_model_list[i])

        print 'Evaluate on Explicit/Implicit discourse relation'
        print '----------------------------------------------------'
        _, tmp_all_result = evaluate(model,(test_X, test_X_eos_list), test_Y, discourse = 'all')

        print 'Evaluate on Explicit discourse relation'
        print '----------------------------------------------------'
        _ , tmp_explicit_result = evaluate(model, (test_X, test_X_eos_list), test_Y, discourse = 'explicit')

        print 'Evaluate on Implicit discourse relation'
        print '----------------------------------------------------'
        best_macro_Fscore, tmp_implicit_result = evaluate(model, (test_X, test_X_eos_list), test_Y, discourse = 'implicit')

        best_result = (tmp_all_result, tmp_explicit_result, tmp_implicit_result)
        #best_result = [tmp_implicit_result]   
        
        each_iteration_result_list.append(best_result)
        each_iteration_macro_Fscore_list.append(best_macro_Fscore)
        if best_macro_Fscore > overall_best_macro:
            overall_best_macro = best_macro_Fscore
            overall_best_result = best_result

    overall_average_result = average_result(each_iteration_result_list)
    print '-------------------------------------------------------------------------'
    print 'Overall Average Result:'
    print 'Evaluate on Explicit/Implicit discourse relation'
    print '-------------------------------------------------------------------------'
    print_evaluation_result(overall_average_result[0])

    print 'Evaluate on Explicit discourse relation'
    print '-------------------------------------------------------------------------'
    print_evaluation_result(overall_average_result[1])

    print 'Evaluate on Implicit discourse relation'
    print '-------------------------------------------------------------------------'
    print_evaluation_result(overall_average_result[2])
