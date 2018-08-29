import sys
import os
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import BaseSequenceLabelingSplitImpExp

from sklearn import metrics
import numpy as np
import cPickle


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

def apply_weighted_class(Y):
    print 'Weighting class...'

    weights = Variable(torch.zeros((Y[0].size(-1))), requires_grad = False)
    for y in Y:
        weights = weights + torch.sum(y.abs(),0)

    weights = (1 / weights)

    for y in Y:
        for j in range(y.size(0)):
            y[j,:] = y[j,:] * weights

def split_y(X,X_eos_list,X_label_length_list,Y):
    new_y = []
    new_x = []
    new_x_eos_list =[]
    new_x_label_length_list = []

    for i in range(Y[0].size(-1)):
        new_y.append([])
        new_x.append([])
        new_x_eos_list.append([])
        new_x_label_length_list.append([])

    positive_sample = [0] * len(new_y)
    negative_sample = [0] * len(new_y)
    for j in range(len(Y)):
        y = Y[j]
        x = X[j]
        x_eos_list = X_eos_list[j]
        x_label_length_list = X_label_length_list[j]

        for i in range(Y[0].size(-1)):
            new_yi = torch.zeros(y.size(0),2)

            for k in range(y.size(0)):
                if y[k,i].data[0] != 0:
                    new_yi[k,0] = y[k,i].data[0]
                elif torch.sum(y[k,:]).data[0] > 0:
                    new_yi[k,1] = 1
                elif torch.sum(y[k,:]).data[0] < 0:
                    new_yi[k,1] = -1

            new_yi = Variable(new_yi,requires_grad = False)

            if torch.sum(new_yi[:,0].abs()).data[0] > 0:
                positive_sample[i] += torch.sum(new_yi[:,0].abs()).data[0]
                #negative_sample[i] += torch.sum(new_yi[:,1].abs()).data[0]
                new_y[i].append(new_yi)
                new_x[i].append(x)
                new_x_eos_list[i].append(x_eos_list)
                new_x_label_length_list[i].append(x_label_length_list)
            elif negative_sample[i] <= positive_sample[i]*0.5:
                negative_sample[i] += torch.sum(new_yi[:,1].abs()).data[0]
                new_y[i].append(new_yi)
                new_x[i].append(x)
                new_x_eos_list[i].append(x_eos_list)
                new_x_label_length_list[i].append(x_label_length_list)

    return new_x,new_x_eos_list,new_x_label_length_list,new_y

def load_data(weighted_class = False):
    print 'Loading Data...'
    outfile = open(os.path.join(os.getcwd(),'data/pdtb_implicit_moreexplicit_discourse_withoutAltLex_paragraph_multilabel_addposnerembedding.pt'),'r')
    pdtb_data = torch.load(outfile)
    outfile.close()

    dev_X,dev_Y,train_X,train_Y,test_X,test_Y = pdtb_data['dev_X'],pdtb_data['dev_Y'],pdtb_data['train_X'] ,pdtb_data['train_Y'],pdtb_data['test_X'],pdtb_data['test_Y']

    dev_X_eos_list = dev_X[2]
    dev_X_label_length_list = dev_X[1]
    dev_X = dev_X[0]
    dev_X,dev_X_eos_list,dev_X_label_length_list,dev_Y = split_y(dev_X,dev_X_eos_list,dev_X_label_length_list,dev_Y)

    train_X_eos_list = train_X[2]
    train_X_label_length_list = train_X[1]
    train_X = train_X[0]
    train_X,train_X_eos_list,train_X_label_length_list,train_Y = split_y(train_X,train_X_eos_list,train_X_label_length_list,train_Y)

    test_X_eos_list = test_X[2]
    test_X_label_length_list = test_X[1]
    test_X = test_X[0]
    test_X,test_X_eos_list,test_X_label_length_list,test_Y = split_y(test_X,test_X_eos_list,test_X_label_length_list,test_Y)

    if weighted_class:
        apply_weighted_class(train_Y)
        apply_weighted_class(dev_Y)
        apply_weighted_class(test_Y)

    feed_data_cuda(train_X+train_Y+test_X+test_Y)

    return dev_X,dev_X_label_length_list,dev_X_eos_list,dev_Y,train_X,train_X_label_length_list,train_X_eos_list,train_Y,test_X,test_X_label_length_list,test_X_eos_list,test_Y

#epsilon = 1.0e-6
def calculate_loss(predict,target,criterion,alpha = 1):
    #predict = torch.clamp(predict, min=np.log(epsilon), max=np.log(1-epsilon))
    if alpha == 1:
        return -torch.dot(predict.view(-1),target.abs().view(-1))
    elif alpha == 0:
        return -torch.dot(predict.view(-1),target.clamp(0,1).view(-1))
    elif alpha >= 100:
        return -torch.dot(predict.view(-1),target.clamp(-1,0).abs().view(-1))
    else:
        return -torch.dot(predict.view(-1),target.clamp(0,1).view(-1))- alpha*torch.dot(predict.view(-1),target.clamp(-1,0).abs().view(-1))

def train(model,X,Y,optimizer,criterion,alpha = 1):
    optimizer.zero_grad()
    loss = 0

    X_eos_list = X[1]
    X = X[0]

    for i in range(len(Y)):
        sample = X[i]
        sample_eos_list = X_eos_list[i]
        target = Y[i]

        output = model(sample, sample_eos_list, target)
        loss += calculate_loss(output,target,criterion, alpha = alpha)

    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    total_norm = nn.utils.clip_grad_norm(model.parameters(), 5.0)

    optimizer.step()
    return loss.data[0]

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

        list_1.append(real_label)
        list_2.append(predict_label)

    if len(list_1) > 0:
        real_Y = np.stack(list_1)
        predict_Y = np.stack(list_2)

        return predict_Y,real_Y
    else:
        return None,None

def print_evaluation_result(result):
    predict_Y,target_Y,loss = result[0],result[1],result[2]

    print 'Confusion Metric'
    print metrics.confusion_matrix(target_Y,predict_Y)
    print 'Accuracy'
    print metrics.accuracy_score(target_Y, predict_Y)
    print 'loss'
    print loss
    print 'Binary Precision/Recall/F-score'
    print metrics.precision_recall_fscore_support(target_Y, predict_Y, pos_label = 0, average='binary')
    return metrics.precision_recall_fscore_support(target_Y, predict_Y, pos_label = 0, average='binary')[2], result

def evaluate(model,X,Y, discourse = 'implicit'):
    model.eval()
    X_eos_list = X[1]
    X = X[0]

    predict_Y_list = []
    target_Y_list = []
    loss = 0

    for i in range(len(Y)):
        sample = X[i]
        sample_eos_list = X_eos_list[i]
        target = Y[i]

        predict = model(sample, sample_eos_list, target)

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
            loss += -np.sum(predict*target)

            predict = np.argmax(predict,axis = 1)
            target = np.argmax(target,axis = 1)

            predict_Y_list.append(predict)
            target_Y_list.append(target)

    predict_Y = np.concatenate(predict_Y_list,axis=0)
    target_Y = np.concatenate(target_Y_list,axis=0)
    model.train()

    return print_evaluation_result((predict_Y,target_Y,loss))

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
        loss = 0
        for result in result_list:
            predict_Y_list.append(result[0])
            target_Y_list.append(result[1])
            loss += result[2]
        predict_Y = np.concatenate(predict_Y_list,axis=0)
        target_Y = np.concatenate(target_Y_list,axis=0)
        return (predict_Y,target_Y,loss)
    
    return (average_result_list(all_result_list),average_result_list(explicit_result_list),average_result_list(implicit_result_list))

def trainEpochs(model, X, Y, valid_X, valid_Y, batch_size, n_epochs, print_every=1, evaluate_every = 1, optimizer_type = 'adam', weight_decay = 0, alpha = 1, use_scheduler = False):
    if optimizer_type == 'adadelta': 
        optimizer = optim.Adadelta(model.parameters(), lr = 0.5, weight_decay = weight_decay)  #baseseq2seq lr = 0.5
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr = 0.005, weight_decay = weight_decay)  #baseseq2seq lr = 0.005
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(),lr= 0.0005, weight_decay = weight_decay) #baseseq2seq lr= 0.0002
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9, weight_decay = weight_decay)
    else:
        print "optimizer not recommend for the task!"
        sys.exit()

    if use_scheduler:
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience = 10)

    criterion = nn.NLLLoss()

    X_label_length_list = X[1]
    X_eos_list = X[2]
    X = X[0]

    start = time.time()
    random_list = range(len(Y))
    print_loss_total = 0  # Reset every print_every
    best_macro_Fscore = -1
    best_result = None
    print '----------------------------------------------------'
    print 'Training start: ' + '#training_samples = ' + str(len(Y))
    for epoch in range(1, n_epochs + 1):
        print 'epoch ' + str(epoch) + '/' + str(n_epochs)
        random.shuffle(random_list)

        i = 0
        target_length = 0
        batch = [] 
        while i < len(random_list):
            batch.append(random_list[i])
            target_length += X_label_length_list[random_list[i]]
            i = i + 1

            if target_length >= batch_size or i >= len(random_list):
                batch_X_eos_list = []
                batch_X = []
                batch_Y = []

                for index in batch:
                    batch_X.append(X[index])
                    batch_X_eos_list.append(X_eos_list[index])
                    batch_Y.append(Y[index])

                loss = train(model, (batch_X, batch_X_eos_list), batch_Y, optimizer, criterion, alpha = alpha)
                print_loss_total += loss

                target_length = 0
                batch = []

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch*1.0/ n_epochs), epoch, epoch*1.0 / n_epochs * 100, print_loss_avg))

        if epoch % evaluate_every == 0:
            print '----------------------------------------------------'
            print 'Step Evaluation: #valid_samples= ' + str(len(valid_Y))
            print 'Evaluate on Explicit/Implicit discourse relation'
            print '----------------------------------------------------'
            _, tmp_all_result = evaluate(model,valid_X,valid_Y, discourse = 'all')

            print 'Evaluate on Explicit discourse relation'
            print '----------------------------------------------------'
            _, tmp_explicit_result = evaluate(model,valid_X,valid_Y, discourse = 'explicit')

            print 'Evaluate on Implicit discourse relation'
            print '----------------------------------------------------'
            tmp_macro_Fscore, tmp_implicit_result = evaluate(model,valid_X,valid_Y, discourse = 'implicit')
            if tmp_macro_Fscore > best_macro_Fscore:
                best_macro_Fscore = tmp_macro_Fscore
                best_result = (tmp_all_result, tmp_explicit_result, tmp_implicit_result)

            if use_scheduler:
                scheduler.step(tmp_macro_Fscore)

    print 'Training completed!'
    return best_macro_Fscore,best_result


batch_size_list = [128]  # fixed 128 > 256
hidden_size_list = [300] # fixed 300>600>100
dropout_list = [5]  # 3>2>0>5
l2_reg_list = [0]   # fixed 0
nb_epoch_list = [50]
encoder_sentence_embedding_type_list = ['max'] # max > last
sentence_zero_inithidden_list = [False]
#decoder_type_list = ['tensor']
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
                            #for decoder_type in decoder_type_list:
                                for weight_decay in l2_reg_list:
                                    for dropout in dropout_list:
                                        parameters = {}
                                        parameters['nb_epoch'] = nb_epoch
                                        parameters['sentence_embedding_type'] = sentence_embedding_type
                                        parameters['sentence_zero_inithidden']= sentence_zero_inithidden
                                        parameters['num_layers'] = num_layers
                                        parameters['batch_size'] = batch_size
                                        parameters['hidden_size'] = hidden_size
                                        #parameters['decoder_type'] = decoder_type
                                        parameters['optimizer_type'] = optimizer_type
                                        parameters['dropout'] = dropout * 0.1
                                        parameters['weight_decay'] = weight_decay
                                        parameters_list.append(parameters)


if __name__ == "__main__":
    _,_,_,_,train_X,train_X_label_length_list,train_X_eos_list,train_Y,test_X,_,test_X_eos_list,test_Y = load_data(weighted_class= False)
    word_embedding_dimension = test_X[0][0].size(-1)
    number_class = 2

    discourse_sense_list = ['Temporal','Comparison','Contingency','Expansion']
    for i in range(0,4):
        print '-----------------------------------------------------------------------------'
        print 'Binary classifier for label: ' + discourse_sense_list[i]
        print '-----------------------------------------------------------------------------'
        train_Yi = train_Y[i]
        train_Xi = train_X[i]
        train_Xi_eos_list = train_X_eos_list[i]
        train_Xi_label_length_list = train_X_label_length_list[i]
        
        test_Yi = test_Y[i]
        test_Xi = test_X[i]
        test_Xi_eos_list = test_X_eos_list[i]

        '''dev_Yi = dev_Y[i]
        dev_Xi = dev_X[i]
        dev_Xi_eos_list = dev_X_eos_list[i]'''

        macro_F1_list = []
        for parameters in parameters_list:
            overall_best_result = None
            overall_best_macro = -1
            each_iteration_result_list = []
            each_iteration_macro_Fscore_list = []

            for iteration in range(10):
                model = BaseSequenceLabelingSplitImpExp(word_embedding_dimension, number_class, hidden_size=parameters['hidden_size'], sentence_embedding_type = parameters['sentence_embedding_type'], 
                        sentence_zero_inithidden = parameters['sentence_zero_inithidden'], cross_attention = False, attention_function = 'dot', NTN_flag = False, num_layers = parameters['num_layers'], dropout = parameters['dropout'])
                
                if use_cuda:
                    model = model.cuda()

                best_macro_Fscore, best_result = trainEpochs(model, (train_Xi,train_Xi_label_length_list,train_Xi_eos_list), train_Yi, (test_Xi, test_Xi_eos_list), test_Yi, 
                                                            batch_size = parameters['batch_size'], n_epochs = parameters['nb_epoch'], optimizer_type = parameters['optimizer_type'], 
                                                            weight_decay = parameters['weight_decay'], alpha = 0, use_scheduler = False)
                
                print '----------------------------------------------------'
                print 'Experiment Iteration ' +  str(iteration+1) + ' Evaluation: #test_samples= ' + str(len(test_Y))
                print 'Evaluate on Explicit/Implicit discourse relation'
                print '----------------------------------------------------'
                print_evaluation_result(best_result[0])

                print 'Evaluate on Explicit discourse relation'
                print '----------------------------------------------------'
                print_evaluation_result(best_result[1])

                print 'Evaluate on Implicit discourse relation'
                print '----------------------------------------------------'
                print_evaluation_result(best_result[2])

                each_iteration_result_list.append(best_result)
                each_iteration_macro_Fscore_list.append(best_macro_Fscore)
                if best_macro_Fscore > overall_best_macro:
                    overall_best_macro = best_macro_Fscore
                    overall_best_result = best_result

            print '--------------------------------------------------------------------------'
            print 'Overall Best Result:'
            print 'Evaluate on Explicit/Implicit discourse relation'
            print '-------------------------------------------------------------------------'
            print_evaluation_result(overall_best_result[0])

            print 'Evaluate on Explicit discourse relation'
            print '-------------------------------------------------------------------------'
            print_evaluation_result(overall_best_result[1])

            print 'Evaluate on Implicit discourse relation'
            print '-------------------------------------------------------------------------'
            print_evaluation_result(overall_best_result[2])


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
            macro_F1_list.append(print_evaluation_result(overall_average_result[2])[0])

            print 'Implicit discourse relation classification Binary F_score std:' + str(np.std(each_iteration_macro_Fscore_list))
            print str(parameters)
            sys.stdout.flush()
        
        print '-----------------------------------------------------------------------------'
        print 'Overall Binary Evaluation for label: ' + discourse_sense_list[i]
        for i in range(len(parameters_list)):
            print str(parameters_list[i]) + '       Binary F-1:  ' + str(macro_F1_list[i]) + '\n'
        print '-----------------------------------------------------------------------------'
