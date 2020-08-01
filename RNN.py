#   RNN.py
#   a recurrent neural network to process gene expression data
import os
import shutil
import random
import argparse
import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
import json
import time
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.layers import Activation, TimeDistributed
from keras.layers import LSTM
from keras.layers import GRU
from keras.models import Sequential, load_model
from keras.utils import plot_model
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from TCdata import TCdata

### Variables
TIMESTEPS = 21
DATA_DIM = 2
VALID_PER = 0.2
DROPOUT = 0.2
DELETE = 0

### Parsing and Configuration
def parse_args():
    desc = "Keras implementation of 'Recurrent Neural Network(RNN)'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('-l', '--lstm_units', type=int, default=21, help='Number of LSTM units')

    parser.add_argument('-n', '--network', type=int, default=5, help='Network being trained/tested')

    parser.add_argument('-e', '--num_epochs', type=int, default=60, help='Number of epochs to run')

    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')

    parser.add_argument('-d', '--no_dropout', action='store_true', help='Don\'t use dropout')

    parser.add_argument('-r', '--run_opt', type=int, default=3, help='An integer: 1 to train, 2 to test, 3 for both')

    return check_args(parser.parse_args())

### Checking arguments
def check_args(args):
    # --lstm_units
    try:
        assert args.lstm_units >= 1
    except:
        print('number of LSTM units must be larger than or equal to 1')

    # --network
    try:
        assert args.network > 0 and args.network < 6
    except:
        print('number of networks must be larger than 0 and smaller than 6')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --run_opt
    try:
        assert args.run_opt == 1 or args.run_opt == 2 or args.run_opt == 3
    except:
        print('run_opt must be 1(to train) or 2(to test) or 3(for train, test)')

    return args

### FROM NN (Selene): START
def getBalancedData(input):
    pos_input = []
    all_neg_input = []
    neg_input = []
    bal_data = []

    for n in input:
        if n[2][0]==1:
            pos_input.append(n)
        else:
            all_neg_input.append(n)

    numpos = len(pos_input)
    numneg = len(all_neg_input)
    idx = random.sample(range(numneg), numpos)
    for i in idx:
        neg_input.append(all_neg_input[i])
    for n in pos_input:
        bal_data.append(n)
    for n in neg_input:
        bal_data.append(n)
    return bal_data

def getData(input):
    datar = []
    datat = []
    labels = []
    for n in input:
        datar.append(n[1][0])
        datat.append(n[1][1])
        labels.append(n[2][0])
    SS = StandardScaler()
    rscale = SS.fit_transform(datar)
    tscale = SS.fit_transform(datat)
    datar = np.array(rscale)
    datat = np.array(tscale)
    labels = np.array(labels)
    return [datar, datat, labels]

def getTrainTest(nn_input, train_keys, test_keys):
    r_train = []
    t_train = []
    train = []
    train_labels = []
    r_test = []
    t_test = []
    test = []
    test_labels = []
    for t in train_keys:
        for n in nn_input:
            if n[0]==t:
                r_train.append(n[1][0])
                t_train.append(n[1][1])
                k = [list(t) for t in zip(n[1][0], n[1][1])]
                train.append(k)
                train_labels.append(n[2][0])
    for t in test_keys:
        for n in nn_input:
            if n[0]==t:
                r_test.append(n[1][0])
                t_test.append(n[1][1])
                k = [list(t) for t in zip(n[1][0], n[1][1])]
                test.append(k)
                test_labels.append(n[2][0])
    r_train = np.array(r_train)
    t_train = np.array(t_train)
    train = np.array(train)
    train_labels = np.array(train_labels)
    r_test = np.array(r_test)
    t_test = np.array(t_test)
    test = np.array(test)
    test_labels = np.array(test_labels)
    return [train, r_train, t_train, train_labels, test, r_test, t_test, test_labels]
### FROM NN (Selene): END

### main function
def main(args):
    # parameters
    DATA_DIR = 'data/'
    f = ['blah', 'D4_100_1_timeseries.tsv', 'D4_100_2_timeseries.tsv', 'D4_100_3_timeseries.tsv', 'D4_100_4_timeseries.tsv', 'D4_100_5_timeseries.tsv']
    n = ['blah', 'D4_100_1_goldstandard.tsv', 'D4_100_2_goldstandard.tsv', 'D4_100_3_goldstandard.tsv', 'D4_100_4_goldstandard.tsv', 'D4_100_5_goldstandard.tsv']
    RESULTS_DIR = 'output/'
    MODELS_DIR = 'models/'
    IMAGES_DIR = 'imgs/'

    # Delete old results & models folder (if needed)
    if DELETE:
        try:
            shutil.rmtree(RESULTS_DIR)
            shutil.rmtree(MODELS_DIR)
            shutil.rmtree(IMAGES_DIR)
        except:
            pass

    # Create new results & models dir (if doesn't exist)
    try:
         os.mkdir(RESULTS_DIR)
         os.mkdir(MODELS_DIR)
         os.mkdir(IMAGES_DIR)
    except:
         pass

    # Set variables from args
    lstm_units = args.lstm_units
    network = args.network
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    dropout = not args.no_dropout
    run_opt = args.run_opt
    opt = 'adam' #TODO: if you want to change optimizer

    # Prepare the Dream data
    all_TC_data = TCdata(DATA_DIR+f[network], DATA_DIR+n[network], f[network][0:2])
    TC = all_TC_data.TC
    RNN_network = all_TC_data.network
    challenge = all_TC_data.challenge
    numGs = all_TC_data.numGs
    numTPs = all_TC_data.numTPs
    numPTs = all_TC_data.numPTs
    numPos = all_TC_data.numPos
    numNeg = all_TC_data.numNeg
    t = all_TC_data.t
    RNN_TCdata = all_TC_data.get2TCwLabels(2, False, [])
    balanced = getBalancedData(RNN_TCdata)

    ## Check how the data is: for debug (if required)
    ##print('TC[G1]: ' + str(TC['G1']))
    ##print('TC[G1][2]: ' + str(TC['G1'][2]))
    #print('network: ' + str(network))
    #print('challenge: ' + str(challenge))
    #print('numGs: ' + str(numGs))
    #print('numTPs: ' + str(numTPs))
    #print('numPTs: ' + str(numPTs))
    #print('numPos: ' + str(numPos))
    #print('numNeg: ' + str(numNeg))
    #print('t: ' + str(t))
    #print('TC: ' + str(TC))
    #print('RNN_TCdata: ' + str(RNN_TCdata))
    #print('balanced: ' + str(balanced))

    ### RNN related data prep for feeding into network
    kname = 'output/keys_' + f[network][0:9] + '.json'
    with open(kname, 'r') as fl:
        mykeys = json.load(fl)
    train_keys = mykeys['train']
    test_keys = mykeys['test']
    [train, r_train, t_train, train_labels, test, r_test, t_test, test_labels] = getTrainTest(RNN_TCdata, train_keys, test_keys)
    #print('train: ' + str(train))
    #print('test: ' + str(test))

    n_samples_train = len(t_train)
    n_samples_test = len(t_test)
    #print('n_samples: ' + str(n_samples_train) + ', ' + str(n_samples_test))
    ##total_batch_train = int(n_samples_train/batch_size)
    ##total_batch_test = int(n_samples_test/batch_size)
    ##print('batches: ' + str(total_batch_train) + ', ' + str(total_batch_test))

    model = Sequential()
    #print(train.shape[0]) ## Samples
    #print(train.shape[1]) ## Time steps
    #print(train.shape[2]) ## features
    model.add(LSTM(lstm_units, input_shape=(train.shape[1], train.shape[2])))
    if dropout:
        model.add(Dropout(DROPOUT))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=0.01)
    start_time = time.time()
    plot_model(model,
              to_file='imgs/rnn_model.png',
              show_shapes=True)
    exit(1)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #print("Compilation Time : ", time.time() - start_time)
    print(model.summary())

    if dropout:
        fname = 'n' + str(network) + '_' + str(batch_size) + '_' + str(num_epochs) + '_' + opt + '_dropout'
    else:
        fname = 'n' + str(network) + '_' + str(batch_size) + '_' + str(num_epochs) + '_' + opt
    if run_opt%2 == 1:
        print('Batch_size: {}, Epochs: {}, Optimizer: {}, Droput: {}'.format(batch_size, num_epochs, opt, dropout))
        hname = MODELS_DIR + 'RNN_hist_'+ fname + '.json'
        checkpointer = ModelCheckpoint(filepath=MODELS_DIR + 'RNN_model_' + fname + '-{epoch:02d}.hdf5', verbose=1, period=5)
        mname = MODELS_DIR + 'RNN_final_model_' + fname + '.hdf5'
        #train = train.reshape(len(train), train.shape[1], train.shape[2]) ##Already in this shape as (samples, time steps, features)
        history = model.fit(train, train_labels, batch_size, num_epochs, validation_data=(test, test_labels), callbacks=[checkpointer])
        with open(hname, 'w') as fl:
            json.dump(history.history, fl)
        model.save(mname)
    if run_opt >= 2:
        if run_opt == 2:
            model = load_model(MODELS_DIR + 'RNN_final_model_' + fname + '.hdf5')
        test_result = model.evaluate(test, test_labels, verbose=1)
        print('Test loss:', test_result[0])
        print('Test accuracy:', test_result[1])
        test_pred = model.predict(test, batch_size=batch_size, verbose=1)
        test_out = np.reshape(test_labels, (test_labels.shape[0], 1))
        results = np.hstack((test_out, test_pred))
        oname = RESULTS_DIR + 'RNN_results_' + fname + '.csv'
        np.savetxt(oname, results, delimiter=',')

        stats = {}
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, test_pred)
        auc_keras = auc(fpr_keras, tpr_keras)
        stats['fpr'] = fpr_keras.tolist()
        stats['tpr'] = tpr_keras.tolist()
        stats['auc'] = float(auc_keras)
        stats['thresholds'] = thresholds_keras.tolist()
        sname = RESULTS_DIR + 'RNN_stats_all_' + fname + '.json'
        with open(sname, 'w') as fl:
            json.dump(stats, fl)

        stats = {}
        stats['pred'] = test_pred.tolist()
        stats['true'] = test_labels.tolist()
        sname = RESULTS_DIR + 'RNN_stats_' + fname + '.json'
        with open(sname, 'w') as fl:
            json.dump(stats, fl)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        iname = IMAGES_DIR + 'RNN_roc_auc_' + fname + '.png'
        plt.savefig(iname)

###
if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    if (args.no_dropout):
        print('RNN.py --lstm_units ' + str(args.lstm_units) + ' --network ' + str(args.network) + ' --num_epochs ' + str(args.num_epochs) + ' --batch_size ' + str(args.batch_size) + ' --no_dropout --run_opt ' + str(args.run_opt))
    else:
        print('RNN.py --lstm_units ' + str(args.lstm_units) + ' --network ' + str(args.network) + ' --num_epochs ' + str(args.num_epochs) + ' --batch_size ' + str(args.batch_size) + ' --dropout --run_opt ' + str(args.run_opt))

    # main
    main(args)
