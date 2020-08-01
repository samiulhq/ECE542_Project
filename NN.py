#   NN.py
#   a simple neural network to perform classification on lower dimensional
#   gene expression data (output from AE.py)

from keras.layers import Input, Dense
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from keras.layers.merge import concatenate
from keras.utils import plot_model

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import json

OUTPUT_PATH = 'output/'
DATA_PATH = 'data/'


def getTC(ffull, nfull):
    from TCdata import TCdata
    data1 = TCdata(ffull, nfull, 'D4')
    TCdata = data1.get2TCwLabels(2, False, [])
    return TCdata

def getLowD(file):
    # load low dimensional data
    lowD_data = {}
    with open(lowDFile,'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            lowD_data[row[0]] = [float(row[1]), float(row[2]), float(row[3])]
    return lowD_data

def replaceTC(infile1,infile2):
    outfile = infile1
    for f in outfile:
        g1 = f[0][0]
        g2 = f[0][1]
        p = f[0][2]
        g1key = g1 + '-' + str(p)
        g2key = g2 + '-' + str(p)
        lowd = [infile2[g1key], infile2[g2key]]
        f.append(lowd)
        #f[1] = [[],[]]
        #f[1][0]=infile2[g1key]
        #f[1][1]=infile2[g2key]
    return outfile

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
    keys = []
    for n in input:
        keys.append(n[0])
        datar.append(n[3][0])
        datat.append(n[3][1])
        labels.append(n[2][0])
    SS = StandardScaler()
    rscale = SS.fit_transform(datar)
    tscale = SS.fit_transform(datat)
    datar = np.array(rscale)
    datat = np.array(tscale)
    labels = np.array(labels)
    return [datar, datat, labels, keys]

def getTrain(datar, datat, labels):
    r_train = []
    t_train = []
    train_labels = []
    train_keys = []
    numrecs = datar.shape[0]
    numtrain = int(numrecs*.80)
    idxtrain = random.sample(range(numrecs), numtrain)
    for i in idxtrain:
        r_train.append(datar[i])
        t_train.append(datat[i])
        train_keys.append(keys[i])
        train_labels.append(labels[i])
    r_train = np.array(r_train)
    t_train = np.array(t_train)
    train_labels = np.array(train_labels)
    return [r_train, t_train, train_labels, train_keys, idxtrain]

def getTest(datar, datat, labels, idxtrain):
    r_test = []
    t_test = []
    test_keys = []
    test_labels = []
    numrecs = datar.shape[0]
    idxtest = []
    # generate idxtest
    for i in range(numrecs):
        if i not in idxtrain:
            idxtest.append(i)
            r_test.append(datar[i])
            t_test.append(datat[i])
            test_keys.append(keys[i])
            test_labels.append(labels[i])

    r_test = np.array(r_test)
    t_test = np.array(t_test)
    test_labels = np.array(test_labels)
    return [r_test, t_test, test_labels, test_keys, idxtest]

def getTrainTest(nn_input, train_keys, test_keys):
    r_train = []
    t_train = []
    train_labels = []
    r_test = []
    t_test = []
    test_labels = []
    for t in train_keys:
        for n in nn_input:
            if n[0]==t:
                r_train.append(n[3][0])
                t_train.append(n[3][1])
                train_labels.append(n[2][0])
    for t in test_keys:
        for n in nn_input:
            if n[0]==t:
                r_test.append(n[3][0])
                t_test.append(n[3][1])
                test_labels.append(n[2][0])
    r_train = np.array(r_train)
    t_train = np.array(t_train)
    train_labels = np.array(train_labels)
    r_test = np.array(r_test)
    t_test = np.array(t_test)
    test_labels = np.array(test_labels)
    return [r_train, t_train, train_labels, r_test, t_test, test_labels]

def kerasModel(datar, datat,optimizer='rmsprop'):
    # Regulator Channel
    inputs1 = Input(shape=(datar.shape[1],))
    denser = Dense(256, activation='relu')(inputs1)
    denser = Dense(128, activation='relu')(denser)
    denser = Dense(64, activation='relu')(denser)
    denser = Dense(32, activation='relu')(denser)
    flatr = Dense(32, activation='relu')(denser)

    # Target Channel
    inputs2 = Input(shape = (datat.shape[1],))
    denset = Dense(256, activation='relu')(inputs2)
    denset = Dense(128, activation='relu')(denset)
    denset = Dense(64, activation='relu')(denset)
    denset = Dense(32, activation='relu')(denset)
    flatt = Dense(32, activation='relu')(denset)

    # Merge Channels
    merged = concatenate([flatr, flatt])

    # Interpretation
    dense1 = Dense(200, activation='relu')(merged)
    dense2 = Dense(20, activation='relu')(dense1)
    outputs = Dense(1, activation='sigmoid')(dense2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    return model

if __name__ == "__main__":
    finalRun = True
    genModelOnly = False
    genSamFile = True
    haveKeys = True

    f = 'D4_100_5_timeseries.tsv'
    n = 'D4_100_5_goldstandard.tsv'
    ffull = DATA_PATH + f
    nfull = DATA_PATH + n
    TCdata = getTC(ffull,nfull)

    lowDFile = OUTPUT_PATH + 'D4_100_5_timeseries_3D_save1.tsv'
    lowDdata = getLowD(lowDFile)

    nn_input = replaceTC(TCdata,lowDdata)
    balanced = getBalancedData(nn_input)

    if genSamFile:
        lname = lowDFile[:-4] + '_GenePairs.csv'
        with open(lname, 'w') as f:
            row = 'R,T,P,Rfull'
            for x in range(21):
                row = row + ','
            row = row + 'Tfull'
            for x in range(21):
                row = row + ','
            row = row + 'Label,Rlow,,,Tlow\n'
            f.write(row)
            for n in balanced:
                row = n[0][0] + ',' + n[0][1] + ',' + str(n[0][2]) + ','
                for x in range(21):
                    row = row + str(n[1][0][x]) + ','
                for x in range(21):
                    row = row + str(n[1][1][x]) + ','
                row = row + str(n[2][0]) + ','
                for x in range(3):
                    row = row + str(n[3][0][x]) + ','
                for x in range(2):
                    row = row + str(n[3][1][x]) + ','
                row = row + str(n[3][1][2]) + '\n'
                #print(row)
                f.write(row)
    exit(1)

    if haveKeys:
        kname = 'output/keys_' + f[0:9] + '.json'
        with open(kname, 'r') as f:
            mykeys = json.load(f)
        train_keys = mykeys['train']
        test_keys = mykeys['test']
        [r_train, t_train, train_labels, r_test, t_test, test_labels] = getTrainTest(nn_input, train_keys, test_keys)
    else:
        [datar, datat, labels, keys] = getData(balanced)
        [r_train, t_train, train_labels, train_keys, idxtrain] = getTrain(datar, datat, labels)
        [r_test, t_test, test_labels, test_keys, idxtest] = getTest(datar,
                                                datat,
                                                labels,
                                                idxtrain)
        finalkeys = {}
        finalkeys['train'] = train_keys
        finalkeys['test'] = test_keys
        kname = 'output/keys_' + f[0:9] + '.json'
        with open(kname, 'w') as f:
            json.dump(finalkeys, f)

    if finalRun:
        # set to best network (highest accuracy)
        batch_size = [100]
        epochs = [75]
        optimizer = ['adam']
    else:
        batch_size = [5, 10, 50, 100, 250]
        epochs = [5, 10, 50, 75]
        optimizer = ['rmsprop', 'adam', 'sgd']

    if genModelOnly:
        model = kerasModel(r_train, t_train, optimizer[0])
        plot_model(model,
              to_file='imgs/NN_model.png',
              show_shapes=True)
    else:
        for b in batch_size:
            for e in epochs:
                for o in optimizer:
                    print('B: {}, E: {}, O: {}'.format(b,e,o))
                    model = kerasModel(r_train, t_train,o)
                    # model.summary()
                    history = model.fit([r_train, t_train],
                             train_labels,
                             batch_size=b,
                             epochs=e,
                             validation_split=0.2)
                    print('Max Accuracy: {}'.format(max(history.history['acc'])))
                    hname = 'output/hist_' + str(b) + '_' + str(e) + '_' + o + '.json'
                    with open(hname, 'w') as f:
                        json.dump(history.history, f)
                    mname = 'models/model_' + str(b) + '_' + str(e) + '_' + o + '.h5'
                    model.save(mname)
                    test_pred = model.predict([r_test, t_test],
                                              batch_size=b,
                                              verbose=1)
                    test_true = np.reshape(test_labels,(test_labels.shape[0],1))
                    results = {}
                    results['true'] = test_true.tolist()
                    results['pred'] = test_pred.tolist()
                    rname = 'output/results_' + str(b) + '_' + str(e) + '_' + o +  '_' + ffull[12] + '.json'
                    print(rname)
                    with open(rname, 'w') as f:
                        json.dump(results, f)
