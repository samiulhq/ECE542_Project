# Implementation of LeNet-5 in keras
# [LeCun et al., 1998. Gradient based learning applied to document recognition]
# Some minor changes are made to the architecture like using ReLU activation instead of
# sigmoid/tanh, max pooling instead of avg pooling and softmax output layer
# code from https://github.com/TaavishThaman/LeNet-5-with-Keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import plot_model
import numpy as np
import json
from TCdata import TCdata
from sklearn.metrics import confusion_matrix

def getTrainTest(nn_input, train_keys, test_keys):
    x_train = []
    x_test = []
    y_train = []
    y_test=[]
    test_labels = []
    for t in train_keys:
        for n in nn_input:
            if n[0]==t:
                x_train.append(n[1])
                y_train.append(n[2])

    for t in test_keys:
        for n in nn_input:
            if n[0]==t:
                x_test.append(n[1])
                y_test.append(n[2])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return [x_train, y_train, x_test, y_test]


np.random.seed(12345)
DATA_PATH = 'data/'
f = 'D4_100_5_timeseries.tsv'
n = 'D4_100_5_goldstandard.tsv'


kname = 'output/keys_' + f[0:9] + '.json'

with open(kname, 'r') as tf:
        mykeys = json.load(tf)
        train_keys = mykeys['train']
        test_keys = mykeys['test']

data1 = TCdata(DATA_PATH + f,DATA_PATH + n,'D4')
data2 = data1.get2TCwLabels(2, False, [])


y=[l[2] for l in data2]
x=[l[1] for l in data2]

y=np.asanyarray(y)
x=np.asanyarray(x)

#split positive and negative example
ind_positive=np.where(y==1)[0]
ind_negative=np.where(y==0)[0]
y_positive=y[ind_positive]
x_positive=x[ind_positive]

y_negative=y[ind_negative]
x_negative=x[ind_negative]


#create balanced dataset equal positive and negative example
num_pos_case=np.alen(y_positive)
num_neg_case=np.alen(y_negative)
ind_tmp=np.arange(num_neg_case)
np.random.shuffle(ind_tmp)
ind =ind_tmp[0:num_pos_case]
y_final=np.concatenate([y_positive,y_negative[ind]])
x_final=np.concatenate([x_positive,x_negative[ind]])
total_ind=np.arange(np.alen(y_final))
np.random.shuffle(total_ind)
y_final=y_final[total_ind]
x_final=x_final[total_ind]



[x_train, y_train, x_test, y_test] = getTrainTest(data2, train_keys, test_keys)


#generate train and test split
#x_train, x_test, y_train, y_test = train_test_split( x_final, y_final, test_size=0.33, random_state=42)


#(x_train, y_train), (x_test, y_test) = mnist.load_data()

### NEED TO ADD CODE HERE THAT LOADS MNIST DATA  AND LABELS FROM DATASETS
## NORMALIZE BY 255
## RESHAPE TO 28 BY 28 IMG

#Visualizing the data
sample = np.array(x_train[0:10])
for s in sample:
    s = s.reshape(2,21)
    print(s)
#sample = sample.reshape([2,21])
#plt.imshow(sample[1], cmap='gray')
#plt.savefig('test.png',dpi=600)



#Reshape the training and test set
x_train = x_train.reshape(x_train.shape[0], 2, 21, 1)
x_test = x_test.reshape(x_test.shape[0], 2, 21, 1)
#x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

#Padding the images by 2 pixels since in the paper input images were 32x32
x_train = np.pad(x_train, ((0,0),(1,1),(1,2),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(1,1),(1,2),(0,0)), 'constant')




#Standardization
mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)
x_train = (x_train - mean_px)/(std_px)


mean_px = x_test.mean().astype(np.float32)
std_px = x_test.std().astype(np.float32)
x_test = (x_test - mean_px)/(std_px)

#One-hot encoding the labels
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)

kname = 'output/keys_' + f[0:9] + '.json'


import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


model = Sequential()
#Layer 1
#Conv Layer 1
model.add(Conv2D(filters = 50,
                 kernel_size = [2,3],
                 strides = 1,
                 activation = 'relu',
                 input_shape = (4,24,1),padding='same')) # change
#Pooling layer 1
model.add(MaxPooling2D(pool_size = [1,2], strides = [1,2]))

#Layer 2
#Conv Layer 2
model.add(Conv2D(filters = 25,
                 kernel_size = [2,3],
                 strides = 1,
                 activation = 'relu',
                 input_shape = (4,12,6),padding='same')) # change

#Pooling Layer 2
model.add(MaxPooling2D(pool_size =[1,2], strides = [1,2]))

#Conv Layer 3
model.add(Conv2D(filters = 15,
                 kernel_size = [2,3],
                 strides = 1,
                 activation = 'relu',
                 input_shape = (4,6,6),padding='same')) # change

#Pooling Layer 3
model.add(MaxPooling2D(pool_size =[1,2], strides = [1,2]))
#Flatten
model.add(Flatten())

#Layer 3
#Fully connected layer 1
model.add(Dense(units = 120, activation = 'relu'))

#Layer 4
#Fully connected layer 2
model.add(Dense(units = 100, activation = 'relu'))

#Layer 5
#Output Layer
model.add(Dense(units = 2, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
plot_model(model,to_file='imgs/cnn_model.png',show_shapes=True)
exit(1)
model.fit(x_train , y_train, steps_per_epoch = 50, epochs = 30)

y_pred = model.predict_classes(x_test)
y_score=model.predict(x_test)

conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)
acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
print('Overall accuracy: {} %'.format(acc*100))

stats = {}
#stats['prob_class0'] = y_score[:,0].tolist()
stats['pred'] = y_score[:,1].tolist()
#stats['predicted_class'] = y_pred.tolist()
stats['true'] = y_test.tolist()


model_name='models/CNN_' + f[0:10]+'.h5'
model.save(model_name)

sname = 'output/stats_' + f[0:10] +'.json'
with open(sname, 'w') as f:
    json.dump(stats, f)


#Converting one hot vectors to labels


#index = np.arange(1, 9901)

#labels = labels.reshape([len(labels),1])
#index = index.reshape([len(index), 1])

#final = np.concatenate([index, labels], axis = 1)

#Prediction csv file
#np.savetxt("pred_1.csv", final, delimiter = " ", fmt = '%s')
