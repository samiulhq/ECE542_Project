#   AE.py
#   an autoencoder to produce lower dimensional gene expression data


from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import csv

from TCdata import TCdata
DATA_PATH = 'data/'

f = 'D4_100_5_timeseries.tsv'
n = 'D4_100_5_goldstandard.tsv'


# this is the size of our encoded representations
encoding_dim = 3  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(21,))
# "encoded" is the encoded representation of the input
encoded = Dense(64, activation='relu')(input_img)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
#encoded = Dense(encoding_dim, activation='relu')(input_img)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(21, activation='tanh')(decoded)
#decoded = Dense(21, activation='tanh')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (21-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-3]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mean_squared_error')

in_data =  TCdata(DATA_PATH + f, DATA_PATH + n, f[0:2])
data = in_data.extractData2([])

genes = list(data.keys())
genes_train = genes[0:800+1]
genes_test = genes[800:-1]

patterns = np.array(list(data.values()))
patterns_train = patterns[0:800]
print(patterns_train[1].shape)
patterns_test = patterns[800:]
plot_model(autoencoder,
              to_file='imgs/AE_model.png',
              show_shapes=True)
autoencoder.fit(patterns_train, patterns_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(patterns_test, patterns_test))
            # encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(patterns_test)
decoded_imgs = decoder.predict(encoded_imgs)


#  generate lower dimension representation for all patterns
patterns3D = encoder.predict(patterns)
o = f.split('.')[0]
o = o + '_3D' + '.tsv'
out_data = {}
for i, g in enumerate(genes):
    out_data[g] = patterns3D[i]
with open('output/' + o, 'w') as f:
    for key in out_data.keys():
        l = key
        for v in out_data[key]:
            l = l + '\t' + str(v)
        f.write("%s\n"%l)


# Plot in 3-D
import matplotlib.pyplot as plt

f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10,10))
for k, v in out_data.items():
    ax1.scatter(v[0],v[1])
    ax1.set_xlabel('1st Dimension')
    ax1.set_ylabel('2nd Dimension')
    ax2.scatter(v[0],v[2])
    ax2.set_xlabel('1st Dimension')
    ax2.set_ylabel('3rd Dimension')
    ax3.scatter(v[1],v[2])
    ax3.set_xlabel('2nd Dimension')
    ax3.set_ylabel('3rd Dimension')
plt.savefig('imgs/lowdim.png')
plt.show()
