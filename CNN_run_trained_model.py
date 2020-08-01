import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.models import load_model
import numpy as np
from TCdata import TCdata
from sklearn.metrics import confusion_matrix
DATA_PATH = 'data/'
f = 'D4_100_1_timeseries.tsv'
n = 'D4_100_1_goldstandard.tsv'