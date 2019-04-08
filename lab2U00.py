#!/usr/bin/env python
# coding: utf-8
# from 6.3-advanced-usage-of-recurrent-neural-networks.py
#  study data  + run simply fully-connected model  and evaluate_naive_method ()
import os
from matplotlib import pyplot as plt
import numpy as np
import sys

################# global data to be set in getVal 
float_data =None
lookback = None
step = None
delay = None
batch_size = None
tryTPU = False




################################# functions
def initData (doRunLocal=True):
    # get the repo
    global tryTPU

    if doRunLocal:
      data_dir = '/home/syl1/doNotBackup/RepoDl04' 
    else:
      os.system('git clone https://github.com/SylGrafe/RepoDl04.git')
      data_dir = 'RepoDl04'
      tryTPU=True  

    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
    getData (theFilename=fname)

    if tryTPU:
      # check for TPU  
      try:
        device_name = os.environ['COLAB_TPU_ADDR']
        TPU_ADDRESS = 'grpc://' + device_name
        print('Found TPU at: {}'.format(TPU_ADDRESS))
        return TPU_ADDRESS

      except KeyError:
        print('TPU not found')
        tryTPU=False
        return None




def getData (theFilename=None):
  global float_data
  global lookback
  global   step 
  global delay 
  global batch_size 

  fname = theFilename
  if (float_data is None):

    print ("getData () will read " , fname)
    f = open(fname)
    data = f.read()
    f.close()

    # define the other characteristics 
    lookback = 1440
    step = 6
    delay = 144
    batch_size = 128


    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]


    # Let's convert all of these 420,551 lines of data into a Numpy array:

    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
      values = [float(x) for x in line.split(',')[1:]]
      float_data[i, :] = values

    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std
  
  return float_data


def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):



    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets





#print (type (train_gen))

def getValSteps ():
  float_data = getData ()
  # This is how many steps to draw from `val_gen`
  # in order to see the whole validation set:
  val_steps = (300000 - 200001 - lookback) // batch_size
  return val_steps

def getTestSteps ():
  float_data = getData ()

  # This is how many steps to draw from `test_gen`
  # in order to see the whole test set:
  test_steps = (len(float_data) - 300001 - lookback) // batch_size
  return test_steps

def getShape ():
  float_data = getData ()

  return float_data.shape[-1]

def getXXYY ():
  float_data = getData ()
  return lookback // step




def getTrainGen ():
  float_data = getData ()
  train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)
  return train_gen
  

def getValGen ():
  float_data = getData ()

  val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)

  return val_gen


def getTestGen ():
  float_data = getData ()

  test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

  return test_gen


if __name__ == "__main__":



  import keras
  keras.__version__

  import datetime
  from keras.datasets import mnist
  from keras.utils import to_categorical
  from keras import layers
  from keras import models
  from keras.models import Sequential
  from keras import layers
  from keras.optimizers import RMSprop
  
  
  initData ()
  model = Sequential()
  model.add(layers.Flatten(input_shape=(lookback // step, getShape())))
  model.add(layers.Dense(32, activation='relu'))
  model.add(layers.Dense(1))
  
  model.compile(optimizer=RMSprop(), loss='mae')
  print ("model.fit_generator () ")
  history = model.fit_generator(getTrainGen () ,
                                steps_per_epoch=500,
                                epochs=20,
                                validation_data=getValGen(),
                                validation_steps=getValSteps())
  
  
  import matplotlib.pyplot as plt
  
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  
  epochs = range(len(loss))
  
  plt.figure()
  
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  
  plt.show()
  
  
  
