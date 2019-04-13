#!/usr/bin/env python
# coding: utf-8
# from 6.3-advanced-usage-of-recurrent-neural-networks.py
#  study data  + run simply fully-connected model  and evaluate_naive_method ()
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
import time

################# global data to be set in getVal 
float_data =None
lookback = None
step = None
delay = None
batch_size = None
tryTPU = False
secSinceEpochs=[]  # keep the dates as integer seconds since epoch

theWeatherDataFilename = 'jena_climate_2009_2016.csv'
fname = "" # full path to theWeatherDataFilename depends of the environment (local or colab)

# a propos the format of dates  recored in  'jena_climate_2009_2016.csv'
#  '01.01.2009 00:10:00'
dateFmt = "%d.%m.%Y %H:%M:%S"






################################# function initData ()
# must be call prior to calling  other  any functions
# get the right path to retrieve the data
# retrieve the data from the file into array
# retrive the TPU address if any

def initData (doRunLocal=True):
    # get the repo
    global tryTPU
    global fname

    # the pgm may be run in colab or on local host
    if doRunLocal:
      data_dir = '/home/syl1/doNotBackup/RepoDl04' 
    else:
      os.system('git clone https://github.com/SylGrafe/RepoDl04.git')
      data_dir = 'RepoDl04'
      tryTPU=True  

    fname = os.path.join(data_dir, theWeatherDataFilename)
    getData ()

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


############################ function getDateStamples ()

def getDateStamples ():
# retrun the date corresponding to the records that will be yield in the generators
  global secSinceEpochs

  # it takes some time to convert all date to seconds sine epoch
  if ( len(secSinceEpochs) == 0 ):

    print ("getDateStamples () will read " , fname)
    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    for i, line in enumerate(lines):
      # keep only the relevant dates one each "step line"
      if ( i % step == 0):
        # keep the date as integer (seconds since epochs)
        timeStr= line [:line.index(',') ]
        timeStruct = time.strptime(timeStr, dateFmt)
        secSinceEpochs.append (time.mktime(timeStruct) )
      
        # to print back the date  as a str use   fromtimestamp
        # ts_epoch =  secSinceEpochs [0]
        # testStr= datetime.datetime.fromtimestamp(ts_epoch).strftime(dateFmt)
        # print ("DEBUG getData() , firstDate " , testStr  ) 
        # sys.exit (1)

      

  return  secSinceEpochs


############################ function getOneDate ()

def getOneDate ( rawInd ):
# retrun the date   corresponding to the  line in the file
# as  only ONE of step line is kept for the data generator 
# also only one of step data is keept in the array  secSinceEpochs

  if ( len(secSinceEpochs) == 0 ):
    # the array is empty must read the file once
    getDateStamples ()
  stepInd = rawInd // step 
  if ( stepInd >= len (secSinceEpochs) or stepInd < 0 )  :
    print ("getOneDate () %d invalid rawInd" % (rawInd))
    return None
  else:
    ts_epoch=secSinceEpochs[stepInd]
    dateStr = datetime.datetime.fromtimestamp(ts_epoch).strftime(dateFmt)
    return dateStr



############################ function getData
# create the dataset from the content  of filename 
# normalise the training dataset
# Let's convert all of these 420,551 lines of data into a Numpy array:

def getData ():

  global float_data
  global lookback
  global step 
  global delay 
  global batch_size 

  if (float_data is None):
    #  float_data does not exist so you must read the file to calculate float_data
    print ("getData () will read " , fname)
    f = open(fname)
    data = f.read()
    f.close()

    # define the other characteristics 
    lookback = 1440  # keep  10 previous days of data in the temporal sequence
    step = 6   # keep only one data per hour
    delay = 144  # look ahead  time is one day
    batch_size = 128


    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
      values = [float(x) for x in line.split(',')[1:]]
      float_data[i, :] = values
    mean = float_data[:200000].mean(axis=0)
    float_data -= mean
    std = float_data[:200000].std(axis=0)
    float_data /= std
  
  return float_data


######################### abstract function generator ()
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


#########################  funtion getOneBatch ()
def getOneBatch (  min_index, max_index,
              shuffle=False, batch_size=128 ):

        if lookback == None:
          return
        if delay == None:
          return
        step=6  


        if max_index is None:
            max_index = len(float_data) - delay - 1
        i = min_index + lookback
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
                           float_data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = float_data[indices]
            targets[j] = float_data[rows[j] + delay][1]
        return samples, targets







#################### functions to retrun the generators 
def getValSteps ():
  float_data = getData ()
  # This is how many steps to draw from `val_gen`
  # in order to see the whole validation set:
  print ("" , batch_size) 
  val_steps = (300000 - 200001 - lookback) // batch_size
  return val_steps

def getTestSteps ():
  float_data = getData ()

  # This is how many steps to draw from `test_gen`
  # in order to see the whole test set:
  test_steps = (len(float_data) - 300001 - lookback) // batch_size
  return test_steps

def getFeatNb ():
  float_data = getData ()

  return float_data.shape[-1]

def getTSLen ():
# retrun the nb of elements in the temporal  sequence
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
  """
  # test retrieve the dates correspondind to the limit of the 
  # train data , val data and test data
  ii = 0  
  print ( "date [%d] : %s " % ( ii ,  getOneDate ( ii )  ) )
  ii = 200000
  print ( "date [%d] : %s " % ( ii ,  getOneDate ( ii )  ))
  ii = 300000
  print ( "date [%d] : %s " % ( ii ,  getOneDate ( ii )  ))
  ii = 420550
  print ( "date [%d] : %s " % ( ii ,  getOneDate ( ii )  ))
  """


  
  model = Sequential()
  model.add(layers.Flatten(input_shape=(lookback // step, getFeatNb())))
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
  
  
  
