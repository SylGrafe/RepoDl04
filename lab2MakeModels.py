# -*- coding: utf-8 -*-
# sygr0003 , UMU54907 , VT2019 , lab2 weather predictions
# create models  
import sys
import keras
keras.__version__
import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt

import keras
keras.__version__
from keras import models
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten,Input
from tensorflow.keras import backend as K
from keras.optimizers import RMSprop
from tensorflow.python.keras.layers import Input, LSTM, GRU , Bidirectional, Dense, Embedding



######################### function makeGGDDModel ()
# 2 gru layers   with dropout=0.1 ,  and recurrent_dropout = 0.5  , 2 dense layers 
def makeGGDDModel(TSLen, nbOfFeat,  batch_size=None , lrPar=0.001 , u1=32 , u2=64 ,
                  d1=32,d2=64 ):
    source = Input(shape=(TSLen, nbOfFeat),
                   batch_size=batch_size,
                   dtype=tf.float32, name='Input')
    gru1 = GRU(u1, name='GRU1' ,
         dropout=0.1,  recurrent_dropout=0.5, 
               return_sequences=True )(source)
    gru2 = GRU(u2, name='GRU2',
               dropout=0.1,  recurrent_dropout=0.5  )(gru1)
    
    dense1 = Dense(d1, name='Dense1')(gru2)
    dense2 = Dense(d2, name='Dense2')(dense1)

    predicted_var = Dense(1, name='Output')(dense2)
    
    model = tf.keras.Model(inputs=[source], outputs=[predicted_var])
    model.compile(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=lrPar),
        loss='mae' )
        
    return model



######################### function makeGGGDDModel ()
# 3 gru layers   with dropout=0.1 ,  and recurrent_dropout = 0.5 ,
#  2 dense layers  more than the dense layer for predicted_var
def makeGGGDDModel(TSLen, nbOfFeat,  batch_size=None , lrPar=0.001 , u1=32 , u2=64 ,u3=64 ,
                  d1=32,d2=64 ):
    source = Input(shape=(TSLen, nbOfFeat),
                   batch_size=batch_size,
                   dtype=tf.float32, name='Input')
    gru1 = GRU(u1, name='GRU1' ,
         dropout=0.1,  recurrent_dropout=0.5, 
               return_sequences=True )(source)
    gru2 = GRU(u2, name='GRU2',
               dropout=0.1,  recurrent_dropout=0.5 ,return_sequences=True  )(gru1)

    gru3 = GRU(u3, name='GRU3',
               dropout=0.1,  recurrent_dropout=0.5  )(gru2)
    
    dense1 = Dense(d1, name='Dense1')(gru3)
    dense2 = Dense(d2, name='Dense2')(dense1)

    predicted_var = Dense(1, name='Output')(dense2)
    
    model = tf.keras.Model(inputs=[source], outputs=[predicted_var])
    model.compile(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=lrPar),
        loss='mae' )
        
    return model







######################### function  makeLLDDModel ()
# 2 lstm  layers  with dropout=0.1 ,  and recurrent_dropout = 0.5 ,
#  2 dense layers more than the dense layer for predicted_var
def makeLLDDModel(TSLen, nbOfFeat,  batch_size=None , lrPar=0.001 , u1=32 , u2=64 ,
                  d1=32,d2=64 ):
    source = Input(shape=(TSLen, nbOfFeat),
                   batch_size=batch_size,
                   dtype=tf.float32, name='Input')
    lstm1 = LSTM(u1, name='LSTM1' ,
         dropout=0.1,  recurrent_dropout=0.5, 
               return_sequences=True )(source)
    lstm2 = LSTM(u2, name='LSTM2',
               dropout=0.1,  recurrent_dropout=0.5  )(lstm1)
    
    dense1 = Dense(d1, name='Dense1')(lstm2)
    dense2 = Dense(d2, name='Dense2')(dense1)

    predicted_var = Dense(1, name='Output')(dense2)
    
    model = tf.keras.Model(inputs=[source], outputs=[predicted_var])
    model.compile(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=lrPar),
        loss='mae' )
        
    return model



  



######################### function makeGRUGRUModel ()
def makeGRUGRUModel(TSLen, nbOfFeat,  batch_size=None , lrPar=0.001 , u1=32, u2=64, d1=1 ):
# 2 gru layers   with dropout=0.1 ,  and recurrent_dropout = 0.5  ,
#  max one extra dense layers  more than the dense layer for predicted_var

    source = Input(shape=(TSLen, nbOfFeat),
                   batch_size=batch_size,
                   dtype=tf.float32, name='Input')
    gru1 = GRU(u1, name='GRU1' ,
         dropout=0.1,  recurrent_dropout=0.5, 
               return_sequences=True )(source)
    gru2 = GRU(u2, name='GRU2',
               dropout=0.1,  recurrent_dropout=0.5  )(gru1)
    
    if d1 == 1:
      predicted_var = Dense(d1, name='Output')(gru2)
    elif d1 > 1:
      dense1 = Dense(d1, name='Dense1')(gru2)
      predicted_var = Dense(1, name='Output')(dense1)
    
    model = tf.keras.Model(inputs=[source], outputs=[predicted_var])
    model.compile(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=lrPar),
        loss='mae' )
        
    return model








######################### function makeGRU2Model ()
def makeGRU2Model(TSLen, nbOfFeat,  batch_size=None , lrPar=0.001):
# 3 gru layers with dropout=0.1 ,  and recurrent_dropout = 0.5 ,
# the first gru has always  32 unit and one second  gru has always  64 units
    source = Input(shape=(TSLen, nbOfFeat),
                   batch_size=batch_size,
                   dtype=tf.float32, name='Input')
    gru1 = GRU(32, name='GRU1' ,
         dropout=0.1,  recurrent_dropout=0.5, 
               return_sequences=True )(source)
    gru2 = GRU(64, name='GRU2',
               dropout=0.1,  recurrent_dropout=0.5  )(gru1)
    
    predicted_var = Dense(1, name='Output')(gru2)
    model = tf.keras.Model(inputs=[source], outputs=[predicted_var])
    model.compile(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=lrPar),
        loss='mae' )
        
    return model





######################### function makeGRU1Model ()
# 1 gru layer with 32 units
# This is the  first recurrent baseline from  the book
# 6.3-advanced-usage-of-recurrent-neural-networks.py

def makeGRU1Model(TSLen, nbOfFeat,  batch_size=None , lrPar=0.001):
    source = Input(shape=(TSLen, nbOfFeat),
                   batch_size=batch_size,
                   dtype=tf.float32, name='Input')
    gru1 = GRU(32, name='GRU')(source)
    predicted_var = Dense(1, name='Output')(gru1)
    model = tf.keras.Model(inputs=[source], outputs=[predicted_var])
    model.compile(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=lrPar),
        loss='mae' )
        
    return model





######################### function makeDenseModel ()
# simple dense model one layers with 32 units 
def makeDenseModel(TSLen, nbOfFeat,  batch_size=None , lrPar=0.001):
    """
Create the model , You must pass in an input shape and batch size as TPUs
(and XLA) require  fixed shapes.

    """
    
    source = Input(shape=(TSLen, nbOfFeat) ,
                   batch_size=batch_size,
                   dtype=tf.float32, name='Input')
    flat1 = tf.layers.flatten (source)
    dense1 = Dense (32, activation='relu') (flat1)
    predicted_var = Dense(1, name='Output')(dense1)
    model = tf.keras.Model(inputs=[source], outputs=[predicted_var])
    
    model.compile(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=lrPar),
        loss='mae')

    return model
  
######################### function makeLSTMLSTMModel ()
def makeLSTMLSTMModel(TSLen, nbOfFeat,  batch_size=None , lrPar=0.001 , u1=32 , u2=64 ):
# 2 lstm layers  with dropout=0.1 ,  and recurrent_dropout = 0.5 

    source = Input(shape=(TSLen, nbOfFeat),
                   batch_size=batch_size,
                   dtype=tf.float32, name='Input')
    lstm1 = LSTM(u1, name='LSTM1' ,
         dropout=0.1,  recurrent_dropout=0.5, 
               return_sequences=True )(source)
    lstm2 = LSTM(u2, name='LSTM2',
               dropout=0.1,  recurrent_dropout=0.5  )(lstm1)
    
    predicted_var = Dense(1, name='Output')(lstm2)
    model = tf.keras.Model(inputs=[source], outputs=[predicted_var])
    model.compile(
        optimizer=tf.train.RMSPropOptimizer(learning_rate=lrPar),
        loss='mae' )
        
    return model


  
######################################################################
############### main for test only
######################################################################
if __name__ == "__main__":
  import sys
  sys.path.insert(0, "/home/syl1/bin/python/deepL/lab2/lib")
  import lab2U00


  inputBS=128
  myUnits1 = 32 
  myUnits2 = 64
  myD1=64
  myD2 =128
  # for some values of  myNbOfBatchsPerEpoch 
  # there is a long time between each epoch
  myNbOfBatchsPerEpoch=500
  myLr=0.001

  #  init data generators 
  TpuAddress = lab2U00.initData (doRunLocal=True)


  tSLen1 =  lab2U00.getTSLen()
  nbOfFeat1 = lab2U00.getFeatNb()

  # create a model
  modelStruct="gggdd"
  if modelStruct == "dense":    
      model=makeDenseModel (tSLen1 , nbOfFeat1, batch_size = inputBS, lrPar=myLr )
  elif modelStruct == "gru1":    
      model=makeGRU1Model (tSLen1 , nbOfFeat1, batch_size = inputBS,  lrPar=myLr)
  elif modelStruct == "gru2":    
      model=makeGRU2Model (tSLen1 , nbOfFeat1, batch_size = inputBS,  lrPar=myLr)
  elif modelStruct == "grugru":    
      model=makeGRUGRUModel (tSLen1 , nbOfFeat1, batch_size = inputBS,  
          lrPar=myLr , u1=myUnits1 , u2=myUnits2  , d1=myD1)
      infoStr="u1:%d, u2:%d , d1:%d" % (myUnits1 , myUnits2 , myD1)
  elif modelStruct == "ggdd":    
      model=makeGGDDModel (tSLen1 , nbOfFeat1, batch_size = inputBS,  
          lrPar=myLr , u1=myUnits1 , u2=myUnits2  , d2=myD2 ,  d1=myD1)
      infoStr="u:%d:%d , d:%d,%d" %    (myUnits1 , myUnits2 , myD1 , myD2)    
  elif modelStruct == "lldd":    
      model=makeLLDDModel (tSLen1 , nbOfFeat1, batch_size = inputBS,  
          lrPar=myLr , u1=myUnits1 , u2=myUnits2  , d2=myD2 ,  d1=myD1)
      infoStr="u1:%d:%d , d:%d,%d" %    (myUnits1 , myUnits2 , myD1 , myD2)    
  
  elif modelStruct == "lstmlstm":    
      model=makeLSTMLSTMModel (tSLen1 , nbOfFeat1, batch_size = inputBS,  
          lrPar=myLr , u1=myUnits1 , u2=myUnits2)
      infoStr="u:%d,%d " % (myUnits1 , myUnits2  )
  elif modelStruct == "gggdd":  
    myU3=64
    model=makeGGGDDModel (tSLen1 , nbOfFeat1, batch_size = inputBS,  
        lrPar=myLr , u1=myUnits1 , u2=myUnits2 ,u3=myU3 , d2=myD2 ,  d1=myD1)
    infoStr="u:%d,%d,%d , d:%d,%d " %    (myUnits1 , myUnits2, myU3,  myD1 , myD2)    
  
      
  else:  
    print (" \n***** EXIT  , %s invalid modeStruct" % (modelStruct))  
    sys.exit(1)
  
  
  

