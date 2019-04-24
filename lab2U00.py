#!/usr/bin/env python
# coding: utf-8
# sygr0003 , UMU54907 , VT2019 , lab2 weather predictions
# from 6.3-advanced-usage-of-recurrent-neural-networks.py
# read data , preprocess the data and create generators
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
import time
import datetime


################# global data most of them to be set by  some  function
float_data =None # the data wil lbe normalized
lookback = None # related to the lenth of the timesequence
selInterval = None  # use to select part of the raw data
timeAhead = None # related to the timeAhead for the prediction
batch_size = None # batch size for the generators
tryTPU = False
secSinceEpochs=[]  # to keep the dates as integer seconds since epoch
theMean = 0 # the mean value use for the  normalisation of the temperature
theStd = 0 # the standart variation used for the normalisation f the temperature
theWeatherDataFilename = 'jena_climate_2009_2016.csv'
fname = "" # full path to theWeatherDataFilename depends of the environment (local or colab)

rawIndTest0 = None

"""
contain of one line 
['"Date Time"', '"p (mbar)"', '"T (degC)"', '"Tpot (K)"', '"Tdew (degC)"', '"rh (%)"', '"VPmax (mbar)"', '"VPact (mbar)"', '"VPdef (mbar)"', '"sh (g/kg)"', '"H2OC (mmol/mol)"', '"rho (g/m**3)"', '"wv (m/s)"', '"max. wv (m/s)"', '"wd (deg)"']
"""
# a propos the format of dates  recored in  'jena_climate_2009_2016.csv'
#  '01.01.2009 00:10:00'
dateFmt = "%d.%m.%Y %H:%M:%S"





#################################### functions ##############################
################################# function initData ()
# must be called prior to calling  other  any functions
# get the right path to the cvs file  contains the data
# read the file and put  the data into float_arr a np.array
# retrieve the TPU address if any

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
    # create the nb arrays
    print ("DEBUG initData() fname" , fname )
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



############################ function getData ()
# create the dataset from the content  of filename 
# Let's convert the  420,551 lines of data into a Numpy array:

def getData ():

  global float_data
  global lookback
  global selInterval 
  global timeAhead 
  global batch_size 
  global theMean
  global theStd

  if (float_data is None):
    #  float_data does not exist so you must read the file to calculate float_data
    print ("getData () will read " , fname)
    f = open(fname)
    data = f.read()
    f.close()

    # define the other characteristics 
    lookback = 1440  # keep  10 previous days of data in the temporal sequence
    selInterval = 6   # keep only one record per hour ( 6*10mn = 60mn = 1h)
    timeAhead = 144  #  predict for next day , 1day=24h <=> 24*6=144 elements ahead in float_data
    batch_size = 128  
    """
    about batch size  , 128 element in  a batch issued from float_data 
    gives a  time shift of  (126/6) hours between the first and last element of  in a batch 
    128/6 hours =  21.3 hours =  21h and 20 minutes. 
    """

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]

    float_data = np.zeros((len(lines), len(header) - 1))
    for i, line in enumerate(lines):
      values = [float(x) for x in line.split(',')[1:]]
      float_data[i, :] = values

    # normalise the dataset using mean and std from training data !!
    theMean = float_data[:200000].mean(axis=0)
    float_data -= theMean
    theStd = float_data[:200000].std(axis=0)
    float_data /= theStd

    # remember that the temperature  is found at   float_data[i][1]
    # print ("DEBUG getData () temperature  std: %.4f mean %.4f" % (std[1] , mean[1]))
    # this gave   DEBUG getData () std: 8.8525 mean 9.0773
    
  return float_data


######################### abstract function generator ()
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
#  Remember that ony  "one of selInterval" line from the file is kept for building the data
# in a timesteps see indices in the code

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
        # the highest indices of sampel to be yield is indices[-1]
        # because of the steps is may differ with rows[-1]
        # but last  target in a batch  will always be found at rows[-1] + delay]
        yield samples, targets




        
######################### abstract function rawIndGen ()
# generate the ind corresponding to the oldest data from one batch
# TODO  20190424  verif not in use anymore
def rawIndGen (data, lookback, delay, min_index, max_index,
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
            # rows is an array of length batch_size
            # each el is increase by one compare to the previous element
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        # I really  hope that  rows[-1] about the highest indice 
        #  samples yield by the  generator at each batch
        yield rows[-1]

        

# calculate real temperatures using std and mean 
def realTemp (temp):
  return theStd[1]*temp + theMean[1]


#########################  funtion getOneBatch ()
# TEST TEST retrun one batch sample
# TODO  20190424  verif not in use anymore
def getOneBatch (  min_index, max_index,
              shuffle=False, batch_size=128 ):

        if lookback == None:
          return
        if timeAhead == None:
          return
        selInterval=6  

        if max_index is None:
            max_index = len(float_data) - timeAhead - 1
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
                           lookback // selInterval,
                           float_data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], selInterval)
            samples[j] = float_data[indices]
            targets[j] = float_data[rows[j] + timeAhead][1]
        return samples, targets




########################  diverse functions 
def getValSteps ():
  float_data = getData ()
  # This is how many steps to draw from `val_gen`
  # in order to see the whole validation set:
  # print ("DEBUG getValSteps () batch_size: " , batch_size) 
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
# return the nb of elements in the temporal  sequence
  float_data = getData ()
  return lookback // selInterval




#################### functions to retrun the train , val and test generators 
def getTrainGen ():
  float_data = getData ()
  train_gen = generator(float_data,
                      lookback=lookback,
                      delay=timeAhead,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=selInterval, 
                      batch_size=batch_size)
  return train_gen
  

def getValGen ():
  float_data = getData ()

  val_gen = generator(float_data,
                    lookback=lookback,
                    delay=timeAhead,
                    min_index=200001,
                    max_index=300000,
                    step=selInterval,
                    batch_size=batch_size)

  return val_gen


def getTestGen ():
  float_data = getData ()

  test_gen = generator(float_data,
                     lookback=lookback,
                     delay=timeAhead,
                     min_index=300001,
                     max_index=None,
                     step=selInterval,
                     batch_size=batch_size)

  return test_gen


#################### function genXYZ ()
# define another generator to retrieve batches that do not  # always starts at min_index=300001 
# theMinIndex  is a parameter to the generator
# usefull for getting  specific  predictions 
def getGenXYZ  (theMinIndex=300001):
  float_data = getData ()

  genXYZ =  generator(float_data,
                     lookback=lookback,
                     delay=timeAhead,
                     min_index=theMinIndex,
                     max_index=None,
                     step=selInterval,
                     batch_size=batch_size)

  return genXYZ


# rawInd corresponding to getGenXYZ
def getRawIndGenXYZ ( theMinIndex=300001):
# TODO 20190424 verif that it is not in use anymore
  float_data = getData ()

  rawInd_genXYZ = rawIndGen (float_data,
                     lookback=lookback,
                     delay=timeAhead,
                     min_index=theMinIndex,
                     max_index=None,
                     step=selInterval,
                     batch_size=batch_size)

  return rawInd_genXYZ


###############################3 functions getIndices
def getIndiceXYZ ( min_index , batchNb ):
# retruns the indice of the first target yield by  getGenXYZ () for batchNb

  firstTargetInd = min_index + lookback + ( batchNb * batch_size)  + timeAhead
  return firstTargetInd 




# generate the indices corresponding to the testgen
def getTestRawIndGen ():
# TODO 20190424 verif that it is not in use anymore
  float_data = getData ()

  test_rawIndGen = rawIndGen (float_data,
                     lookback=lookback,
                     delay=timeAhead,
                     min_index=300001,
                     max_index=None,
                     step=selInterval,
                     batch_size=batch_size)

  return test_rawIndGen

############################ function getDateStamples ()

def getDateStamples ():
# populate the  array  secSinceEpochs
# containing the the dates corresponding to the records yield by the generators
# also only "one of selInterval" dates  is kept in the array  secSinceEpochs
# the dates are kept as seconds since epoch

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
      # keep only the relevant dates one each "selInterval line"
      if ( i % selInterval == 0):
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
# return the date (as a string )  corresponding to the  line rawInd in the file
# but remember that a   only ONE of selInterval line is kept for the data generator 

  if ( len(secSinceEpochs) == 0 ):
    # the array is empty must read the file once
    getDateStamples ()
  stepInd = rawInd // selInterval 
  if ( stepInd >= len (secSinceEpochs) or stepInd < 0 )  :
    print ("getOneDate () %d invalid rawInd" % (rawInd))
    return None
  else:
    ts_epoch=secSinceEpochs[stepInd]
    dateStr = datetime.datetime.fromtimestamp(ts_epoch).strftime(dateFmt)
    return dateStr


############################ function getOneFloatData ()

def getOneFloatData ( rawInd ):
# return the floatData coresponding to the line  rawInd in the cvs file
# but remember that a   only ONE of selInterval line is kept for the data generator 

  if ( rawInd >= len (float_data) or rawInd < 0 )  :
    print ("getOneFloatData  () %d invalid rawInd" % (rawInd))
    return None
  else:
    return float_data[rawInd]
    return dateStr
  
  
###################################### function readValues FromFile ()
# read some selected lines from the file
# retruns the dates and temp and tempaHead extracted from theses lines
def readValuesFromFile (indArr , offset):
# TODO  20190424  verif not in use anymore
    readTempArr = [] 
    readDateArr=[]
    tempAheadArr = [] 
    aheadInterval=144
    print ("readValuesFromFile () will read " , fname)
    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    ik = 0 
    aHeadInd = -1
    for ij, line in enumerate(lines):
      if ik < len(indArr) and ij == (indArr[ik] + offset): 
        #print ("DEBUG readValuesFromFile found " , ij , end='-> ' )
        aHeadInd=indArr[ik] + offset +aheadInterval
        lineArr= line.split(',')
        readTemp=float(lineArr[2])
        readDate =  lineArr[0]
        readTempArr.append(readTemp)  
        readDateArr.append(readDate)  
        ik+=1
      # fetch the value one day ahead of the latest value in tempArr  
      if  aHeadInd > -1 and ij == aHeadInd: 
        #print ("DEBUG ahead " , ij)
        lineArr= line.split(',')
        readTemp=float(lineArr[2])
        tempAheadArr.append(readTemp)  
    
    
      if (ij > indArr[-1] + offset + aheadInterval + 2):
        # print ("DEBUG break att" , ij )
        break
      
    return  readDateArr , readTempArr , tempAheadArr


  
###################################### function readValues FromFile ()
# read some selected lines from the file
# retruns the dates and temp and tempaHead extracted from theses lines
def readTempsInFile (startInd , bs, offset):

    readTempArr = [] 
    startDate=None
    previousDayTempArr = [] 
    aheadInterval=144
    # print ("DEBUG readTempsInFile () will read " , fname)
    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    ik = 0 
    ii=0  
    aHeadInd = -1
    for ij, line in enumerate(lines):
      if  (ij == (startInd  + offset+ ik)  and ik< bs): 
        if ij == 0 :
          # print ("DEBUG readTempsInFile() found first readTemp at " , ij , end='-> ' )
          pass
        lineArr= line.split(',')
        readTemp=float(lineArr[2])
        readTempArr.append(readTemp)  
        # retrieve the date of the first temp in the array
        if ik == 0 :   
          startDate =  lineArr[0]
        ik+=1
         
      # fetch the values  one day before   
      if  (ij == startInd  + offset+ ii  -aheadInterval  and ii< bs): 
        if ii == 0: 
          # print ("DEBUG readTempsInFile () found first previous temp at " , ij , end='-> ' )
          pass
        lineArr= line.split(',')
        readTemp=float(lineArr[2])
        previousDayTempArr.append(readTemp)  
        ii+=1    
    
      if (ik > bs + 2):
        # print ("DEBUG readtempsInfile () break att" , ij )
        break
      
    return  startDate , readTempArr , previousDayTempArr



######################################################################
############### main for test only
######################################################################



# if __name__ == "__main__":
if False:


  import keras
  keras.__version__

  from keras import layers
  from keras import models
  from keras.models import Sequential
  from keras import layers
  from keras.optimizers import RMSprop
  
  
  initData (doRunLocal=False)

  # choose indices that are multiple of batch_size  
  indArr =  [0,128 ,256 , 384 , 512 ]
  rawIndArr=[]
  dateArr= []
  ik=0
  ij=0

  # get the corresponding indices( in float_data  array ) from  testRawIndGen

  for rawInd in getTestRawIndGen():
    # print ("%d %d --> %d" % (ij , ik , rawInd))
    if ik < len(indArr) and ij == indArr[ik]:
      rawIndArr.append(rawInd)
      dateArr.append(getOneDate ( rawInd ))      
      ik+=1

    ij +=1
    if (ij > indArr[-1]):
      break

      
  # get corresponding  temperature  from testGen
  tempsFromLabelBatch= []
  tempsFromDataBatch=[]
  ik=0
  ij=0

  for data_batch, labels_batch  in getTestGen():
    if ik < len(indArr) and ij == indArr[ik]:
      tempsFromLabelBatch.append(labels_batch[0])
      tempsFromDataBatch.append(data_batch[0][0][1])

      ik+=1

    ij +=1
    if (ij > indArr[-1]):
      break
      

  # read the values direcly from the file      
  csvDates , csvTemp  = readValuesFromFile (rawIndArr  , -2) 

  # make a nice print   
  print ( "rawInd temps : Fromdata -> label \n  tempFloatData  \n date adn temp from csv")
  for ijk in range(len(dateArr)):
    print ( "\n%d  %d %s  %.2f -> %.2f" % ( 
      ijk , rawIndArr[ijk] , dateArr[ijk] , 
      realTemp(tempsFromDataBatch[ijk]) , realTemp(tempsFromLabelBatch[ijk])) )
  
    floatDataTemp =getOneFloatData (rawIndArr[ijk]) [1] 
    print ("float_data --------->  %.2f" %   realTemp(floatDataTemp))
    print ("csv --->     %s %.2f" %       ( csvDates[ijk] , csvTemp[ijk] ))
  
    



######################################################################
############### main for test only
######################################################################

if False :


  import keras
  keras.__version__

  from keras.datasets import mnist
  from keras.utils import to_categorical
  from keras import layers
  from keras import models
  from keras.models import Sequential
  from keras import layers
  from keras.optimizers import RMSprop
  
  
  initData ()
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


if __name__ == "__main__":

    initData ()

    # test getGenXYZ ()
    print ("DEBUG DEBUG getGenXYZ ()")
    myMinIndex=0
    # get the data and labels from the test generator for the  choosenBatch batch
    batchCounter=0
    choosenBatch=0
    for data_batch, labels_batch in  getGenXYZ  (theMinIndex=myMinIndex):
      batchCounter+=1
      if (batchCounter >= choosenBatch):
        print ("DEBUG break getGenXYZ  after %d batches" % (batchCounter))
        break

    # get the corresponding rawInd    for this batch
    firstTargetInd=  getIndiceXYZ ( myMinIndex , choosenBatch )
    print ("firstTargetInd=%d" %  (firstTargetInd ) )

    batchCounter=0
    for rawInd in getRawIndGenXYZ (theMinIndex=myMinIndex):
      batchCounter+=1
      print (" %d--> %d" % ( batchCounter , rawInd))
      if (batchCounter >= choosenBatch):
        print ("DEBUG break getRawIndGenXYZ  after %d batches" %   (batchCounter))
        print ( " minInd:%d  gave  rawInd: %d " % ( myMinIndex , rawInd) )
        break

    # retrieve the data from the file   
    firstDate , csvTempDay2 , csvTempDay1 = readTempsInFile (firstTargetInd , 128,  -2)

  

    plt.figure(  figsize=(10, 8))
    # plot the values from the csv file  
    ax = plt.subplot(2, 1, 1)
  
    plt.plot( csvTempDay1, 'b.', label='previous day')
    plt.plot(csvTempDay2, color="black", label='day (same as true)')
    theTitle ="naive model : read temp from csv file,  first stample at %s" %  (
      firstDate  )
    plt.title(theTitle)
    plt.legend()
  
    plt.show()
  

if False:  
    model = Sequential()
    model.add(layers.Flatten(input_shape=(lookback // selInterval, getFeatNb())))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
  
    model.compile(optimizer=RMSprop(), loss='mae')
    print ("model.fit_generator () ")
    history = model.fit_generator(getTrainGen () ,
                                steps_per_epoch=500,
                                epochs=20,
                                validation_data=getValGen(),
                                validation_steps=getValSteps())
  
  
    import matplotlib.pyplot as pltH
  
    loss = history.history['loss']
    val_loss = history.history['val_loss']
  
    epochs = range(len(loss))
  
    plt.figure()
  
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
  
    plt.show()
  
  

