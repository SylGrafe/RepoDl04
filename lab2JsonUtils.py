#!/usr/bin/python3.5

import sys
import os
import sys
import datetime
import json
from json import JSONDecoder, JSONDecodeError
from json import encoder
import re
import numpy as np
import traceback
"""
from keras import losses
from keras import metrics
"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from collections import namedtuple
encoder.FLOAT_REPR = lambda o: format(o, '.4f')
NOT_WHITESPACE = re.compile(r'[^\s]')
sortFlg = False

"""
Contains functions and the Class  ConfigAndResults meant to be used
to access results and characteristics  from models is saved on disk as json string

the functions are 
dump some results on disk 
read  results from disk
present a summary if  of the records 
ask the user for a indice to retrive a given record from the list


Remember 
dumps instance of ConfigAndResults  
but when reading info from disk will get  a dictionnary representation of a  instance ConfigAndResults and not a real  instance of   ConfigAndResults . there fore the use on namedtulped

Remember that this file must on the python path to be imported
"""

#################################################################################33
############################################# Class ConfigAndResults
class ConfigAndResults:
# The class represent the main information about a model and the training results 
# but as the info will be saved on disk there are almost no methods associated to the class
# instead   globals function    using  the dictionnary representation of the class wil lbe used

# the constructor
  def __init__(self,modelStruct, compInfo,histDict ,
          histParams , timeStamp, info="" ,h5 = ""  , testRes = None , codeRef="" ):
    self.modelStruct = modelStruct  # string to identify the model  layers
    self.compInfo = compInfo  # more info about compilation
    self.histDict = histDict # history.history
    self.histParams = histParams  # history.params
    self.timeStamp = timeStamp  
      # time at which the pgm which create a instance of this class was started
    self.info = info # if not empty more info to identify the test
    self.h5 = h5 # if not empty name of the saved model 
    self.testRes = testRes # if not empty test Resultat        
    self.codeRef = codeRef # name of the pgm who created the instance
  def print_params (self):
    print (self.histParams)

# overwrite __str__ used when printing an instance of the class
  def __str__(self):
    return "(%s %s \n%s s\n%s\n\n%s\n%s) " % (self.codeRef , self.modelStruct,
    self.compInfo,self.histDict , 
    self.histParams , self.timeStamp)


  def toString (self):
    # print ("here in toString")
    # no need too much precision , round the values of the floats 
    selfDict = round_floats(self.__dict__)
    selfNT = namedtuple("selfNT", selfDict.keys())(*selfDict.values())
    
    return "(%s %s at:%s \ntestRes:%s\ncompInfo: %s\ninfo:%s\n\n%s\n%s) " %     (
    selfNT.codeRef , selfNT.modelStruct , selfNT.timeStamp ,  
     selfNT.testRes ,      selfNT.compInfo,      selfNT.info,
    selfNT.histDict , selfNT.histParams ) 



############################### global functions  #####################################
##################################################### function  round_floats(...)
def round_floats(o):
    if isinstance(o, float): return round(o, 4)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o


##################################### function getMaxValAcc  ()
def getMaxValAcc (someNamedTuple):
# retrieve the max value in the array of val_acc 
# retrun also the indice at which this max value was found 
# and the length of the array of values
    # print ("here in getMaxValAcc ()")
    try : 
      theHistDict = someNamedTuple.histDict
      foundval_acc=False
      for ki in theHistDict.keys():
        if ki.find('val_acc') != -1 :
          foundval_acc=True
          theValues = theHistDict[ki]
          maxVal= max (theValues)
          indMaxVal = theValues.index (maxVal)
          maxValDict = { indMaxVal, maxVal}
          # print ( " max : val_acc[%d]  == %f" %( indMaxVal , maxVal ) )
          return maxVal , indMaxVal , len(theValues)
      if not   foundval_acc : 
          print ("doSomeTest val_acc not found  " )
          return None  , None , None 

    except Exception as e:
      exceptVar = traceback.format_exc()
      print( "exception while  doSomeTest   " )
      print( exceptVar   )
      return None  , None , None 



##################################### function getMinValLoss  ()
def getMinValLoss (someNamedTuple):
# retrieve the min  value in the array of val_loss 
# return also the indice at which this min  value was found 
# and the length of the array of values
    # print ("here in getMinValLoss ()")
    try : 
      theHistDict = someNamedTuple.histDict
      foundval_loss=False
      for ki in theHistDict.keys():
        if ki.find('val_loss') != -1 :
          foundval_loss=True
          theValues = theHistDict[ki]
          minVal= min (theValues)
          indMinVal = theValues.index (minVal)
          minValDict = { indMinVal, minVal}
          # print ( " min : val_loss[%d]  == %f" %( indMinVal , minVal ) )
          return minVal , indMinVal , len(theValues)
      if not   foundval_loss : 
          print ("doSomeTest val_loss not found  " )
          return None  , None , None 

    except Exception as e:
      exceptVar = traceback.format_exc()
      print( "exception while  getMinValLoss ()   " )
      print( exceptVar   )
      return None  , None , None 





##################################################### function sortArr ()
# sort the array of dictionnary  in reverse order using the testAccuracy as a criteria for sorting

def sortArr (theTests):
  # print ("\n\n sortArr")
  ii=0
  valArr=[]
  for someDict in theTests :
    valArr.append(someDict['testRes'][1])

  theNpArr = np.array(valArr)  
  theArgSortArr =  np.argsort(theNpArr)
  reverseSortedIndArr  = np.flip (theArgSortArr , 0) 

  # create a new arr of resulted sorted  (inverse order )by the value of testRes
  sortedTestRes = []
  for theSortedInd in reverseSortedIndArr:
    # print (theSortedInd)
    sortedTestRes.append  ( theTests [theSortedInd])
  return sortedTestRes  




################################################### function  dumpOnFile(...)
# dump an object on file using json
def dumpOnFile (someObj , theDumpFileName):
    # now let's  encode and  write  on disk
    print ("************ dumpOnFile () : append to file  %s  " % (theDumpFileName) )
    print ("\n Dumping object %s in the file  %s " % ( someObj.modelStruct , theDumpFileName))

    theDumpFile = open(theDumpFileName, 'a')
    json.dump(round_floats(someObj.__dict__),theDumpFile )

    # close the file, and your pickling is complete
    theDumpFile.close()
    

    

################################################### function plotHistogram(...)
# plot a histogram of labels values given a encodedlist of labels
def plotHistogram (encodedList ):
  decodedList=[]  
  for i in range(encodedList.shape[0]):
    encoded_label = encodedList[i]
    labelVal = np.argmax (encodedList[i])
    decodedList.append(labelVal)
  plt.hist( decodedList, bins=20) 
  plt.xlabel('labels')
  plt.ylabel('nbofLabels')
  #Display histogram
  plt.show()    

################################################### function plotHist(...)
# plot  acc (and val_acc if any ) in a subplot and 
# and loss  (and val_loss if any ) 
# given a instance of  ConfigAndResults or 
# a  namedTuple representing an instance of  ConfigAndResults
def plotHist (someNamedTuple ):

  try : 
    theHistDict = someNamedTuple.histDict
    theTitle  = someNamedTuple.codeRef + ", " +someNamedTuple.modelStruct +\
            ", "         + someNamedTuple.info

    if not isinstance (theHistDict, dict) :
      print ("\n**** plotHist() failure expecting a dict and got: %s"  % (type(theHistDict)) )
      return 

    symbols = ['b' , 'bo']
    ii=0
    
    # try to plot acc
    ax = plt.subplot(2, 1, 1)
    ax.set_title(theTitle)
    for ki in theHistDict.keys():
      if ki.find('acc') != -1 :
        # print(ki)
        theValues = theHistDict[ki]
        #print ("\n\n type(theValues ) "  , type(theValues ) , "---> " , theValues)
        epochs = range(1, len(theValues) + 1)
        plt.plot(epochs, theValues, symbols[ii%2], label=ki)
        plt.xlabel('Epochs')
        plt.ylabel('acc')
        ii+=1
    if (ii != 0 ):    
      plt.legend()
    
    ax = plt.subplot(2, 1, 2)
    jj= 0
    for ki in theHistDict.keys():
      if ki.find('loss') != -1 :
        # print(ki)
        theValues = theHistDict[ki]
        epochs = range(1, len(theValues) + 1)
        plt.plot(epochs, theValues, symbols[jj%2], label=ki)
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        jj+=1
    if (jj != 0 ):    
      plt.legend()
          
    # strange that you need to change hspace  and not wspace
    # to get  space enouth to see the tite of the second plot
    plt.subplots_adjust(hspace = 0.5)
    

    if (ii == 0 and jj== 0 ) :
      print ("plotHist() failure  there was neither acc or loss to plot ")
    else:
     plt.show()
      
  except Exception as e:
    print ("\n********** plotHist() fatal error: " , str(e))
  
################################################### function printAllFromFile(...)
def printAllFromFile (theDumpFileName , sorted=False):

  try : 

    theTests = readAllFromFile (theDumpFileName)
    if (theTests == None ):
      print ("printAllFromFile (%s) error file  empty or invalid format" % (theDumpFileName))
      return
    if (sorted):
      theArr=sortArr(theTests)
    else:
      theArr=theTests


    ii=0
    for someDict in theArr:
      ii+=1
      print(" %2d : %s\n" % (ii, someDict) )
    print ("************************************************")

  except Exception as e:
    print ("\n********** printAllFromFile(%s)  fatal error :\n%s" % (theDumpFileName , str(e)))



################################################### function  printHeadersFromFile(...)
# read dump file and print part of each dump 
# remember that the results are all dictionnaries
# possibility to sort the array of dictionnary  in reverse order 
# using the testAccuracy as a criteria for sorting

def printHeadersFromFile (theDumpFileName , sorted=False):

  try : 
    theTests = readAllFromFile (theDumpFileName)
    if (sorted):
      theArr=sortArr(theTests)
    else:
      theArr=theTests

    ii=0
    print(" indice <codeRef> timeStample, modelStruct, info "  )
    #print("test[loss,acc] <--> max (val_acc) at i/nb epochs:\n")
    print("test[loss,acc] <--> min (val_loss) at i/nb epochs:\n")

    for someDict in theArr:
    
      # use named tuple to access the dict as it would be real instance of ConfigAndResults 
      someNT = namedtuple("SomeNT", someDict.keys())(*someDict.values())
      # retrieve the best val accuracy  

      maxVal , indMaxVal , nbOfEpochs = getMaxValAcc(someNT)

      print(" %2d <%s> %s,  %s, %s" % 
      (ii, someNT.codeRef ,someNT.timeStamp , someNT.modelStruct , someNT.info ) )

      if (maxVal) :
        print(" %s <--> %s at %d/%d:\n" % 
        ( someNT.testRes  , maxVal , indMaxVal , nbOfEpochs ) )
      else:
        # print loss instead 
        minVal , indMinVal , nbOfEpochs = getMinValLoss(someNT)
        print(" %s <--> %s at %d/%d:\n" % 
        ( someNT.testRes  , minVal , indMinVal , nbOfEpochs ) )



      ii+=1
    print ("************************************************")

  except Exception as e:
    print ("\n********** printHeaderFromFile() %s fatal error %s" % (theDumpFileName , str(e)))





################################################### function getOneResFromFile(...)
# read in dumpfile the info about  the object referenced by refStr or ind
# return as a namedtuple  the first instance found
# the value of sorted should be the same as the one used for printHeadersFromFile
def getOneResFromFile (theDumpFileName , refStr= "" , ind=-1 , sorted=False ):
  try : 
    theTests = readAllFromFile (theDumpFileName)
    if (sorted):
      theArr=sortArr(theTests)
    else:
      theArr=theTests

    ii=0
    foundRes=False
    # remember that the info  read from the dump file are  dictionnaries
    for someDict in theArr:
      # use named tuple to access the dict as it would be real instance of ConfigAndResults 
      someNT = namedtuple("SomeNT", someDict.keys())(*someDict.values())
      
      # print ("DEBUG " , someNT)
      if (ii == ind or someNT.timeStamp == refStr ):
        foundRes=True
        print (" %s got record for  %d" % (refStr , ii)) 
        return someNT
      ii+=1

    if not  foundRes:
      print ("getOneResFromFile() no results for file:%s ref:%s ind:%s" %
         ( theDumpFileName , refStr , ind ))
    return None
  except Exception as e:
    print ("\n********** getOneResFromFile() %s fatal error %s" % (theDumpFileName , str(e)))
    return None




################################################### function printOneRes(...)
# expecting a namedtuple  similar to an instance of ConfigAndResults
# print some parts of the content
def printOneRes ( someResult , doAll):

  try : 
   if not doAll:
     print ("%s, %s, %s, %s " %  (someResult.codeRef ,someResult.modelStruct ,
                    someResult.compInfo , someResult.timeStamp)) 
     print (someResult.info)
     print ("test Results %s  " % (someResult.testRes  )) 
     # for fun show also the best val accuracy if there is some
     maxVal , indMaxVal , nbOfEpochs = getMaxValAcc(someResult)
     if maxVal : 
       print ("best val accuracy: at epochs  %d /%d  value %s" % ( 
                indMaxVal , nbOfEpochs , maxVal ) )

     #  loss is interesting also
     minVal , indMinVal , nbOfEpochs = getMinValLoss(someResult)
     print ("best val loss : at epochs  %d /%d  value %s" % ( 
                indMinVal , nbOfEpochs , minVal ) )


     print ("\nhistory Params: %s " % (  someResult.histParams)) 
     
   else:     
     print ("%s, %s, %s, %s " %  (someResult.codeRef ,someResult.modelStruct ,
                    someResult.compInfo , someResult.timeStamp)) 
     print (someResult.info)
     print ("test Results %s  " % (someResult.testRes  )) 
     # for fun show also the best val accuracy

     maxVal , indMaxVal , nbOfEpochs = getMaxValAcc(someResult)
     if maxVal : 
       print ("best val accuracy: at epochs  %d /%d  value %s" % ( 
                  indMaxVal , nbOfEpochs , maxVal ) )

     #  loss is interesting also
     minVal , indMinVal , nbOfEpochs = getMinValLoss(someResult)
     print ("best val loss : at epochs  %d /%d  value %s" % ( 
                indMinVal , nbOfEpochs , minVal ) )

     print ("\nhistory Params: %s " % (  someResult.histParams)) 
     print ("\nFist history: %s " % (  someResult.histDict)) 
     
     
  except Exception as e:
    print ("\n********** printOneRes()  fatal error %s" % ( str(e)))
    return None



################################################### function readAllFromFile ()
def readAllFromFile (theDumpFileName):

  # print ("readAllFromFile(%s)" % theDumpFileName)
  try : 
    if (not os.path.isfile(theDumpFileName)):
      print("readAllFromFile(),  error %s is not a file"  %  ( theDumpFileName ) )
      return

    theArr = []
    # now open a file for reading
    filePtr2 = open(theDumpFileName, 'r')
    document = filePtr2.read()
    # print (document)

    for obj in decode_stacked(document):
      theArr.append(obj)

    # close the file, just for safety
    filePtr2.close()
    return theArr

  except Exception as e:
    print ("\n********** readAllFromFile() fatal error :%s" % (theDumpFileName , str(e)))




############################## function decode_stacked(...)
def decode_stacked(document, pos=0, decoder=JSONDecoder()):
    while True:
        match = NOT_WHITESPACE.search(document, pos)
        if not match:
            return
        pos = match.start()

        try:
            obj, pos = decoder.raw_decode(document, pos)
        except JSONDecodeError:
            # do something sensible if there's some error
            raise
        yield obj


##################################### function doProceedUserInput ()

def doProceedUserInput (theDumpFileName):
  global sortFlg
  # get  user input
  # rawinput no data conversion
  myPrompt = "\nindStr [moreStr] > " 
  print(myPrompt, end=' ')

  try:
    theUserInfo ="""
      explore the content of the dumpfile %s
      if indStr== s     summary : print list of all records  in the file
      if indStr== o     ordered summary , best results first
      if indStr ==e     exit the Pgm
      if indStr== valid record indice       print part of the record
      if indStr==validindice and moreStr==a   print the record 
      if indStr==validindice and moreStr==p   print part of the record and plot the history
      """ % ( theDumpFileName)

    userStr= input()
    print ("...%s..." % userStr)
    if (userStr == "" or userStr.lower().find('h') != -1 or 
             userStr.lower().strip().find('?') != -1 ):
    
      # print ( theUserInfo)
      return   
    doPrintSummary =False
    if (userStr.lower().strip().find('o') == 0 ):
      # the user ask to see a summary of all records in the dumpfile
      sortFlg = True
      doPrintSummary =True
    elif (userStr.lower().strip().find('s') == 0 ):
      sortFlg = False
      doPrintSummary =True
    if (doPrintSummary) :
      printHeadersFromFile (theDumpFileName, sorted=sortFlg)
      return
    if (userStr.lower().find('e') == 0 ):
      print ("doProceedUserInput() exit")
      sys.exit(1)

    try:
      indStr , inputStr =  userStr.split()
    except ValueError:
      # maybe the user did not enter the string after the indice
      #print ("doProceedUserInput () here in ValueError ")
      #print ("doProceedUserInput invalid input")
      inputStr=" "
      indStr = userStr

    try:
      theInd=int ( indStr )
    except ValueError:
      # maybe the user did not enter the string after the indicise
      print ("doProceedUserInput invalid input %s should be an indice" % indStr)
      return

    someRes = getOneResFromFile (theDumpFileName , ind=theInd, sorted=sortFlg)
    if (someRes):
        # print ("type(someRes): " , type(someRes))
        if (inputStr.lower() == 'a'):
          printOneRes (someRes , True)
        else :
          printOneRes (someRes , False)
        if (inputStr.lower() == 'p'):
          plotHist(someRes)  
  except KeyboardInterrupt:
      print ("Bye , You hit control-C ")
      sys.exit(0)


  except ValueError:
    # maybe the user did not enter the string after the indicise
    print ("doProceedUserInput () here in ValueError ")
    print ("doProceedUserInput invalid input")

    inputStr=" "

  except KeyboardInterrupt:
    print ("Bye , You hit control-C ")
    sys.exit(0)
  except Exception as e:
      exceptVar = traceback.format_exc()
      print( "exception in doProceedUserInput()  " )
      print( exceptVar   )






############################################### main 
#  test create an object and  dump it in the benchmark


if __name__ == "__main__":

  doDump=False
  theDummyDumpName = "dummy.json"


  if doDump:
    dummyHist={'h1':"dummy" , 'h2':"dummy"}
    dummyParams={'p1':"dummy" , 'p2':"dummy"}
    dummyObj = ConfigAndResults ('dummyinfo' , "dummy compilering info", 
        dummyHist ,  dummyParams,
    datetime.datetime.now().strftime("%d%m at %H:%M:%S") )
    print  ("TEST TEST wil ltry to dump dummy object :%s" % (dummyObj)  )

    dumpOnFile (dummyObj , theDummyDumpName)


  doTestSorting=True
  theDummyDumpName = "../repo/results.json"
  # printAllFromFile(theDummyDumpName)

  if (doTestSorting):
    print ("\n\n present the results  sorted by  some criteria  ")
    theTests = readAllFromFile (theDummyDumpName)
    sortedTestRes = sortArr2 (theTests)
  
    print ("list of results sorted by  test accuracy , (inverse order))")
    ii=0
    for someDict in sortedTestRes :
      someNT = namedtuple("SomeNT", someDict.keys())(*someDict.values())
      # someNT.testRes is a list [loss ,acc] so the 2 el is the acc
      print ("%d %f  %s  %s" % (ii , someNT.testRes[1] , 
      someNT.codeRef , someNT.modelStruct ))
      ii+=1




  doSomeTest=False
  if (doSomeTest):
    theDummyDumpName = "../repo/results.json"
    someRes = getOneResFromFile (theDummyDumpName  , ind=0)
    print ("testing  getMaxValAcc:  " , getMaxValAcc (someRes) )    


print ("\n")


  


