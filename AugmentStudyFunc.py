#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Check for GPU Support
import tensorflow as tf
print(tf.__version__ )
tf.config.experimental.list_logical_devices('GPU')


# ## Imports



import tensorflow as tf
#from tensorflow.keras.layers import Input
#import tensorflow.keras.applications as tf_app
#import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
from core import class_prep as pcm
#import random
import os
#from tensorflow.keras.optimizers import SGD,Adam
#import albumentations as A
#from sklearn.preprocessing import LabelBinarizer




def prefilter_DataSet(datasetCSV,withAugmentation=False,labelColumnName='category_2',trainSet=['without_shaders']):

    
    ###### -- Prefilter the dataframe 
    df_in=pd.read_csv(datasetCSV,sep=';') 
    df_filt=df_in[df_in['attributes'].isin(trainSet)]
    if not withAugmentation:
        df_filt=df_in[df_in['augSetting']=='original']
    # no variants in original study
    df_filt=df_filt[df_filt['variantNum']=='base']

    df_test=df_filt[df_filt['set']=='test']
    df_train=df_filt[df_filt['set']=='train']

    #display(df_train)

    return df_train, df_test,labelColumnName,trainSet

def prepareHyperStudy(ran_states,optis,batches,lrates,trainSets,tt_splits=[0.2],augMentationSets=None):
    #built study argument lists for parallelization
    # TODO: Add ImageAugmentation Sets, if not fixed
    # this can be more elegant
    studyArgList=[]
    for trainSet in trainSets:
        for ranS in ran_states:
            for opt in optis:
                for batch in batches:
                    for lr in lrates:
                        for tts in tt_splits:
                            studyArgList.append((trainSet,ranS,opt,batch,lr,tts))
    return studyArgList



def trainExpSetting(selectedModel,input_shape,numEpochs,trainX,trainY,valX,valY,lb,pathToTrained,saveFirstModel,
    trainSet,ranS,opt,batch,lr,tts=0.2,dataAugSetting='original',appName=''
    ):
    # Give the model a unique name
    dataSetting=appName
    
    print('[Info] Training Model with ranState',ranS, 'and Model', selectedModel.name)

    #- Set intitial model Architecture
    modelArc=selectedModel

    # - Build and Compile Model
    model = pcm.buildCompileModel(modelArc,input_shape, dataSetting,lb,l_rate=lr,opt=opt)
    # - Train the model
    err_count=0
    try:
        model, hist_df_Test = pcm.trainModel(model,trainX,trainY,batch,valX,valY,numEpochs,pathToTrained,saveModel=saveFirstModel,
            saveHistory=False,nameAppendix=trainSet)
    except:
        err_count+=1
        if err_count >2:
            print(" [Error during Training ... to many failed attempts]")
            return 0
        else:
            print(" [Error during Training ... restart]")
            model, hist_df_Test = pcm.trainModel(model,trainX,trainY,batch,valX,valY,numEpochs,pathToTrained,saveModel=saveFirstModel,
                saveHistory=False,nameAppendix=trainSet)

    print('[Info] -- Finished -- Training Model with ranState',ranS, 'and Model', selectedModel.name)
    hist_df = pd.DataFrame(hist_df_Test) 

    # - once the first model is saved, opt out saving the rest to save disk memory, comment out if everything should be stored
    if saveFirstModel:
        saveFirstModel=False
    hist_df['ran_state']=ranS
    hist_df['trainSet']='-'.join(trainSet)
    hist_df['model']=selectedModel.name
    hist_df['optimizer']=opt
    hist_df['batch_size']=batch
    hist_df['learning_rate']=lr
    hist_df['augSetting']=dataAugSetting
    hist_df['test_split']=tts
    # - save History Collection: 
    hist_csv_file = pathToTrained+selectedModel.name+appName+'_history.csv'
    if os.path.exists(hist_csv_file):
        hist_df_col= pd.read_csv(hist_csv_file) 
        hist_df_col=pd.concat([hist_df_col,hist_df], ignore_index=False)
    else:
        hist_df_col=hist_df
    # - save History Collection: 
    with open(hist_csv_file, mode='w') as f:
        hist_df_col.to_csv(f,index=False)
            
    return model

def testExpSetting(model,dataTest,labelsTest,lb,pathToTrained,
    trainSet,ranS,opt,batch,lr,tts=0.2,dataAugSetting='original',appName=''
    ):
    #print(hist_df_col)
    print('[Info] Test Model with ranState',ranS, 'and Model', model.name)
    # - Validate model and calculate precision, recall and f1-score
    preds_df= pcm.validateModel(model,dataTest,labelsTest,batch,lb,pathToTrained,savePredictionTest=False,nameAppendix=trainSet)
    print('[Info]  -- Finished -- Test Model with ranState',ranS, 'and Model', model.name)
    preds_df['ran_state']=ranS
    preds_df['trainSet']='-'.join(trainSet)
    preds_df['model']=model.name
    preds_df['optimizer']=opt
    preds_df['batch_size']=batch
    preds_df['learning_rate']=lr
    preds_df['augSetting']=dataAugSetting
    preds_df['test_split']=tts

    # - save Class Report Collection: 
    preds_csv_file = pathToTrained+model.name+appName+'_ClassReport.csv'
    if os.path.exists(preds_csv_file):
        preds_df_col=pd.read_csv(preds_csv_file)
        preds_df_col=pd.concat([preds_df_col,preds_df], ignore_index=False)
    else:
        preds_df_col=preds_df

    # - save Class Report Collection: 
    with open(preds_csv_file, mode='w') as f:
        preds_df_col.to_csv(f,index=False)
    return 1

def runExperiment(dataTrain,trainLabels,selectedModel,input_shape,numEpochs,lb,pathToTrained,saveFirstModel,
    dataTest,labelsTest,
    trainSet,ranS,opt,batch,lr,tt_split=0.2,dataAugSetting='original',appName=''
    ):
    ## -- use augmented TrainingDataSet instead!
    trainX,trainY,valX,valY= pcm.split_data(dataTrain,trainLabels,tt_split=tt_split,ran_state=ranS) 
    model = trainExpSetting(selectedModel,input_shape,numEpochs,trainX,trainY,valX,valY,lb,pathToTrained,saveFirstModel,
    trainSet,ranS,opt,batch,lr,tt_split,dataAugSetting=dataAugSetting,appName=appName)
    testExpSetting(model,dataTest,labelsTest,lb,pathToTrained,
    trainSet,ranS,opt,batch,lr,tt_split,dataAugSetting=dataAugSetting,appName=appName)
    return 1
