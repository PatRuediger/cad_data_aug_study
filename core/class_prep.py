#!/usr/bin/env python
# coding: utf-8

# # Few Shot - Zero Shot Deep Learning Part Classifier

from pathlib import Path
from readline import append_history_file
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.layers import AveragePooling2D
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
#from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow.keras.applications as tf_app
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from imutils import paths
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
#from tqdm.keras import TqdmCallback
#import datetime
#import PIL
import copy
import os



## KeyWords for Model Architectures

# global EFFNETB0 = tf_app.EfficientNetB0
# global EFFNETB1 = tf_app.EfficientNetB1
# global EFFNETB2 = tf_app.EfficientNetB2
# global EFFNETB3 = tf_app.EfficientNetB3
# global EFFNETB4 = tf_app.EfficientNetB4
# global EFFNETB5 = tf_app.EfficientNetB5
# global EFFNETB6 = tf_app.EfficientNetB6
# global EFFNETB7 = tf_app.EfficientNetB7

# global DENSE169 = tf.app.DenseNet169
# global DENSE201 = tf.app.DenseNet201

# global RESNET50 = tf.app.ResNet50V2
# global RESNET101= tf.app.ResNet101V2
# global RESNET152= tf.app.ResNet152V2

# global XCEPNET  = tf.app.Xception

# ## Some models need special preprocessing
# global modelsPreprocessTF={RESNET50,RESNET101,RESNET152,XCEPNET}
# global modelsPreprocessTorch={DENSE169,DENSE201}


def loadData(datasetCSV,dataFilesPrefix,input_shape,trainAtt,tt_split=0.2,ran_state=42,subset=1.0,labelColumnName='category_2',
                streaming=False,s_batch=8,onlyFileNames=False,fileHandlerMode='dmu-exp',df_filtered=None):
    # load training data
    if onlyFileNames:
        df_train,trainFiles = loadTrainDataFileNamesOnly(datasetCSV,dataFilesPrefix,input_shape,trainAtt,labelColumnName=labelColumnName,
                        subset=subset,fileHandlerMode=fileHandlerMode,df_filtered=df_filtered)
        df_test,testFiles= loadTestDataFileNamesOnly(datasetCSV,dataFilesPrefix,input_shape,trainAtt,labelColumnName=labelColumnName,
                        subset=subset,fileHandlerMode=fileHandlerMode,df_filtered=df_filtered)
    elif streaming:
        print('####-- Streaming DATA ... ')
        train_generator,vali_generator,labels,lb,df_train=loadTrainDataStreaming(datasetCSV,dataFilesPrefix,input_shape,trainAtt,
                                       tt_split=tt_split,labelColumnName=labelColumnName,s_batch=s_batch,
                                       fileHandlerMode=fileHandlerMode,df_filtered=df_filtered)
        test_generator,labelsTest,df_test,lbT=loadTestDataStreaming(datasetCSV,dataFilesPrefix,input_shape,labelColumnName=labelColumnName,
                                        s_batch=s_batch,fileHandlerMode=fileHandlerMode,df_filtered=df_filtered)
    else:
        dataTrain,labels,lb,df_train=loadTrainData(datasetCSV,dataFilesPrefix,input_shape,trainAtt,labelColumnName=labelColumnName,
                                        subset=subset,fileHandlerMode=fileHandlerMode,df_filtered=df_filtered)
        # split train data
        trainX,trainY,testX,testY=split_data(dataTrain,labels,tt_split,ran_state)
        # load test data
        dataVal,labelsTest,df_test,lbT=loadTestData(datasetCSV,dataFilesPrefix,input_shape,labelColumnName=labelColumnName,
                                    subset=subset,fileHandlerMode=fileHandlerMode,df_filtered=df_filtered)
    if onlyFileNames:
        return df_train,trainFiles,df_test,testFiles
    elif streaming:
        print('[INFO] Train data is streamed')
        return train_generator,vali_generator,lb,labels,test_generator,labelsTest,df_train,df_test
    else:
        return trainX,trainY,testX,testY,lb,dataTrain,labels,dataVal,labelsTest,df_train,df_test

def loadTrainDataStreaming(datasetCSV,dataFilesPrefix,input_shape,trainAtt,tt_split=0.2,labelColumnName='category_2',s_batch=1,fileHandlerMode='dmu-exp',df_filtered=None):
    print("[INFO] loading Training images as data stream...")
    if df_filtered is None:
        df=pd.read_csv(datasetCSV,sep=';')
        # -- Select which one to use
        df_train=df[df['set']=='train']
        df_train=df_train[df_train['attributes'].isin(trainAtt)]
    # - Use the prefiltered dataframe instead
    else:
        df_train=df_filtered.copy()

    #load and transform train data
    datagen=ImageDataGenerator(validation_split=tt_split)
    filenames=[]
    # Choose proper Filename Handling - Add function if needed
    fnH=fileNameHandler(mode=fileHandlerMode)
    for i in tqdm(range(len(df_train))):
        filename=fnH.filenameHandlerDMU(dataFilesPrefix,input_shape,df_train,i)
        filenames.append(filename)
    df_train['res_filename']=filenames
    train_generator=datagen.flow_from_dataframe(dataframe=df_train, directory="",
                                        x_col='res_filename', y_col=labelColumnName, class_mode="categorical", 
                                        target_size=input_shape,subset='training', batch_size=s_batch)
    vali_generator=datagen.flow_from_dataframe(dataframe=df_train, directory="",
                                            x_col='res_filename', y_col=labelColumnName, class_mode="categorical", 
                                            target_size=input_shape,subset='validation', batch_size=s_batch)
    
    ##--- convert the data and labels to NumPy arrays
    labelsIn=list(df_train.category_2)   
    labelsIn = np.array(labelsIn)
    ##--- perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labelsIn)
    return train_generator,vali_generator,labels,lb,df_train

def loadTestDataStreaming(datasetCSV,dataFilesPrefix,input_shape,labelColumnName='category_2',s_batch=1,fileHandlerMode='dmu-exp',df_filtered=None):
     ##--- load and transform test data
    print("[INFO] loading Test images as data stream...")
    if df_filtered is None:
        df=pd.read_csv(datasetCSV,sep=';')
        # -- Select which one to use
        df_test=df[df['set']=='test']
        df_test=df_test[df_test['depth_normal']=='-']
    # - Use the prefiltered dataframe instead
    else:
        df_test=df_filtered.copy()
    labelsTest=list(df_test[labelColumnName])
    dataVal=[]
     #load and transform train data
    datagen=ImageDataGenerator()
    filenames=[]
    # Choose proper Filename Handling - Add function if needed
    fnH=fileNameHandler(mode=fileHandlerMode)
    for i in tqdm(range(len(df_test))):
        filename=fnH.filenameHandlerDMU(dataFilesPrefix,input_shape,df_test,i)
        filenames.append(filename)
    df_test['res_filename']=filenames
    test_generator=datagen.flow_from_dataframe(dataframe=df_test, directory="",
                                        x_col='res_filename', y_col=labelColumnName, class_mode="categorical", 
                                        target_size=input_shape,subset='training', batch_size=s_batch)
    
    #########
    #for the test data
    labelsInT=list(df_test.category_2)
    dataVal = np.array(dataVal)
    labelsInT = np.array(labelsInT)
    # perform one-hot encoding on the labels
    lbT = LabelBinarizer()
    labelsTest = lbT.fit_transform(labelsInT)
    #########

    return test_generator,labelsTest,df_test,lbT



def loadTrainData(datasetCSV,dataFilesPrefix,input_shape,trainAtt=[],
                    labelColumnName='category_2',modelName='',subset=1.0,fileHandlerMode='dmu-exp',df_filtered=None):
    # initialize the set of labels from the dataset we are
    # going to train our network on
    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading Training images...")
    if df_filtered is None:
        df=pd.read_csv(datasetCSV,sep=';')
        # -- Select which one to use
        df_train=df[df['set']=='train']
        if subset < 1.0:
            dropCols=int(subset*len(df_train))
            df_train=df_train.iloc[-dropCols:]
        df_train=df_train[df_train['attributes'].isin(trainAtt)]
    # - Use the prefiltered dataframe instead
    else:
        print("[INFO] use prefiltered DF...",len(df_filtered))
        df_train=df_filtered.copy()

    # -- load and transform train data
    # -- look for the proper labels column
    labels=list(df_train[labelColumnName])
    dataTrain=[]
    
    fnH=fileNameHandler(mode=fileHandlerMode)
    for i in tqdm(range(len(df_train))):
        # load the image, convert it to RGB channel ordering, and resize
        # it to be specified input_shape, ignoring aspect ratio

        # -- Check if desired resolution already exists:
        filename=fnH.filenameHandlerDMU(dataFilesPrefix,input_shape,df_train,i)
        # -- TF Version --
        file_content = tf.io.read_file(filename)
        # Read JPEG or PNG  image from file
        reshaped_image = tf.io.decode_image(file_content)
        #drop the alpha channel if 4 channel image
        reshaped_image=reshaped_image[:,:,0:3]
        #convert to float (0,1)
        #reshaped_image=reshaped_image/255.0
        dataTrain.append(reshaped_image)
        

    #########
    ##--- convert the data and labels to NumPy arrays
    labelsIn=list(df_train[labelColumnName])   
    dataTrain = np.array(dataTrain)
    labelsIn = np.array(labelsIn)
    ##--- perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labelsIn)

    #Check for preprocessing
    dataTrain= preprocessNeeded(modelName,dataTrain)

    return dataTrain,labels,lb,df_train

def loadTrainDataFileNamesOnly(datasetCSV,dataFilesPrefix,input_shape,trainAtt,labelColumnName='category_2',modelName='',subset=1.0,fileHandlerMode='dmu-exp',df_filtered=None):
    # initialize the set of labels from the dataset we are
    # going to train our network on
    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading Training image File paths...")
    if df_filtered is None:
        df=pd.read_csv(datasetCSV,sep=';')
        # -- Select which one to use
        df_train=df[df['set']=='train']
        if subset < 1.0:
            dropCols=int(subset*len(df_train))
            df_train=df_train.iloc[-dropCols:]
        df_train=df_train[df_train['attributes'].isin(trainAtt)]
    # - Use the prefiltered dataframe instead
    else:
        print("[INFO] use prefiltered DF...",len(df_filtered))
        df_train=df_filtered.copy()

    
    # -- load and transform train data
    # -- look for the proper labels column
    trainFilenames=[]
    fnH=fileNameHandler(mode=fileHandlerMode)
    for i in tqdm(range(len(df_train))):
        # load the image, convert it to RGB channel ordering, and resize
        # it to be specified input_shape, ignoring aspect ratio

        # -- Check if desired resolution already exists:
        filename=fnH.filenameHandlerDMU(dataFilesPrefix,input_shape,df_train,i)
        if os.path.exists(filename):
            trainFilenames.append(filename)
        #- otherwise load and resize
        else:
            print(" [Error] File does not exist,",filename)
            return 0
    df_train['filepaths']=trainFilenames
    return df_train,trainFilenames

def loadTestDataFileNamesOnly(datasetCSV,dataFilesPrefix,input_shape,trainAtt,labelColumnName='category_2',modelName='',subset=1.0,fileHandlerMode='dmu-exp',df_filtered=None):
    # initialize the set of labels from the dataset we are
    # going to train our network on
    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading Test image Filepaths ...")
    if df_filtered is None:
        df=pd.read_csv(datasetCSV,sep=';')
        # -- Select which one to use
        df_test=df[df['set']=='test']
        #df_test=df_test[df_test['depth_normal']=='-']
        if subset < 1.0:
            dropCols=int(subset*len(df_test))
            df_test=df_test.iloc[-dropCols:]
    # - Use the prefiltered dataframe instead
    else:
        df_test=df_filtered.copy()
    
    
    # -- load and transform train data
    # -- look for the proper labels column
    trainFilenames=[]
    fnH=fileNameHandler(mode=fileHandlerMode)
    for i in tqdm(range(len(df_test))):
        # load the image, convert it to RGB channel ordering, and resize
        # it to be specified input_shape, ignoring aspect ratio

        # -- Check if desired resolution already exists:
        filename=fnH.filenameHandlerDMU(dataFilesPrefix,input_shape,df_test,i)
        if os.path.exists(filename):
            trainFilenames.append(filename)
        #- otherwise load and resize
        else:
            print(" [Error] File does not exist,",filename)
            return 0
    df_test['filepaths']=trainFilenames
    return df_test,trainFilenames

def loadTestData(datasetCSV,dataFilesPrefix,input_shape,modelName='',labelColumnName='category_2',subset=1.0,fileHandlerMode='dmu-exp',df_filtered=None):
     ##--- load and transform test data
    print("[INFO] loading Test images...")
    if df_filtered is None:
        df=pd.read_csv(datasetCSV,sep=';')
        # -- Select which one to use
        df_test=df[df['set']=='test']
        df_test=df_test[df_test['depth_normal']=='-']
        if subset < 1.0:
            dropCols=int(subset*len(df_test))
            df_test=df_test.iloc[-dropCols:]
    # - Use the prefiltered dataframe instead
    else:
        df_test=df_filtered.copy()
    dataVal=[]
    fnH=fileNameHandler(mode=fileHandlerMode)
    for i in tqdm(range(len(df_test))):
        # load the image, convert it to RGB channel ordering, and resize
        # it to be specified input_shape, ignoring aspect ratio

        # -- Check if desired resolution already exists:
        filename=fnH.filenameHandlerDMU(dataFilesPrefix,input_shape,df_test,i)

        # -- TF Version --
        file_content = tf.io.read_file(filename)
        # Read JPEG or PNG  image from file
        reshaped_image = tf.io.decode_image(file_content)
        #drop the alpha channel if 4 channel image
        reshaped_image=reshaped_image[:,:,0:3]
        #convert to float (0,1)
        #reshaped_image=reshaped_image/255.0
        dataVal.append(reshaped_image)

    #########
    #for the test data
    labelsInT=list(df_test[labelColumnName])
    dataVal = np.array(dataVal)
    labelsInT = np.array(labelsInT)
    # perform one-hot encoding on the labels
    lbT = LabelBinarizer()
    labelsTest = lbT.fit_transform(labelsInT)
    #########

    #Check for preprocessing
    dataVal= preprocessNeeded(modelName,dataVal)

    return dataVal,labelsTest,df_test,lbT

def split_data(dataTr,labelsTr,tt_split=0.2,ran_state=42):

    (trainX, testX, trainY, testY) = train_test_split(dataTr, labelsTr,
        test_size=tt_split, stratify=labelsTr, random_state=ran_state)

    print(trainX.shape,trainY.shape)
    return trainX,trainY,testX,testY

def preProcessData(trainDataX,mode='tf'):
    # -- it is always the same for every model architecture. direct usage of image util function leads to errors.
    if mode=='tf':
        return tf_app.resnet_v2.preprocess_input(trainDataX)
    if mode =='torch':
        return tf_app.densenet.preprocess_input(trainDataX)  
    if mode =='inception_res':
        return tf_app.inception_v3.preprocess_input(trainDataX)
    if mode =='inception':
        return tf_app.inception_resnet_v2.preprocess_input(trainDataX)  



def buildCompileModel(modelA,input_shape, dataSetting,lb,l_rate=1e-3,opt=Adam):
    # - Select an optimizer -- ResNet works best with SGD, EfficientNet with Adam
    opt=opt(learning_rate=l_rate)

    #This is only for the naming
    modelName=modelA.name+dataSetting
    #architecture - for rgb input images 3- channels are used
    modelArc=modelA
    model=classyTransferModel(modelArc,input_shape,len(lb.classes_))
    model.summary()

    print("[INFO] compiling model...")
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])
    model._name = modelName
    return model

def trainModel(model,trainXm,trainY,batchSize,testXm,testY,numEpochs,pathToTrained,callbacks=[],saveModel=True,
    saveHistory=True,nameAppendix='',streaming=False,train_generator=None,vali_generator=None,augmentImages=False):
    if streaming:
        STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
        STEP_SIZE_VALID=vali_generator.n//vali_generator.batch_size
        H = model.fit(
                train_generator,
                steps_per_epoch=STEP_SIZE_TRAIN,
                validation_data=vali_generator,
                validation_steps=STEP_SIZE_VALID,   
            #validation_data=valAug.flow(testX, testY),
            epochs=numEpochs,
            callbacks=callbacks)
    if augmentImages:
        STEP_SIZE_TRAIN=len(trainXm)//batchSize
        H= model.fit(train_generator,
                steps_per_epoch=STEP_SIZE_TRAIN,
                validation_data=(testXm,testY),
                validation_steps=len(testXm) // batchSize,
                epochs=numEpochs,
                callbacks=callbacks 
        )
    else:
        H = model.fit(

                x=trainXm, y=trainY, batch_size=batchSize,
                steps_per_epoch=len(trainXm) // batchSize,
                validation_data=(testXm,testY),
                validation_steps=len(testXm) // batchSize,
                epochs=numEpochs,
                callbacks=callbacks
                )
    # ##- Save the model
    if saveModel:
        model.save(pathToTrained+model.name+nameAppendix)
        print("Saved model to disk")
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(H.history) 
    if saveHistory:
        # save to csv: 
        hist_csv_file = pathToTrained+model.name+'_'+nameAppendix+'_history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

    return model,hist_df

def validateModel(model,dataValm,labelsTest,batchSize,lb,pathToTrained,savePredictionTest=True,nameAppendix=''):
    # evaluate the network
    print("[INFO] evaluating network...")

    predictions = model.predict(x=dataValm, batch_size=batchSize)

    c_report=classification_report(labelsTest.argmax(axis=1),
        predictions.argmax(axis=1), target_names=lb.classes_,output_dict=True)
    df_class_report = pd.DataFrame.from_dict(c_report).transpose()
    df_class_report = df_class_report.reset_index()
    df_class_report.rename(columns={ df_class_report.columns[0]: "label" }, inplace = True)
    
    #drop last two columns with average values
    df_class_report=df_class_report[:-2]
    # Write the global accuracy as a columns and extract it from the last line in the table of the classification report
    df_class_report['accuracy']=list(df_class_report[-1:]['precision'])[0]
    # drop last column
    df_class_report=df_class_report[:-1]
    if savePredictionTest:
        df_class_report.to_csv(pathToTrained+model.name+'_'+nameAppendix+'_ClassReport.csv')
    return df_class_report


def preprocessNeeded(modelName,dataX):
    '''    Checks if special preprocessing is needed.
    Collection of model names, that needs preprocessing to BGR and range [-1,1], see tf.keras.applications if 
    the one you are using is missing.
    '''
    modelsPreprocessTF={'resnet50v2','resnet101v2','resnet152v2','vgg16',
            'vgg19','xception'}
    modelsPreprocessInception={'inception_v3'}
    modelsPreprocessInceptionRes={'inception_resnet_v2'}
    modelsPreprocessTorch={'densenet121','densenet169','densenet201'}

    ## -- TODO There is still some kind of overwrite bug in trainX sometimes
    if modelName in modelsPreprocessTF:
        print('PREPRROCESS INPUT- TF')
        #-- make a copy
        dataXm=copy.deepcopy(dataX)
        dataXm=preProcessData(dataXm)
        return dataXm
    if modelName in modelsPreprocessTorch:
        print('PREPRROCESS INPUT- Torch')
        #-- make a copy
        dataXm=copy.deepcopy(dataX)
        dataXm=preProcessData(dataXm,mode='torch')
        return dataXm
    if modelName in modelsPreprocessInception:
        print('PREPRROCESS INPUT- Inception')
        #-- make a copy
        dataXm=copy.deepcopy(dataX)
        dataXm=preProcessData(dataXm,mode='inception')
        return dataXm
    if modelName in modelsPreprocessInceptionRes:
        print('PREPRROCESS INPUT- InceptionResnet')
        #-- make a copy
        dataXm=copy.deepcopy(dataX)
        dataXm=preProcessData(dataXm,mode='inception_res')
        return dataXm
    else:
        print('NO PREPROCESSING NEEDED - return initial values')
        return dataX
    


def trainTestModel(modelA,input_shape, dataSetting,lb, trainX,trainY,testX,testY,numEpochs,dataVal,labelsTest,pathToTrained,tt_split=0.2,ran_state=42,safeModel=False,
                    batchSize=32,saveHistory=True,savePredictionTest=True,augmentImages=False):
    '''
    DEPRACTED - do not use anymore. Only for old code snipptes
    use buildCompileModel(), trainModel() and validateModel() instead.
    '''
    # - Select an optimizer -- ResNet works best with SGD, EfficientNet with Adam
    opt=Adam(learning_rate=1e-3)

    #This is only for the naming
    modelName=modelA.name+dataSetting+'tt_0.'+str(int(tt_split*10))+'ranS_'+str(ran_state)
    #architecture - for rgb input images 3- channels are used
    modelArc=modelA


    #augment Images?- 
    #augmentImages = False
    if augmentImages:
        datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)
        

    # - Collection of model names, that needs preprocessing to BGR and range [-1,1], see tf.keras.applications if 
    # the one you are using is missing.
    modelsPreprocessTF={'resnet50v2','resnet101v2','resnet152v2','vgg16',
            'vgg19','xception'}
    modelsPreprocessInception={'inception_v3'}
    modelsPreprocessInceptionRes={'inception_resnet_v2'}
    modelsPreprocessTorch={'densenet121','densenet169','densenet201'}


    #-- make a copy
    trainXm=copy.deepcopy(trainX)
    testXm =copy.deepcopy(testX)
    dataValm=copy.deepcopy(dataVal)

    #if modelA.name not in {'efficientnetb0','efficientnetb1','efficientnetb2','efficientnetb4'}:
    ## -- TODO There is still some kind of overwrite bug in trainX sometimes
    if modelArc.name in modelsPreprocessTF:
        print('PREPRROCESS INPUT- TF')
        trainXm=preProcessData(trainXm)
        testXm=preProcessData(testXm)
        dataValm=preProcessData(dataValm)
        print(trainXm[0].min(),trainXm[0].max())
    if modelArc.name in modelsPreprocessTorch:
        print('PREPRROCESS INPUT- Torch')
        trainXm=preProcessData(trainXm,mode='torch')
        testXm=preProcessData(testXm,mode='torch')
        dataValm=preProcessData(dataValm,mode='torch')
        print(trainXm[0].min(),trainXm[0].max())
    if modelArc.name in modelsPreprocessInception:
        print('PREPRROCESS INPUT- Inception')
        trainXm=preProcessData(trainXm,mode='inception')
        testXm=preProcessData(testXm,mode='inception')
        dataValm=preProcessData(dataValm,mode='inception')
        print(trainXm[0].min(),trainXm[0].max())
    if modelArc.name in modelsPreprocessInceptionRes:
        print('PREPRROCESS INPUT- InceptionResnet')
        trainXm=preProcessData(trainXm,mode='inception_res')
        testXm=preProcessData(testXm,mode='inception_res')
        dataValm=preProcessData(dataValm,mode='inception_res')
        print(trainXm[0].min(),trainXm[0].max())



    baseModel = modelArc

    # Freeze base model
    num_classes=len(lb.classes_)
    baseModel.trainable=False
    x = baseModel.input
    x = baseModel(x,training=False)
    x = Flatten(name="flatten")(x)
    x = Dense(input_shape[0]*2, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model=Model(inputs=baseModel.input,outputs=outputs)
    print(model.summary())


    # compile our model (this needs to be done after our setting our
    # layers to being non-trainable)
    print("[INFO] compiling model...")

    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])


    #-- Setup Tensorboard
    # log_dir = pathToTrained+modelName+'_log'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
    #             histogram_freq=5,
    #             update_freq='batch',
    #             profile_batch='1,32')


    # train the head of the network for a few epochs (all other layers
    # are frozen) -- this will allow the new FC layers to start to become
    # initialized with actual "learned" values versus pure random
    print("[INFO] training head...")

    if augmentImages:
        print('---- Using tf.ImageDataGenerator ---')
        # fits the model on batches with real-time data augmentation:
        H= model.fit(datagen.flow(trainXm, trainY, batch_size=batchSize),
                #validation_data=datagen.flow(testX, testY,batch_size=8),
                validation_data=(testXm,testY),
                steps_per_epoch=len(trainXm) // batchSize,
                epochs=numEpochs
                # callbacks=[TqdmCallback(verbose=2),
                # tensorboard_callback]
                )
                
    else:
        H = model.fit(

            x=trainXm, y=trainY, batch_size=batchSize,
            steps_per_epoch=len(trainXm) // batchSize,
            validation_data=(testXm,testY),
            validation_steps=len(testXm) // batchSize,
            #validation_data=valAug.flow(testX, testY),
            epochs=numEpochs
            # callbacks=[TqdmCallback(verbose=2),
            # tensorboard_callback]
            )



    # ##- Save the model
    if safeModel:
        model.save(pathToTrained+modelName)
        print("Saved model to disk")
    


    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(H.history) 
    if saveHistory:
        # save to csv: 
        hist_csv_file = pathToTrained+modelName+'_history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)



    # evaluate the network
    print("[INFO] evaluating network...")

    predictions = model.predict(x=dataValm, batch_size=32)

    c_report=classification_report(labelsTest.argmax(axis=1),
        predictions.argmax(axis=1), target_names=lb.classes_,output_dict=True)
    df_class_report = pd.DataFrame(c_report).transpose()
    #display(df_class_report)
    print(df_class_report)
    if savePredictionTest:
        df_class_report.to_csv(pathToTrained+modelName+'_ClassReport.csv')

    return model,hist_df,df_class_report

def testModel(pathToTrained,modelName,model,dataValm,labelsTest,lb,savePredictionTest=True):
    # evaluate the network
    print("[INFO] evaluating network...")

    predictions = model.predict(x=dataValm, batch_size=32)

    c_report=classification_report(labelsTest.argmax(axis=1),
        predictions.argmax(axis=1), target_names=lb.classes_,output_dict=True)
    df_class_report = pd.DataFrame(c_report).transpose()
    #display(df_class_report)
    print(df_class_report)
    if savePredictionTest:
        df_class_report.to_csv(pathToTrained+modelName+'_ClassReport.csv')

    return df_class_report


# ---------- Helper Functions --------------- #
def classyTransferModel(modelArc,input_shape,num_classes,maxLayerSize=380):
    '''
    classical approach for transfer learning:
    Top Layer is removed and two dense layers are added. One for the linear "feature" mapping of 
    the "learned" convolutions and a new "last layer" to map to the respective classes.
    Change the maxLayerSize if you encounter OOM errors on the GPU during training.
    '''
    baseModel = modelArc

    # Freeze base model

    baseModel.trainable=False
    x = baseModel.input
    x = baseModel(x,training=False)
    x = Flatten(name="flatten")(x)
    # TODO Check if scaling this dense layer with the input resolution is a good idea!
    if input_shape[0]>maxLayerSize:
        # - This will solve any OOM Errors on the GPU while training
        x = Dense(maxLayerSize*2, activation="relu")(x)
    else: 
        x = Dense(input_shape[0]*2, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    modelOut=Model(inputs=baseModel.input,outputs=outputs)


    return modelOut   


#- Add an early stopping to avoid overfitting
class CustomStopper(tf.keras.callbacks.EarlyStopping):
    '''
    Custom Callback which trains at least for 10 epochs before checking the early
    stop conditions.
    Minimum number of epochs can be specified with start_epoch=x
    '''
    def __init__(self, monitor='val_loss',
             min_delta=0, patience=5, verbose=0, mode='min',restore_best_weights = True, start_epoch = 10): # add argument for starting epoch
        super(CustomStopper, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

class fileNameHandler():
    '''
        Filename Handler Class, to access the repective image on th system based on the values of a dataframe
        Example Function for DMU- Example given:
    '''
    def __init__(self,mode='dmu-exp'):
        self.mode=mode

    def filenameHandlerDMU(self,dataFilesPrefix,input_shape,df,i):
            #- REIMPLEMENT FOR YOUR DATA NEEDS!
            #- Handler for DMU-Net Blender Export
            # kw_list: keyword list, in order it should be added to the filename string
        if self.mode=='dmu-exp':
            filename=dataFilesPrefix+df.iloc[i].dataset+'/'+df.iloc[i].set+'/'
            filename+=df.iloc[i].category_2+'/'+df.iloc[i].variantNum+'/'
            filename+=df.iloc[i].category_1+'/'+df.iloc[i].attributes+'/'
            filename+=str(input_shape[0])+'_'+str(input_shape[1])+'/'+df.iloc[i].filename
        elif self.mode=='dmu-exp_old':
            filename=dataFilesPrefix+'Results/'+str(input_shape[0])+'_'+str(input_shape[1])+'/'
            filename+=df.iloc[i].set+'/'+df.iloc[i].category_2+'/'+df.iloc[i].category_1+'/'+df.iloc[i].attributes
            filename+='/'+df.iloc[i].filename
        elif self.mode=='dmu-variants':
            #- Handler for DMU-Net Blender Export Variants
            # kw_list: keyword list, in order it should be added to the filename string
            filename=dataFilesPrefix+'Results/'+str(input_shape[0])+'_'+str(input_shape[1])+'/'
            filename+=df.iloc[i].set+'/'+df.iloc[i].category_2+'/'+df.iloc[i].category_1+'/'
            filename+=df.iloc[i].variantNum+'/'+df.iloc[i].attributes
            filename+='/'+df.iloc[i].filename
        elif self.mode=='dmu-var_pre':
            filename=dataFilesPrefix
            filename+=df.iloc[i].prefix
            filename+='/'+df.iloc[i].filename
        else:
            raise Exception('[Error] Wrong Filehandler for Mode:'+self.mode)
        if not os.path.exists(filename):
            raise Exception('[Error] File not found:'+filename)
        return filename
# ---------- End Helper Functions --------------- #


