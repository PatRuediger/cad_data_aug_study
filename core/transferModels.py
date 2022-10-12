from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

from tensorflow.keras.layers import AveragePooling2D
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD,Adam
import tensorflow.keras.applications as tf_app

import class_prep as pcm

import math

## Add more if needed - See tensorflow.keras.applications - make sure to check if preprocessing is 
# needed and handlede in class_prep.preprocessNeeded()    
modelArcs={
    'efficientnetb0': tf_app.EfficientNetB0,
    'efficientnetb1': tf_app.EfficientNetB1,
    'efficientnetb2': tf_app.EfficientNetB2,
    'efficientnetb3': tf_app.EfficientNetB3,
    'efficientnetb4': tf_app.EfficientNetB4,
    'efficientnetb5': tf_app.EfficientNetB5, 
    'efficientnetb6': tf_app.EfficientNetB6, 
    'efficientnetb7': tf_app.EfficientNetB7,      

    # -- Preprocessing for those needed if TF version is < 2.5

    'densenet169': tf_app.DenseNet169,
    'densenet201':tf_app.DenseNet201,

    'resnet50v2': tf_app.ResNet50V2,
    'resnet101v2': tf_app.ResNet101V2,
    'resnet152v2': tf_app.ResNet152V2,

    'xception': tf_app.Xception
}

##############################################
#- Transfer Learning Top Models
##############################################

def DoubleDense(modelName,input_shape,num_classes,dense_neurons=None,gap=False,dropout=True,dropout_rate=0.5):
    '''
    Double Dense Layers:
    Top Layer is removed and two dense layers are added. One for the linear "feature" mapping of 
    the "learned" convolutions and a new "last layer" to map to the respective classes.
    In the standard approach the first dense layers has the same number of neurons as elements in the input shape
            Parameters:
                    modelName: name of the model archtitecture to be used. See tensorflow.keras.applications
                    input_shape: tuple determining the input size
                    num_classes: number of classes in the training data set
                    dense_neurons: number of neurons in additional layer, defaults to input_shape*2
                    gap: If GlobalAveragePooling is used instead of Flatten
                    dropout: If a dropout layer is used
                    dropout_rate: drop_out rate for the dropout layer (between 0.0 and 1.0)
            Returns:
                    Transfer Model and baseModel for furhter processing
    '''
    baseModel = modelArcs[modelName](weights="imagenet", include_top=False,
        input_tensor=Input(shape=(input_shape[0],input_shape[1], 3)))

    # Freeze base model
    baseModel.trainable=False
    x = baseModel.input
    x = baseModel(x,training=False)
    if gap:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        x = Flatten(name="flatten")(x)
    if dense_neurons is None:
        # For every pixel in the input data there is one neuron in the dense layer
        x = Dense(input_shape[0]*2, activation="relu")(x)
    else:
        x = Dense(dense_neurons, activation="relu")(x)
    # standard value is 0.5, could make sense to lower this value
    if dropout:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    modelOut=Model(inputs=baseModel.input,outputs=outputs)

    return modelOut,baseModel   

def OneLayerPush(modelName,input_shape,num_classes,gap=False,dropout=True,dropout_rate=0.5):
    '''
    One Layer Push Through:
    Top Layer is removed and a new "last layer" to map to the respective classes.
    This directly maps the initially learned features from the pretrained net to the new classes.
            Parameters:
                    modelName: name of the model archtitecture to be used. See tensorflow.keras.applications
                    input_shape: tuple determining the input size
                    num_classes: number of classes in the training data set
                    gap: If GlobalAveragePooling is used instead of Flatten
                    dropout: If a dropout layer is used
                    dropout_rate: drop_out rate for the dropout layer (between 0.0 and 1.0)
            Returns:
                    Transfer Model and baseModel for furhter processing
    '''
    baseModel = modelArcs[modelName](weights="imagenet", include_top=False,
        input_tensor=Input(shape=(input_shape[0],input_shape[1], 3)))

    # Freeze base model
    baseModel.trainable=False
    x = baseModel.input
    x = baseModel(x,training=False)
    if gap:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    else:
        x = Flatten(name="flatten")(x)
    # standard value is 0.5, could make sense to lower this value
    if dropout:
        x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    modelOut=Model(inputs=baseModel.input,outputs=outputs)

    return modelOut,baseModel 

##############################################
#- Transfer Learning Fine Tuning
##############################################

def fineTuneBase(baseModel,model,start_layer=None):
    '''
    Run Fine tuning model by reactivating parts of the base model as trainabel.
    This could lead the worser models, if the new classes are not similar to any of the initial training classes.

            Parameters:
                    baseModel: The baseModel used before, should be the object not the name!
                    model: Trained model which should be fine tuned
                    start_layer: If value is between 0.0 and 1.0 relative number of total layers
                                in the baseModel of wich to keep. If greater than 1, the absolut
                                number of layers to retrain is assumed.
            Returns:
                    fine_tuned Model and baseModel for furhter processing
    '''
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(baseModel.layers))
    
    # Fine-tune from this layer onwards
    if start_layer is None:
        # Standard value is slightly more than half of the layers.
        # For deep networks this value should be higher!
        fine_tune_at=math.floor(len(baseModel.layers)*0.65)
    elif start_layer<= 1.0:
        # Assume relative size to len of baseModel Layers is used
        fine_tune_at=math.floor(len(baseModel.layers)*start_layer)
    fine_tune_at = start_layer

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in baseModel.layers[:fine_tune_at]:
        layer.trainable = False
    
    return model,baseModel