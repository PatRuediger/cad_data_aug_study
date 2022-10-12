
import tensorflow.keras.applications as tf_app
import os
from tensorflow.keras.optimizers import SGD,Adam,RMSprop




##########################################################################
################## Global Config Params ###################################


expBlock='preTrained'

# construct the argument parser and parse the arguments

#!! USE THE CSV CONTAINING THE INTENDED AUGMENTATED IMAGES
datasetCSV='/home/ruediger/scratch_link/augmentation_exp/data_completeAugStudy2022_rs54.csv'
#IF NEEDED SET the name - default is 'original'
transformSettingName='AugStudy2022_rs54'

if not os.path.exists(datasetCSV):
    print('Data CSV File not found:',datasetCSV)
dataFilesPrefix='/home/ruediger/scratch_link/augmentation_exp/'
if not os.path.exists(dataFilesPrefix):
    print('Data CSV File not found:',dataFilesPrefix)
    
#Most CNN are designed for 224x224
input_shape=(224,224)



# ### Setup list of Architectures for Image Classification


mkeys = [
    'resnet50v2', 'resnet101v2', 'resnet152v2', 'vgg16', 'vgg19', 
    #    'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb4', 
    #    'densenet169', 'densenet201',
    #      'inception_v3', 'xception', 'inception_resnet_v2', 'NASNet'
         ]

mvalues=[
    tf_app.resnet_v2.ResNet50V2,    tf_app.resnet_v2.ResNet101V2,    tf_app.resnet_v2.ResNet152V2,
    tf_app.vgg16.VGG16,tf_app.vgg19.VGG19,
    # tf_app.efficientnet.EfficientNetB0,    tf_app.efficientnet.EfficientNetB1,
    # tf_app.efficientnet.EfficientNetB2,    tf_app.efficientnet.EfficientNetB4,
    # tf_app.densenet.DenseNet169,    tf_app.densenet.DenseNet201,
    # tf_app.inception_v3.InceptionV3,
    # tf_app.xception.Xception,
    # tf_app.inception_resnet_v2.InceptionResNetV2,
    # tf_app.nasnet.NASNetLarge
]

##########################################################################
################## Study Config Params ###################################

# - Path to store trained models and results
pathToTrained='/home/ruediger/scratch_link/class_exp/trainedModels/AugmentationModels/'
# - wheter to store the fully trained model (only the first random state model is saved)
saveFirstModel=False

# - List of random states for the train/validation split. Value range from 0 to 99
ran_states=[42,38,16,72,3]
#ran_states=[42]
#batch_size
#batches=[16,32,64]
batches=[32]
# stream data - if memory is not enough
streamData=False

#Define the number of training epochs
numEpochs=25

#Define optimizers, make sure to import them beforehand
optis=[Adam,SGD,RMSprop]

#Define Learning rate
lrates=[2.56e-1,1e-1,1e-2,4.5e-2,5e-3]

#Wether to use the augmented Images or not
useAugmentation=True

trainSets=[['without_shaders']]
#Name Appendix for history savings:
appName='_baseLine'

dataAugSetting='wAugmentation'
##########################################################################