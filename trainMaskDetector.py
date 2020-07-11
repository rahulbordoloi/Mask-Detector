###############################################################################
#                       Train face Mask Detector Model                        #
###############################################################################

# * --------------------------------------------------------------------------*
# * AUTHOR: Rahul Bordoloi <mail@rahulbordoloi.me>                            *
# * --------------------------------------------------------------------------*
# * DATE CREATED: 11th July, 2020                                             *
# * ************************************************************************* *

# Importing Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import os

# Hide Warnings
import warnings
warnings.filterwarnings("ignore")

# Parsing Arguments passed through Command Line
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True)                           # Dataset Rel. Add.
ap.add_argument("-m", "--model", type = str, default = "maskDetector.model")  # Name of the Model
args = vars(ap.parse_args())

# Pre-Initialising HyperParameters
INIT_LR = 1e-4            # Intial Learning-Rate
EPOCHS = 20               # No. of Epochs
BS = 32                   # Batch Size

# Putting path of the Dataset Directory in Image Path
print("*** Loading Models ***")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]     # extract the class label from the filename
    image = load_img(imagePath, target_size = (224, 224))   # load the input image (224x224) and preprocess it
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)        # Putting put the labels ie w/ mask or w/o mask

data = np.array(data, dtype = "float32")    
labels = np.array(labels)

lb = LabelBinarizer()            # Binarize the Labels
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Splitting into Train and Test Sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.20, stratify = labels, random_state = 0)

# Image Augmentation
augm = ImageDataGenerator(
    rotation_range = 20,
    zoom_range = 0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range = 0.15,
    horizontal_flip = True,
    fill_mode = "nearest"
)

###############################################################################
#                             Model Training                                  #
###############################################################################

# Base Model - MobileNetV2
baseModel = MobileNetV2(weights="imagenet", include_top = False, input_tensor = Input(shape = (224,224,3)))

# Other Layers of NN
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(128,activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation = "softmax")(headModel)

# Final NN Model
model = Model(inputs = baseModel.input, outputs = headModel)

# loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# Model Compilation
print("*** Compiling our Model ***")
opt = Adam(lr = INIT_LR, decay = INIT_LR/EPOCHS) 
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

print("*** Training our Model ***")
H = model.fit(
    augm.flow(x_train, y_train, batch_size = BS),
    steps_per_epoch = len(x_train) // BS,
    validation_data = (x_test,y_test),
    validation_steps = len(x_test) // BS,
    epochs = EPOCHS
)

# Model Evaluation
print("*** Evaluting our NN ***")
predIdxs = model.predict(x_test, batch_size = BS)
predIdxs = np.argmax(predIdxs, axis = 1)   # Getting the Index of the label with largest predicted probability

print(classification_report(y_test.argmax(axis = 1), predIdxs, target_names = lb.classes_)) # Classification Report

# Saving our Trained Model
print("*** Saving Mask-Detector Model ***")
model.save(args["model"], save_format = "h5")