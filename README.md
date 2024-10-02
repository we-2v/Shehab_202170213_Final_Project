## Introduction

This project involves fine-tuning a VGG16 model using Keras to detect skin diseases such as nevus, melanoma, and seborrheic keratosis.
For dataset https://www.kaggle.com/paoloripamonti/derma-diseases

## Libraries Used

- **Keras**: Utilized for building and training the neural network. It provides easy access to the VGG16 model.
- **OpenCV**: Used for image processing tasks.
- **NumPy**: Essential for numerical computations.
- **Matplotlib**: For plotting and visualizing data.
- **Scikit-learn**: Used for evaluation metrics like confusion matrix and classification report.
- **OS and Glob**: For file and directory management.

## Data Preparation

- **ImageDataGenerator**: Used for data augmentation, which includes operations like rescaling, shifting, and flipping images to increase the diversity of the training dataset.

## Model Architecture

- **Base Model**: VGG16 pre-trained on ImageNet, with the top classification layer removed.
- **Additional Layers**: 
  - Flatten layer to convert 3D outputs to 1D.
  - Dense layers with ReLU activation for learning complex features.
  - Dropout layers to prevent overfitting.
  - Softmax layer for multi-class classification.

## Training

- **Parameters**:
  - Batch Size: 64 for training and 8 for validation.
  - Epochs: 50.
  - Learning Rate: 0.0001.
- **Optimizer**: Adam optimizer for efficient gradient descent.

## Evaluation

- **Performance Metrics**: Training and validation accuracy and loss are plotted to evaluate the model s performance.
- **Confusion Matrix**: Provides insights into model predictions versus actual labels.
- **Classification Report**: Detailed precision, recall, and F1-score for each class.

## Results

- The model achieves a certain level of accuracy and loss on the test dataset, indicating its performance.
- Confusion matrix and classification report provide deeper insights into specific class predictions.

## Random Tests

- The model is tested on random images from the test set to visualize predictions and their confidence levels.

## Conclusion

This project demonstrates an effective use of transfer learning for dermatological image classification, leveraging pre-trained models to achieve substantial accuracy.

# Code Explanation for Derma Diseases Detection

## Import Libraries

```python
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import optimizers
from keras.applications import VGG16
import cv2
import os
import numpy as np
import itertools
import random
from collections import Counter
from glob import iglob
import warnings
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
%matplotlib inline
```

- **Keras Libraries**: Import various Keras modules for building and training the model.
- **OpenCV**: For image processing.
- **OS, NumPy, itertools, random, Counter, iglob**: Utilities for file handling and computations.
- **Warnings**: To ignore unwanted warnings during execution.
- **Scikit-learn**: For evaluation metrics.
- **Matplotlib**: For plotting graphs.

## Settings

```python
BASE_DATASET_FOLDER = os.path.join("..","input","derma_disease_dataset","dataset")
TRAIN_FOLDER = "train"
VALIDATION_FOLDER = "validation"
TEST_FOLDER = "test"

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)

TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.0001
MODEL_PATH = os.path.join("derma_diseases_detection.h5")
```

- **Dataset Paths**: Set paths for train, validation, and test folders.
- **Image Settings**: Define image size and input shape for the model.
- **Training Parameters**: Batch size, epochs, learning rate, and model path are configured.

## Data Augmentation

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode= nearest )

train_generator = train_datagen.flow_from_directory(
    os.path.join(BASE_DATASET_FOLDER, TRAIN_FOLDER),
    target_size=IMAGE_SIZE,
    batch_size=TRAIN_BATCH_SIZE,
    class_mode= categorical , 
    shuffle=True)
```

- **ImageDataGenerator**: Augments images by rescaling, shifting, and flipping.
- **train_generator**: Loads images in batches with augmentation.

## Plot Dataset Description

```python
def percentage_value(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def plot_dataset_description(path, title):
    classes = []
    for filename in iglob(os.path.join(path, "**","*.jpg")):
        classes.append(os.path.split(os.path.split(filename)[0])[-1])

    classes_cnt = Counter(classes)
    values = list(classes_cnt.values())
    labels = list(classes_cnt.keys())

    plt.figure(figsize=(8,8))
    plt.pie(values, labels=labels, autopct=lambda pct: percentage_value(pct, values), 
            shadow=True, startangle=140)

    plt.title(title)    
    plt.show()
```

- **percentage_value**: Calculates the percentage value for pie chart labels.
- **plot_dataset_description**: Plots a pie chart showing the distribution of classes.

## Load VGG16 Model

```python
vgg_model = VGG16(weights= imagenet , include_top=False, input_shape=INPUT_SHAPE)
```

- **VGG16**: Loads the pre-trained VGG16 model without the top layer for transfer learning.

## Freeze Layers

```python
for layer in vgg_model.layers[:-4]:
    layer.trainable = False
```

- **Layer Freezing**: Freezes all layers except the last four to retain learned features.

## Create Model

```python
model = Sequential()
model.add(vgg_model)
model.add(Flatten())
model.add(Dense(256, activation= relu ))
model.add(Dropout(0.2))
model.add(Dense(128, activation= relu ))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation= softmax ))
```

- **Sequential Model**: Builds a new model by adding layers on top of VGG16.
- **Dense & Dropout Layers**: Add fully connected layers with dropout for regularization.

## Compile Model

```python
model.compile(loss= categorical_crossentropy ,
              optimizer=optimizers.Adam(lr=LEARNING_RATE),
              metrics=[ acc ])
```

- **Compile**: Configures the model with loss function and optimizer.

## Train Model

```python
history = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples//train_generator.batch_size,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples//val_generator.batch_size)
```

- **fit_generator**: Trains the model using the training data generator.

## Save Model

```python
model.save(MODEL_PATH)
```

- **Save**: Saves the trained model to a file.

## Evaluate Model

```python
loss, accuracy = model.evaluate_generator(test_generator,steps=test_generator.samples//test_generator.batch_size)
```

- **Evaluate**: Tests the model on the test data to get loss and accuracy.

## Confusion Matrix

```python
Y_pred = model.predict_generator(test_generator,verbose=1, steps=test_generator.samples//test_generator.batch_size)
y_pred = np.argmax(Y_pred, axis=1)
cnf_matrix = confusion_matrix(test_generator.classes, y_pred)
```

- **Predict**: Generates predictions on the test data.
- **Confusion Matrix**: Computes confusion matrix to evaluate prediction performance.

## Plot Confusion Matrix

```python
def plot_confusion_matrix(cm, classes,
                          title= Confusion matrix ,
                          cmap=plt.cm.Blues):
    cm = cm.astype( float ) / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12,12))
    plt.imshow(cm, interpolation= nearest , cmap=cmap)
    plt.title(title, fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=12)
    fmt =  .2f 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel( True label , fontsize=16)
    plt.xlabel( Predicted label , fontsize=16)
plot_confusion_matrix(cnf_matrix, list(classes.values()))
```

- **Plot**: Visualizes the confusion matrix with true and predicted labels.

## Classification Report

```python
print(classification_report(test_generator.classes, y_pred, target_names=list(classes.values())))
```

- **Report**: Prints precision, recall, and F1-score for each class.

## Random Tests

```python
def load_image(filename):
    img = cv2.imread(os.path.join(BASE_DATASET_FOLDER, TEST_FOLDER, filename))
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
    img = img / 255
    return img

def predict(image):
    probabilities = model.predict(np.asarray([img]))[0]
    class_idx = np.argmax(probabilities)
    return {classes[class_idx]: probabilities[class_idx]}
```

- **load_image**: Loads and preprocesses an image for prediction.
- **predict**: Predicts the class of a given image.
