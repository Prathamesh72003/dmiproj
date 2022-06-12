from keras.models import load_model
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np

IMAGE_SIZE = [224, 224]

train_path = 'Datasets/Train'

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

# useful for getting number of classes
folders = glob('Datasets/Train/*')


# our layers - you can add more if you want
x = Flatten()(vgg.output)
# add a fully-connected layer
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.3)

training_set = train_datagen.flow_from_directory('Datasets/Train',target_size=(224, 224),batch_size=32,class_mode='categorical',subset='training')

label_map = (training_set.class_indices)

test_set = train_datagen.flow_from_directory('Datasets/Train',target_size=(224, 224),batch_size=32,class_mode='categorical',subset='validation')

# fit the model
r = model.fit_generator(training_set, validation_data=test_set, epochs=10, steps_per_epoch=len(training_set), validation_steps=len(test_set))

# image = tf.keras.preprocessing.image.load_img("Datasets/Test/akshay4.jpg", target_size=(224, 224))
image = tf.keras.preprocessing.image.load_img("srk.jpg", target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
input_arr = input_arr.astype('float32') / 255

predictions = model.predict(input_arr)

pre_class=predictions.argmax()

print(label_map)
print(pre_class)

model.save('recognition.h5')
