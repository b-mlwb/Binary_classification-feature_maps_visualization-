from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

train_model = VGG16(weights= 'imagenet', include_top = False)

data_gen = ImageDataGenerator(rescale = 1./255)
batchsize = 20

def extract_features(directory , samples):

  features_batch = np.zeros((samples , 4,4 , 512))
  labels_batch = np.zeros((samples))
  datagen = data_gen.flow_from_directory(directory , batch_size=batchsize , target_size=(150,150) , class_mode='binary')
  i = 0
  for inputs , labels in datagen:
    features = train_model.predict(inputs)
    features_batch[i*batchsize : (i+1) * batchsize] = features
    labels_batch[i*batchsize : (i+1) * batchsize] = labels
    i += 1
    if (i*batchsize) >= samples:
      break

  return features_batch , labels_batch
  
training_features , training_labels = extract_features(train_dir , 2000) #make a train_dir containing training samples
validating_features , validating_labels = extract_features(validation_dir , 1000) #make a validation_dir containing validation samples
testing_features , testing_labels = extract_features(test_dir , 1000) #make a test_dir containing test samples

training_features = np.reshape(training_features , (2000 , 4*4*512))
validating_features = np.reshape(validating_features , (1000 , 4*4*512))
testing_features = np.reshape(testing_features , (1000 , 4*4*512))

train_classfier = models.Sequential()
train_classfier.add(layers.Dense(512 , activation="relu" , input_dim=(4*4*512)))
train_classfier.add(layers.Dropout(0.4))
train_classfier.add(layers.Dense(1 , activation='sigmoid'))

train_classfier.compile(loss = 'binary_crossentropy' , optimizer = optimizers.RMSprop(lr = 2e-3) , metrics = ['acc'])

history = train_classfier.fit(training_features , training_labels , epochs = 45 , batch_size = 20 , validation_data=(validating_features , validating_labels))

loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(1 , len(acc) + 1)

plt.plot(epochs , loss , 'ro', label='training_loss')
plt.plot(epochs , val_loss , 'b' , label='validation_loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure()

plt.plot(epochs , acc, 'ro' , label='training_accuracy')
plt.plot(epochs , val_acc, 'b' , label= 'validation_accuracy')
plt.title('Training and validation accuracy')
plt.legend()
