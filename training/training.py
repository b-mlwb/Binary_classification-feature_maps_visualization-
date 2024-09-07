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


train_classfier.save('cats_and_dogs_scratch.h5')

from keras.saving import load_model
class_model = load_model('cats_and_dogs_scratch.h5')

from keras.preprocessing import image
import numpy as np

images = image.load_img('drive/MyDrive/cats_and_dogs/test_set/cats/1504.jpg', target_size=(150,150))
img = image.img_to_array(images)
img = np.expand_dims(img , axis=0)
img /= 255.

print(img.shape)


from keras.models import Model
import matplotlib.pyplot as plt

layers_outputs = [layer.output for layer in class_model.layers[:8]]

activation_model = Model( inputs=class_model.input , outputs=layers_outputs )

activations = activation_model.predict(img)

first_layer = activations[0]

layer_names = []
for layer in class_model.layers[:8]:
  layer_names.append(layer.name)

images_per_row = 16

for layer_name , layer_activation in zip(layer_names , activations):

  filters = layer_activation.shape[-1]
  size = layer_activation.shape[1]

  no_rows = filters // images_per_row
  display = np.zeros((size * no_rows , size * images_per_row))

  for row in range(no_rows):
    for col in range(images_per_row):
      channel_image = layer_activation[0,:,:, row * images_per_row + col]
      channel_image -= channel_image.mean()
      channel_image /= channel_image.std()
      channel_image *= 64
      channel_image += 128
      channel_image = np.clip(channel_image , 0, 255).astype('uint8')
      display[row *size : (row + 1) * size , col * size : (col + 1) * size] = channel_image

  scale = 1./ size
  plt.figure(figsize = (scale * display.shape[1] , scale * display.shape[0]))
  plt.title(layer_name)
  plt.grid(False)
  plt.imshow(display , aspect='auto', cmap = 'viridis')
