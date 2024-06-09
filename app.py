import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request
import io
import threading
import base64
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
# Import your machine learning model module


app = Flask(__name__)

local = threading.local()

def load_img(path, target_size):
    img = Image.open(path)
    img = img.resize(target_size)
    return img
    
def img_to_array(img):
    return np.array(img)
    
@app.route('/')
def home():
    return render_template('main_page.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is uploaded
    if 'image' not in request.files:
        return render_template('main_page.html', message='No image file uploaded')
    
    model = load_model('cats_and_dogs_scratch.h5')
    
    layers_outputs = [layer.output for layer in model.layers[:2]]

    activation_model = Model( inputs=model.input , outputs=layers_outputs )
    # Read the image file from the request
    image_file = request.files['image']
    
    # Ensure the file is a valid image
    if image_file.filename == '':
        return render_template('main_page.html', message='No image file selected')
    
    # Read the image data and convert it to a format suitable for processing
    imgs = load_img(image_file, target_size=(150,150))
    img_t = img_to_array(imgs)
    img_t = np.expand_dims(img_t , axis=0)
    img_t = img_t.astype('float32')
    img_t /= 255.
    # image_np = np.array(image)
    
    # Perform inference using your machine learning model
    activations = activation_model.predict(img_t)
    layer_activation = activations[0]
    filters = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    images_per_row = 9
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

    scale = 1.8/ size
    output_pred = model.predict(img_t)
    val = ''
    if (output_pred[0][0] < 0.5):
        val = 'Cat'
    elif (output_pred[0][0] > 0.5):
        val = 'Dog'
    # Convert the output image array back to an image object
    # output_image = Image.fromarray(img_show)
    
    # Convert the output image to a data URL for displaying in the browser
    output_image_data_url = image_to_data_url(display , scale)
    
    # Pass the output image data URL to the template
    return render_template('main_page.html', output_image=output_image_data_url , prediction=val)

def image_to_data_url(image , scale):
    """Converts a PIL Image to a data URL."""
    # with io.BytesIO() as buffer:
        # Convert image to mode 'RGB' before saving as PNG
    fig, ax = plt.subplots(figsize=(scale * image.shape[1] , scale * image.shape[0]))
    ax.matshow(image , aspect='auto')
    
    # Convert the figure to a PNG image
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
        
    data_uri = base64.b64encode(buffer.getvalue()).decode()
    return 'data:image/png;base64,' + data_uri


if __name__ == '__main__':
    app.run(debug=True)
