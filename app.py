from flask import Flask, render_template, request
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np

app = Flask(__name__)

# Define the CNN model
def create_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # Assuming 10 classes for classification
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_cnn_model()
# Load pre-trained weights if available
# model.load_weights('path_to_weights.h5')

# Update class names with specific animals
class_names = ['cat', 'dog', 'lion', 'tiger', 'elephant', 'bear', 'wolf', 'zebra', 'giraffe', 'monkey']

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)  # Assuming preprocessing is similar to VGG16

    yhat = model.predict(image)
    predicted_class_index = np.argmax(yhat, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    confidence = yhat[0][predicted_class_index]

    classification = '%s (%.2f%%)' % (predicted_class_name, confidence * 100)
    image_filename = imagefile.filename  # Get the name of the uploaded image file

    return render_template('index.html', prediction=classification, image_filename=image_filename)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
