import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your trained model
model = load_model('best_model.h5')  # Change to 'model.keras' if that's your format

# Define your class labels (should match your training class order)
class_labels = [
    "Angelina Jolie", "Brad Pitt", "Denzel Washington","Hugh Jackman",
    "Jennifer Lawrence","Johnny Depp","Kate Winslet","Leonardo DiCaprio",
    "Megan Fox", "Natalie Portman","Nicole Kidman","Robert Downey Jr.", 
    "Sandra Bullock","Scarlett Johansson", "Tom Cruise","Tom Hanks", 
    "Will Smith"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            predicted_class = class_labels[np.argmax(preds)]
            prediction = f"Predicted actor: {predicted_class}"
            image_url = filepath

    return render_template('index.html', prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
