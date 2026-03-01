from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model
model = tf.keras.models.load_model("my_model.keras")

CLASS_NAMES = [
    'Scottish Deerhound',
    'Maltese Dog',
    'Bernese Mountain Dog'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)

            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.reshape(img, (1, 224, 224, 3))

            pred = model.predict(img)
            predicted_index = np.argmax(pred)

            prediction = CLASS_NAMES[predicted_index]
            confidence = round(float(np.max(pred)) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)