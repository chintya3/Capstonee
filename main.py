# Import Library
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'

import io
import tensorflow as tf
import numpy as np 
from PIL import Image
from flask import Flask, request, jsonify

#Load model.h5 pake keras
model = tf.keras.models.load_model("tilapia_disease_detecV3.h5")
#Label yang ada di dalam model.h5
label = ["Terinfeksi Aeromonas", "Ikan Sehat", "Terinfeksi white spot"]

app = Flask(__name__)

#Function yang berfungsi untuk melakukam prediksi pada gambar yang diinput
def predict_label(img):
    i = np.asarray(img) / 250.0
    # sesuaikan dengan input_size yang digunakan pada saat pembuatan model
    i = i.reshape(1, 300, 300, 3)
    pred = model.predict(i)
    result = label[np.argmax(pred)]
    return result

@app.route("/predict", methods=["GET", "POST"])
def index():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error": "no file"})
    
    image_bytes = file.read()
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((300, 300), Image.NEAREST)
    pred_label = predict_label(img)

    #object JSON
    prediction = {"prediction": pred_label}
    return jsonify(prediction) 

if __name__ == "__main__":
    app.run(debug=True)
