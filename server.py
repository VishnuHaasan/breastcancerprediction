from models.cnn import CNNModel
from models.vit import ViTModel
from models.rfr import RfClassifier
from flask import Flask, request, jsonify, send_from_directory
import werkzeug
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route("/train")
def train():
    try:

        cnn_model = CNNModel()
        cnn_model.fit(train_path="/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/balanced_data", test_path="/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/data/test_set")

        vit_model = ViTModel()
        vit_model.fit(path="/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/balanced_data")

        rfr_model = RfClassifier()
        rfr_model.fit()
        
        resp = {
            "Response": "Success",
            "StatusCode": 201,
            "Safe": True,
            "Message": "Model Trained Successfully!"
        }
        return jsonify(resp)

    except Exception as e:
        print(e)
        resp = {
            "Response": "Failure",
            "StatusCode": 500,
            "Safe": False,
            "Message": f"An error has occured {e}"
        }
        return jsonify(resp), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        classifier = RfClassifier.load('/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/models/rfr_model.h5')

        image = request.files['image']
        filename = werkzeug.utils.secure_filename(image.filename)
        image.save("testing/0/"+filename)

        predictions = classifier.predict("/home/local/ZOHOCORP/vishnu-pt5599/Desktop/BreastCancerPrediction/testing/")

        print(predictions)

        resp = {
            "Response": "Success",
            "StatusCode": 200,
            "Safe": True,
        }

        if predictions[0] == 0:
            resp['Message'] = "The cancer is benign."

        else:
            resp['Message'] = "The cancer is malignant."


        os.remove("testing/0/"+filename)

        return jsonify(resp)

    except Exception as e:
        print(e)
    
        resp = {
            "Response": "Failure",
            "StatusCode": 500,
            "Safe": False,
            "Message": f"An error has occured {e}"
        }
        return jsonify(resp), 500

@app.route('/image/<path:path>')
def send_image(path):
    return send_from_directory('balanced_data/0', path)

@app.route('/test', methods=['POST'])
def test():
    dummy = {
        "predictions": [
            {
                "link": "http://localhost:5000/image/14304_idx5_x1_y2601_class0.png",
                "msg": "YOO!, thats so cool"
            }
            for _ in range(9)
        ]
    }

    return jsonify(dummy), 200

if __name__ == "__main__":
    app.run()




    

    

