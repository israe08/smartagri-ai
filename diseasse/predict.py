import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from diseasse.disease_infos import DISEASE_INFO

import os

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_model.keras"
)

model = tf.keras.models.load_model(MODEL_PATH)
class_names = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

def predict_disease(img_path):

    img = image.load_img(img_path, target_size=(224, 224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    info = DISEASE_INFO[predicted_class]

    return {
        "detected": predicted_class != "Tomato_healthy",
        "name": predicted_class,
        "confidence": float(
            round(confidence / 100, 4)
        ),
        "severity": info["severity"],
        "action": info["action"]
    }

if __name__ == "__main__":

    TEST_IMAGE = os.path.join(
        BASE_DIR,
        "image_test.jpg"
    )

    result = predict_disease(TEST_IMAGE)
    print(result)