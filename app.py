from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import os
from ai_edge_litert import interpreter as litert_interpreter

app = Flask(__name__)
CORS(app)

# Load TFLite model
interpreter = litert_interpreter.Interpreter(
    model_path="automated_wafer_inspection_cnn.tflite"
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Local",
    "Near-Full",
    "Normal",
    "Random",
    "Scratch"
]

@app.route("/")
def home():
    return jsonify({
        "message": "Automated Wafer Inspection API is running",
        "endpoint": "/predict"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400

        file = request.files["file"]

        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Run TFLite inference
        interpreter.set_tensor(
            input_details[0]["index"],
            img_array
        )

        interpreter.invoke()

        prediction = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        predicted_index = int(np.argmax(prediction))
        confidence = float(prediction[predicted_index])
        predicted_class = class_names[predicted_index]

        status = "PASSED" if predicted_class == "Normal" else "DEFECTIVE"

        probabilities = {
            class_names[i]: float(prediction[i])
            for i in range(len(class_names))
        }

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percent": round(confidence * 100, 2),
            "status": status,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
