from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import io
import mlflow.pytorch

app = Flask(__name__)

# Load the model
mlflow.set_tracking_uri('http://127.0.0.1:5000')
print("Flask MLflow Tracking URI:", mlflow.get_tracking_uri())
registered_model_name='lenet_model'
model_version='1'
model_uri = f"models:/{registered_model_name}/{model_version}"
model = mlflow.pytorch.load_model(model_uri)
model.eval()

# Define preprocessing (adjust if needed)
transform = transforms.Compose([
    transforms.Grayscale(),  # If image isn't already grayscale
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # mean/std from MNIST
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert('L')  # convert to grayscale

    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()

    return jsonify({'digit': prediction})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5001,debug=True)

    ###
    # call the service using the following
    # curl -X POST -F "file=@sample_img_5.png" http://localhost:5001/predict