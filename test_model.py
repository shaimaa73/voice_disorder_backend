import os
from app.services.model_service import predict_audio

FOLDER = "test_audio"

for file in os.listdir(FOLDER):
    if file.endswith(".wav"):
        file_path = os.path.join(FOLDER, file)

        result = predict_audio(file_path)

        print(f"File: {file}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}")
        print("-" * 30)