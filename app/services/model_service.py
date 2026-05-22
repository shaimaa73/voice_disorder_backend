import os
import numpy as np
import librosa
import torch
import torch.nn as nn

# CONFIG
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "best_voice_model.pt")

CLASSES = ["healthy", "lary", "rek"]
ID2LABEL = {0: "healthy", 1: "lary", 2: "rek"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "sample_rate": 22050,
    "duration": 3.0,
    "n_mfcc": 40,
    "n_mels": 128,
    "hop_length": 512,
    "n_fft": 2048,

    "cnn_channels": [32, 64, 128],
    "lstm_hidden": 128,
    "lstm_layers": 2,
    "dropout": 0.4,
    "num_classes": 3,
}

# MODEL
class CNNBiLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        channels = cfg["cnn_channels"]
        drop = cfg["dropout"]

        self.cnn = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(drop / 2),

            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(drop / 2),

            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout2d(drop / 2),
        )

        self.lstm_hidden = cfg["lstm_hidden"]
        self.lstm_layers = cfg["lstm_layers"]
        self.drop = drop
        self.num_classes = cfg["num_classes"]

        self.lstm = None
        self.classifier = None

    def build_dynamic_layers(self, cnn_out):
        B, C, F, T = cnn_out.shape
        lstm_input_size = C * F

        if self.lstm is None:
            self.lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=self.lstm_hidden,
                num_layers=self.lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=self.drop if self.lstm_layers > 1 else 0.0
            ).to(cnn_out.device)

            self.classifier = nn.Sequential(
                nn.Linear(self.lstm_hidden * 2, 128),
                nn.ReLU(),
                nn.Dropout(self.drop),
                nn.Linear(128, self.num_classes)
            ).to(cnn_out.device)

    def forward(self, x):
        cnn_out = self.cnn(x)

        self.build_dynamic_layers(cnn_out)

        B, C, F, T = cnn_out.shape
        seq = cnn_out.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)

        lstm_out, _ = self.lstm(seq)

        pooled = lstm_out.mean(dim=1)

        return self.classifier(pooled)

# AUDIO PROCESSING
def load_audio(path, sr, duration):
    target_len = int(sr * duration)

    y, _ = librosa.load(path, sr=sr, mono=True)

    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    return y


def extract_features(y, sr, cfg):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=cfg["n_mels"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"]
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=cfg["n_mfcc"],
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"]
    )

    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([mel_db, mfcc, delta1, delta2], axis=0)

    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True) + 1e-9

    features = (features - mean) / std

    return features.astype(np.float32)

# LOAD MODEL
model = CNNBiLSTM(CONFIG).to(DEVICE)

dummy = torch.zeros(1, 1, 248, 130).to(DEVICE)
_ = model(dummy)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model.eval()

# PREDICTION
def predict_audio(file_path):

    y = load_audio(
        file_path,
        CONFIG["sample_rate"],
        CONFIG["duration"]
    )

    features = extract_features(
        y,
        CONFIG["sample_rate"],
        CONFIG
    )

    x = torch.tensor(features).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)

        probs = torch.softmax(logits, dim=1)

        pred = torch.argmax(probs, dim=1).item()

        confidence = probs[0][pred].item()

    return {
        "prediction": ID2LABEL[pred],
        "confidence": float(confidence)
    }