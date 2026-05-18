import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import librosa
import numpy as np
import torch.nn.functional as F
import tempfile

from models.multitask_crnn import MultiTaskCRNN

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NUM_SCENES = 8
NUM_SCENES = 10
NUM_DEVICES = 3

SCENE_LABELS = [
    "airport",
    "bus",
    "metro",
    "metro_station",
    "park",
    "public_square",
    "shopping_mall",
    "street_pedestrian",
    "street_traffic",
    "tram"
]

DEVICE_LABELS = ["Device A", "Device B", "Device C"]

MAX_FRAMES = 320
SR = 44100

model = MultiTaskCRNN(NUM_SCENES, NUM_DEVICES).to(DEVICE)
checkpoint = torch.load("../training/artifacts/checkpoint_epoch12.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def preprocess(file_path):
    y, _ = librosa.load(file_path, sr=SR)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=128,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel)

    x = torch.tensor(mel_db).unsqueeze(0).float()

    T = x.shape[-1]

    if T < MAX_FRAMES:
        x = F.pad(x, (0, MAX_FRAMES - T))
    else:
        x = x[:, :, :MAX_FRAMES]

    return x.unsqueeze(0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    x = preprocess(path).to(DEVICE)

    with torch.no_grad():
        scene_logits, device_logits = model(x)

    scene_probs = torch.softmax(scene_logits, dim=1).cpu().numpy()[0]
    device_probs = torch.softmax(device_logits, dim=1).cpu().numpy()[0]

    scene_idx = int(np.argmax(scene_probs))
    device_idx = int(np.argmax(device_probs))

    return {
        "scene": SCENE_LABELS[scene_idx],
        "device": DEVICE_LABELS[device_idx],
        "scene_probs": scene_probs.tolist(),
        "device_probs": device_probs.tolist()
    }