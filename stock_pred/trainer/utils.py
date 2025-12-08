import os
from pathlib import Path
import datetime
import joblib
import json

MODELS_DIR = Path("stock_pred/models")

def timestamp():
    return datetime.date.today().strftime("%Y%m%d")

def save_args(args):
    """
    Save training arguments to a JSON file with a timestamp.
    """
    filename = MODELS_DIR / f"{timestamp()}_args.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(args, f)
    return filename

def save_model(model, versioned=False):
    """
    Save model locally with optional versioning.
    """
    MODELS_DIR.mkdir(exist_ok=True)

    if versioned:
        filename = MODELS_DIR / f"{timestamp()}_model.pkl"
    else:
        filename = MODELS_DIR / "model.pkl"

    joblib.dump(model, filename)
    return filename

def load_latest_model():
    """
    Load the most recent model file.
    """
    files = list(MODELS_DIR.glob("*model.pkl"))
    if not files:
        raise FileNotFoundError("No model files found in models/")

    latest = max(files, key=os.path.getctime)
    return latest
