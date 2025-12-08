import joblib
import pandas as pd
from pathlib import Path
from stock_pred.trainer.utils import load_latest_model
from stock_pred.trainer.prep_data import prepare_data
import json

# Load the latest model at startup
MODEL_PATH = load_latest_model()
model = joblib.load(MODEL_PATH)

path_obj = Path(MODEL_PATH)
filename = path_obj.name
date_str = filename[:8]
args_path = path_obj.parent / f"{date_str}_args.json"
with open(args_path, 'r', encoding='utf-8') as f:
    model_args = json.load(f)

def predict_next_return():
    """
    Predict next-day return given the most recent close price.
    """
    X,y = prepare_data(model_args['ticker_symbol'], model_args['period'], model_args['interval'])
    last_x = X.tail(1)
    prediction = model.predict(last_x)[0]

    return prediction
