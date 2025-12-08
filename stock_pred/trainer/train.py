from prep_data import train_prep
import utils

import matplotlib.pyplot as plt
import pandas as pd
import joblib
import json

from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

args = {
    "ticker_symbol": "Gold",
    "period": "1y",
    "interval": "1d"
}

def train_model():
    X_train, X_test, y_train, y_test = train_prep(args['ticker_symbol'], args['period'], args['interval'])

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.4f}%")
    print(classification_report(y_test, y_pred))

    return model

if __name__ == "__main__":
    model = train_model()
    utils.save_model(model, versioned=True)
    utils.save_args(args)
