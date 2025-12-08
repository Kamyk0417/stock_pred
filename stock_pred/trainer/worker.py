import time
import datetime
from train import train_model
from utils import save_model

def log(msg):
    print(f"[WORKER] {datetime.datetime.utcnow()} - {msg}")

def run_daily_training():
    last_run_date = None

    while True:
        today = datetime.date.today()

        # Run only once per day
        if today != last_run_date:
            log("Starting daily model retraining...")

            model = train_model()
            
            # Save locally (versioned)
            model_path = save_model(model, versioned=True)
            log(f"Model saved at {model_path}")

            last_run_date = today
            log("Daily retraining completed.")

        time.sleep(300)

if __name__ == "__main__":
    log("Worker started")
    run_daily_training()
