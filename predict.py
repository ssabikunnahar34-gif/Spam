import os
import sys
import joblib
from train import clean_text, train_model

def load_pipeline():
    model_path = os.path.join(os.path.dirname(__file__), "sms_spam_pipeline.pkl")
    if not os.path.exists(model_path):
        print("[!] Model file not found. Running training script...")
        train_model()
    return joblib.load(model_path)

def predict_sms(text, pipeline=None):
    if pipeline is None:
        pipeline = load_pipeline()
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    return "Spam" if pred == 1 else "Ham"

def main():
    pipeline = load_pipeline()
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = predict_sms(text, pipeline)
        print("=" * 50)
        print(f"SMS Input : \"{text}\"")
        print(f"Prediction: {result}")
        print("=" * 50)
    else:
        print("\n--- SMS Spam Predictor ---")
        samples = [
            "Congratulations! You have won a free iPhone. Click here now.",
            "Hi, are we meeting tomorrow at 3 PM?"
        ]
        for s in samples:
            res = predict_sms(s, pipeline)
            print(f"Input : \"{s}\"\nResult: {res}\n")
            
        while True:
            try:
                msg = input("Enter SMS message (or 'exit'): ").strip()
                if msg.lower() in ['exit', 'quit', 'q']:
                    break
                if msg:
                    print(f"--> Prediction: {predict_sms(msg, pipeline)}\n")
            except (KeyboardInterrupt, EOFError):
                break

if __name__ == "__main__":
    main()
