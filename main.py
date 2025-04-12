import pandas as pd
from src.data_processing import preprocess
from src.model import build_model
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    # Load and preprocess the data
    df = pd.read_csv('/Users/mac/Desktop/cybersec-detection-app/data/CICIDS2017_improved/full_dataset.csv')
    df = preprocess(df)  # Now pass the DataFrame to preprocess
    
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # Build the model
    model = build_model(X.shape[1])
    
    # Train the model
    train_model(model, X, y)
    
    # Evaluate the model
    evaluate_model(model, X, y)

if __name__ == '__main__':
    main()
