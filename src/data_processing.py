import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess(df):
    df = df.copy()
    
    df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    df.dropna(inplace=True)

    if 'Label' in df.columns:
        df['Label'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)

    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols = [col for col in cat_cols if col != 'Label']

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    if 'Label' in df.columns:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Label')
    else:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def preprocess_single_row(row):
    # Assumes row is a Series; convert to DataFrame with one row
    df = pd.DataFrame([row])
    
    # Apply whatever transformations you used in preprocess()
    # e.g. scaling, encoding
    df = df.drop(['Label'], axis=1, errors='ignore')
    
    # Make sure to match the training format!
    return df

