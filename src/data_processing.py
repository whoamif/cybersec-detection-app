import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_cicids2017_dataset(path):
    df = pd.read_csv(path, low_memory=False)
    return df

def preprocess(df):
    df = df.copy()
    
    # Supprimer les colonnes inutiles si présentes
    cols_to_drop = ['Flow ID', 'Timestamp', 'Source IP', 'Destination IP']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')

    # Remplacer les valeurs infinies ou NaN
    df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Encodage du label (0 = BENIGN, 1 = ATTACK)
    df['Label'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in x.upper() else 1)

    # Encodage des colonnes catégorielles
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols = [col for col in cat_cols if col != 'Label']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Normalisation des colonnes numériques
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Label')
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
