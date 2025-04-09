import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess(df):
    df = df.copy()

    # Nettoyage : remplacer les infinis par NaN et supprimer les lignes avec NaN
    df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Encodage du label (0 = BENIGN, 1 = ATTACK)
    if 'Label' in df.columns:
        df['Label'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
        df['Label'] = df['Label'].astype('int64')  # Conversion en entier

    # Encodage des colonnes catégorielles (AUCUN .drop('Label'))
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Normalisation des colonnes numériques
    if 'Label' in df.columns:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('Label')
    else:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df