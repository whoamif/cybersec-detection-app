import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

columns = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
]

def load_dataset(path):
    return pd.read_csv(path, names=columns)

def binary_label(label):
    return 0 if label == 'normal' else 1

def preprocess_data(df):
    df = df.copy()
    
    # Supprimer la colonne difficulty (inutile pour le modèle)
    df.drop(['difficulty'], axis=1, inplace=True)

    # Encoder les labels binaire (normal = 0, attack = 1)
    df['label'] = df['label'].apply(binary_label)

    # Encoder les variables catégorielles
    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Séparer X et y
    X = df.drop(['label'], axis=1)
    y = df['label']

    # Normaliser les features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Load the dataset
df = load_dataset('your_dataset.csv')

# Preprocess the data
X_scaled, y = preprocess_data(df)

# 1. Check if the 'difficulty' column was removed
print("Columns after preprocessing: ", df.columns)

# 2. Verify label encoding for the 'label' column
print("Unique values in the label column after encoding: ", df['label'].unique())

# 3. Check encoding for categorical columns
for col in ['protocol_type', 'service', 'flag']:
    print(f"Unique values in {col} after encoding: ", df[col].unique())

# 4. Verify scaling by checking the range of the first few features
print("Min and Max values of first 5 features after scaling:")
print(pd.DataFrame(X_scaled).iloc[:, :5].min(), pd.DataFrame(X_scaled).iloc[:, :5].max())

# 5. Optionally, check the first few rows of X_scaled and y
print("First few rows of X_scaled:\n", X_scaled[:5])
print("First few labels (y):\n", y.head())
