from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('outputs/models/best_model.h5', save_best_only=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=20,
                        batch_size=512,
                        callbacks=[early_stop, checkpoint])
    
    return history
