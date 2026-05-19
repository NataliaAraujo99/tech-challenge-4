from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_model(input_shape):
    """Cria a arquitetura da rede neural LSTM."""
    model = Sequential()

    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(LSTM(50))
    model.add(Dropout(0.2))

    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_absolute_error',
        metrics=['mean_squared_error']
    )

    return model


def train_model(model, X_train, y_train, validation_split=0.1, epochs=100, batch_size=32):
    """Treina o modelo com validação e redução de learning rate."""
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=False,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    return model