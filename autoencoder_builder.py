import pandas as pd
from tensorflow.keras import layers, Model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


def build_autoencoder(
        input_dim: int,
        hidden_layers: list[int],
        latent_dim: int = 8,
        activation: str = "relu",
        use_dropout: bool = False,
        dropout_rate: float = 0.2
) -> Model:
    """
    Builds a customizable autoencoder.

    Parameters:
    - input_dim: Number of input features
    - hidden_layers: List of hidden layer sizes for encoder (mirrored in decoder)
    - latent_dim: Size of the latent space
    - activation: Activation function to use in hidden layers
    - use_dropout: Whether to apply dropout after each layer
    - dropout_rate: Dropout rate (used if use_dropout=True)

    Returns:
    - Keras Model (autoencoder)
    """

    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    x = inputs
    for h in hidden_layers:
        x = layers.Dense(h, activation=activation)(x)
        if use_dropout:
            x = layers.Dropout(dropout_rate)(x)
    latent = layers.Dense(latent_dim, activation="linear", name="latent")(x)

    # Decoder
    x = latent
    for h in reversed(hidden_layers):
        x = layers.Dense(h, activation=activation)(x)
        if use_dropout:
            x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    return Model(inputs, outputs)


def evaluate_autoencoder(
        X: np.ndarray,
        build_fn,
        build_kwargs: dict,
        n_splits: int = 5,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 10,
        verbose: int = 0
) -> tuple:
    """
    Evaluates a given autoencoder architecture using K-Fold CV.

    Parameters:
    - X: Input feature matrix (normalized)
    - build_fn: Function that returns a compiled Keras autoencoder
    - build_kwargs: Keyword arguments for the build_fn
    - n_splits: K in K-Fold
    - epochs: Max training epochs
    - batch_size: Batch size
    - patience: Early stopping patience
    - verbose: Verbosity level

    Returns:
    - mean_mse: Average MSE across folds
    - std_mse: Std of MSE across folds
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mses = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]

        model = build_fn(**build_kwargs)
        model.compile(optimizer='adam', loss='mse')

        es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        model.fit(X_train, X_train,
                  validation_data=(X_val, X_val),
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[es],
                  verbose=verbose)

        X_val_pred = model.predict(X_val, verbose=0)
        mse = mean_squared_error(X_val, X_val_pred)
        mses.append(mse)

    return np.mean(mses), np.std(mses)


def tune_dropout_rate(
        X: np.ndarray,
        base_config: dict,
        dropout_rates: list[float],
        build_fn,
        n_splits: int = 5,
        verbose: int = 0
) -> pd.DataFrame:
    """
    Tunes dropout rate for a given architecture by evaluating reconstruction MSE.

    Parameters:
    - X: Normalized input data
    - base_config: Dictionary with 'input_dim', 'hidden_layers', 'latent_dim', 'activation'
    - dropout_rates: List of dropout rates to try (e.g., [0.0, 0.1, 0.2, 0.3, 0.4])
    - build_fn: The autoencoder constructor function
    - n_splits: K-Fold cross-validation splits
    - verbose: Keras verbosity level

    Returns:
    - DataFrame with results for each dropout rate
    """
    results = []

    for dr in dropout_rates:
        config = base_config.copy()
        config.update({
            "use_dropout": dr > 0,
            "dropout_rate": dr
        })

        mean_mse, std_mse = evaluate_autoencoder(
            X=X,
            build_fn=build_fn,
            build_kwargs=config,
            n_splits=n_splits,
            verbose=verbose
        )

        results.append({
            "dropout_rate": dr,
            "mean_mse": mean_mse,
            "std_mse": std_mse
        })

        print(f"Dropout {dr:.2f}: MSE = {mean_mse:.5f} ± {std_mse:.5f}")

    return pd.DataFrame(results)
