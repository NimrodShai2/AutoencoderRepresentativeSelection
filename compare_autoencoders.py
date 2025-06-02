import argparse
import sys
from utils import read_and_process_UHGG, read_full_UHGG
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from autoencoder_builder import evaluate_autoencoder, build_autoencoder, tune_dropout_rate


def get_args():
    if len(sys.argv) != 3:
        print("Usage: python compare_autoencoders.py <in_file> <model_path>")
        sys.exit(1)
    if not sys.argv[1].endswith('.tsv'):
        print("Error: Input file must be a .tsv file containing UHGG metadata.")
        sys.exit(1)
    if not sys.argv[2].endswith('.keras'):
        print("Error: Model path must end with .keras to save the Keras model.")
        sys.exit(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", type=str, help="Input file containing the UHGG metadata.")
    parser.add_argument("model_path", type=str, help="Path to save the best model.")
    return parser.parse_args()


architectures = [
    {"hidden_layers": [16], "latent_dim": 4},
    {"hidden_layers": [64], "latent_dim": 4},
    {"hidden_layers": [64, 32], "latent_dim": 8},
    {"hidden_layers": [128, 64], "latent_dim": 16},
    {"hidden_layers": [64, 32, 16], "latent_dim": 8},
]

if __name__ == "__main__":
    args = get_args()
    df = read_and_process_UHGG(args.in_file)
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    print("Data loaded and scaled.", flush=True)

    # Sample a random subset of genomes
    np.random.seed(42)  # For reproducibility
    indices = np.random.choice(X.shape[0], size=8000, replace=False)
    X_samp = X[indices]

    print("Evaluating autoencoder architectures...", flush=True)

    # Run the autoencoder evaluation
    results = []
    for arch in architectures:
        kwargs = {
            "input_dim": X_samp.shape[1],
            "hidden_layers": arch["hidden_layers"],
            "latent_dim": arch["latent_dim"],
            "activation": "relu",
            "use_dropout": True,
            "dropout_rate": 0.2
        }

        mean_mse, std_mse = evaluate_autoencoder(X_samp, build_autoencoder, kwargs)
        results.append({
            "hidden_layers": arch["hidden_layers"],
            "latent_dim": arch["latent_dim"],
            "mean_mse": mean_mse,
            "std_mse": std_mse
        })
        print(f"Evaluated {arch}: MSE = {mean_mse:.5f} ± {std_mse:.5f}")

    print("Evaluation complete. Saving results...", flush=True)
    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        x=[str(r["hidden_layers"]) for r in results],
        y=[r["mean_mse"] for r in results],
        yerr=[r["std_mse"] for r in results],
        fmt="o-", capsize=5
    )
    plt.title("Autoencoder Architecture Comparison (MSE)")
    plt.xlabel("Hidden Layer Configuration")
    plt.ylabel("Mean Reconstruction MSE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("autoencoder_comparison.png")
    plt.show()

    print("Plot saved as 'autoencoder_comparison.png'", flush=True)

    best_config = min(results, key=lambda r: r["mean_mse"])
    print(f"\nBest Config: {best_config}", flush=True)

    print("Fine-tuning the best architecture for dropout rate...", flush=True)

    # Fine tune the best architecture for dropout rate
    # Again sample a random subset of genomes
    X_samp = X[np.random.choice(X.shape[0], size=8000, replace=False)]
    best_kwargs = {
        "input_dim": X_samp.shape[1],
        "hidden_layers": best_config["hidden_layers"],
        "latent_dim": best_config["latent_dim"],
        "activation": "relu",
        "use_dropout": True,
        "dropout_rate": 0.2
    }
    dropout_space = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    dropout_results = tune_dropout_rate(
        X=X_samp,
        base_config=best_kwargs,
        dropout_rates=dropout_space,
        build_fn=build_autoencoder,
        n_splits=5,
        verbose=0
    )

    # Save dropout results to a CSV file
    dropout_results.to_csv("dropout_tuning_results.csv", index=False)

    # Plotting dropout tuning results
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        x=dropout_results['dropout_rate'],
        y=dropout_results['mean_mse'],
        yerr=dropout_results['std_mse'],
        fmt='o-',
        capsize=5
    )
    plt.title("Dropout Rate Tuning (MSE)")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Mean Reconstruction MSE")
    plt.xticks(dropout_space)
    plt.tight_layout()
    plt.savefig("dropout_tuning_results.png")
    plt.show()
    print("Dropout tuning results saved as 'dropout_tuning_results.png'", flush=True)

    print("Dropout rate tuning complete. Results:", flush=True)
    best_result = dropout_results.loc[dropout_results['mean_mse'].idxmin()]

    # Construct the best model with the best dropout rate
    best_kwargs["dropout_rate"] = best_result["dropout_rate"]
    best_kwargs["use_dropout"] = best_result["dropout_rate"] > 0.0
    best_model = build_autoencoder(**best_kwargs)

    print(f"\nBest Config: {best_result}", flush=True)
    print(f"Best Dropout Rate: {best_result['dropout_rate']:.2f}", flush=True)
    print(f"Best MSE: {best_result['mean_mse']:.5f} ± {best_result['std_mse']:.5f}", flush=True)
    print("Training the best model on the full dataset...", flush=True)

    # Train the best model on the full dataset
    best_model.compile(optimizer='adam', loss='mse')
    best_model.fit(X, X, epochs=100, batch_size=256, validation_split=0.2, verbose=0)

    print("Training complete. Saving...", flush=True)

    # Save the best model
    best_model.save(args.model_path)
    print(f"Best model saved to {args.model_path}", flush=True)

    # Add the reconstruction MSE to the DataFrame
    X_pred = best_model.predict(X, verbose=0)
    reconstruction_mse = np.mean(np.square(X - X_pred), axis=1)

    df_full = read_full_UHGG(args.in_file)
    df_full['reconstruction_mse'] = reconstruction_mse

    # Normalize the reconstruction MSE
    df_full['reconstruction_mse'] = (df_full['reconstruction_mse'] - df_full['reconstruction_mse'].min()) / \
                                    (df_full['reconstruction_mse'].max() - df_full[
                                        'reconstruction_mse'].min())

    print("Normalized Reconstruction MSE calculated and added to DataFrame.", flush=True)

    # Save the DataFrame with reconstruction MSE
    df_full.to_csv("uhgg_with_reconstruction_mse_metadata.tsv", sep='\t', index=False)
    print("UHGG metadata with reconstruction MSE saved to 'uhgg_with_reconstruction_mse_metadata.tsv'", flush=True)
