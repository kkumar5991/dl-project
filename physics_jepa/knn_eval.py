import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from .data import get_dataset_metadata
from .finetuner import JepaFinetuner, VideoMAEFinetuner
from .utils.hydra import compose


class KNNMixin:
    """Mixin that replaces the train() loop with KNN evaluation on extracted embeddings."""

    def create_head(self, metadata):
        # KNN has no trainable head; return a no-op identity module so the
        # parent class doesn't break if it ever inspects model_components.
        return torch.nn.Identity()

    def train(self):
        """Extract embeddings via the parent pipeline, then run KNN."""
        # Reuse the parent's embedding extraction (writes/reads HDF5 cache)
        train_embeddings, train_labels, val_embeddings, val_labels = self.get_embeddings()

        print(f"Train embeddings shape: {train_embeddings.shape}, labels shape: {train_labels.shape}")
        print(f"Val embeddings shape: {val_embeddings.shape}, labels shape: {val_labels.shape}")

        # Flatten spatial dims if present (e.g. attentive pooling path)
        if train_embeddings.ndim > 2:
            train_embeddings = train_embeddings.reshape(train_embeddings.shape[0], -1)
            val_embeddings = val_embeddings.reshape(val_embeddings.shape[0], -1)

        # Normalize embeddings
        scaler = StandardScaler()
        train_embeddings = scaler.fit_transform(train_embeddings)
        val_embeddings = scaler.transform(val_embeddings)

        # KNN hyper-parameters from config
        k_values = self.cfg.ft.get("k_values", [1, 3, 5, 10, 20])
        task = self.cfg.ft.get("task", "regression")
        metric = self.cfg.ft.get("knn_metric", "euclidean")
        weights = self.cfg.ft.get("knn_weights", "distance")

        metadata = get_dataset_metadata(self.cfg.dataset.name)
        param_names = metadata.constant_scalar_names

        results = {}

        print(f"\n{'='*60}")
        print(f"KNN Evaluation - Task: {task}, Metric: {metric}, Weights: {weights}")
        print(f"Dataset: {self.cfg.dataset.name}, Model: {self.cfg.model.objective}")
        print(f"{'='*60}\n")

        for k in k_values:
            print(f"\n--- K = {k} ---")
            if task == "regression":
                knn = KNeighborsRegressor(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1)
                knn.fit(train_embeddings, train_labels)

                train_preds = knn.predict(train_embeddings)
                val_preds = knn.predict(val_embeddings)

                train_mse = mean_squared_error(train_labels, train_preds)
                val_mse = mean_squared_error(val_labels, val_preds)

                results[k] = {"train_mse": train_mse, "val_mse": val_mse}

                print(f"  Train MSE: {train_mse:.6f}")
                print(f"  Val MSE:   {val_mse:.6f}")

                for j, name in enumerate(param_names):
                    param_train_mse = mean_squared_error(train_labels[:, j], train_preds[:, j])
                    param_val_mse = mean_squared_error(val_labels[:, j], val_preds[:, j])
                    results[k][f"train_mse_{name}"] = param_train_mse
                    results[k][f"val_mse_{name}"] = param_val_mse
                    print(f"  {name} - Train MSE: {param_train_mse:.6f}, Val MSE: {param_val_mse:.6f}")

            elif task in ("classification", "binary_classification"):
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1)
                knn.fit(train_embeddings, train_labels.ravel() if train_labels.ndim > 1 and train_labels.shape[1] == 1 else train_labels)

                val_preds = knn.predict(val_embeddings)
                val_labels_flat = val_labels.ravel() if val_labels.ndim > 1 and val_labels.shape[1] == 1 else val_labels

                val_acc = accuracy_score(val_labels_flat, val_preds)
                val_f1 = f1_score(val_labels_flat, val_preds, average='macro', zero_division=0)

                results[k] = {"val_accuracy": val_acc, "val_macro_f1": val_f1}
                print(f"  Val Accuracy: {val_acc:.4f}")
                print(f"  Val Macro F1: {val_f1:.4f}")

        # Log to wandb
        run_name = self.cfg.ft.get("run_name",
            f"{self.cfg.dataset.name}-{self.cfg.dataset.num_frames}frames-{self.cfg.model.objective}-knn"
            f"{'-randominit' if self.trained_model_path is None else ''}")
        # Determine best k
        if task == "regression":
            best_k = min(results, key=lambda k: results[k]["val_mse"])
        else:
            best_k = max(results, key=lambda k: results[k]["val_accuracy"])

        if self.rank == 0 and not self.cfg.get("dry_run", False):
            wandb.init(project="physics-jepa", name=run_name,
                       config=OmegaConf.to_container(self.cfg))

            # -- Per-k metrics (one row per k value) --
            for k in k_values:
                log_row = {"knn/k": k}
                for metric_name, value in results[k].items():
                    log_row[f"knn/{metric_name}"] = value
                wandb.log(log_row)

            # -- Summary scalars (appear in the runs table) --
            wandb.run.summary["knn/best_k"] = best_k
            wandb.run.summary["knn/metric"] = metric
            wandb.run.summary["knn/weights"] = weights
            wandb.run.summary["knn/num_train"] = train_embeddings.shape[0]
            wandb.run.summary["knn/num_val"] = val_embeddings.shape[0]
            wandb.run.summary["knn/embedding_dim"] = train_embeddings.shape[1]

            if task == "regression":
                wandb.run.summary["knn/best_val_mse"] = results[best_k]["val_mse"]
                wandb.run.summary["knn/best_train_mse"] = results[best_k]["train_mse"]
                for j, name in enumerate(param_names):
                    wandb.run.summary[f"knn/best_val_mse_{name}"] = results[best_k].get(f"val_mse_{name}")
                    wandb.run.summary[f"knn/best_train_mse_{name}"] = results[best_k].get(f"train_mse_{name}")
            else:
                wandb.run.summary["knn/best_val_accuracy"] = results[best_k]["val_accuracy"]
                wandb.run.summary["knn/best_val_macro_f1"] = results[best_k]["val_macro_f1"]

            # -- Comparison table across all k values --
            if task == "regression":
                columns = ["k", "train_mse", "val_mse"] + [f"val_mse_{n}" for n in param_names]
                table = wandb.Table(columns=columns)
                for k in k_values:
                    row = [k, results[k]["train_mse"], results[k]["val_mse"]]
                    row += [results[k].get(f"val_mse_{n}", None) for n in param_names]
                    table.add_data(*row)
            else:
                columns = ["k", "val_accuracy", "val_macro_f1"]
                table = wandb.Table(columns=columns)
                for k in k_values:
                    table.add_data(k, results[k]["val_accuracy"], results[k]["val_macro_f1"])
            wandb.log({"knn/results_table": table})

            # -- t-SNE scatter plots of val embeddings --
            print("Computing t-SNE projection of val embeddings...")
            tsne = TSNE(n_components=2, perplexity=min(30, len(val_embeddings) - 1),
                        random_state=self.seed, max_iter=1000)
            coords = tsne.fit_transform(val_embeddings)

            if task == "regression":
                # One scatter per physical parameter, coloured by continuous value
                for j, name in enumerate(param_names):
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sc = ax.scatter(coords[:, 0], coords[:, 1],
                                    c=val_labels[:, j], cmap="viridis",
                                    s=8, alpha=0.7)
                    plt.colorbar(sc, ax=ax, label=name)
                    ax.set_title(f"t-SNE coloured by {name}")
                    ax.set_xlabel("t-SNE 1")
                    ax.set_ylabel("t-SNE 2")
                    fig.tight_layout()
                    wandb.log({f"knn/tsne_{name}": wandb.Image(fig)})
                    plt.close(fig)
            else:
                # Single scatter coloured by discrete class label
                labels_flat = val_labels.ravel() if val_labels.ndim > 1 and val_labels.shape[1] == 1 else val_labels
                unique_classes = np.unique(labels_flat)
                fig, ax = plt.subplots(figsize=(8, 6))
                for cls in unique_classes:
                    mask = labels_flat == cls
                    ax.scatter(coords[mask, 0], coords[mask, 1],
                               s=8, alpha=0.7, label=str(cls))
                ax.legend(markerscale=3, title="Class")
                ax.set_title("t-SNE coloured by class")
                ax.set_xlabel("t-SNE 1")
                ax.set_ylabel("t-SNE 2")
                fig.tight_layout()
                wandb.log({"knn/tsne_classes": wandb.Image(fig)})
                plt.close(fig)

            wandb.finish()

        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        if task == "regression":
            print(f"Best K: {best_k} (Val MSE: {results[best_k]['val_mse']:.6f})")
        else:
            print(f"Best K: {best_k} (Val Accuracy: {results[best_k]['val_accuracy']:.4f})")

        self.cleanup_embedding_files()
        return results


class JepaKNNEvaluator(KNNMixin, JepaFinetuner):
    """KNN evaluation using a pretrained JEPA encoder."""
    pass


class VideoMAEKNNEvaluator(KNNMixin, VideoMAEFinetuner):
    """KNN evaluation using a pretrained VideoMAE encoder."""
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN evaluation of Physics JEPA representations")
    parser.add_argument("config", type=str, help="Path to config YAML file")
    parser.add_argument("overrides", nargs="*", help="Hydra-style config overrides")
    parser.add_argument("--trained_model_path", type=str, default=None,
                        help="Path to pretrained encoder checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run
    cfg.seed = args.seed

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.model.objective == "jepa":
        evaluator = JepaKNNEvaluator(cfg, trained_model_path=args.trained_model_path)
    elif cfg.model.objective == "videomae":
        evaluator = VideoMAEKNNEvaluator(cfg, trained_model_path=args.trained_model_path)
    else:
        raise ValueError(f"Unknown objective: {cfg.model.objective}")

    evaluator.train()
