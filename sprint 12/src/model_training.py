import os
import sys
import json
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import joblib

# Suppress TensorFlow info/warning logs
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ── Keras model & wrapper ─────────────────────────────────────────────────────
def build_keras_model(input_dim, dropout_rate=0.3, learning_rate=0.001):
    """
    Generic regression NN for tabular data.
    Dense → Dropout → Dense → Dropout → Dense → output ✅
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(dropout_rate),        # Dropout parameter ✅
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  # LR adaptation ✅
        loss='mse',
        metrics=['mae'],
    )
    return model


class KerasRegressorWrapper:
    """
    Sklearn-compatible wrapper for a Keras regression model.
    Supports fit/predict so it can live inside a sklearn Pipeline. ✅
    Callbacks:
      - EarlyStopping     (restore best weights)   ✅
      - ReduceLROnPlateau (halve LR on plateau)    ✅
      - ModelCheckpoint   (save best epoch, final fit only) ✅
    """

    def __init__(self, build_fn=build_keras_model, epochs=150, batch_size=64,
                 validation_split=0.1, dropout_rate=0.3, learning_rate=0.001,
                 checkpoint_path=None):
        self.build_fn         = build_fn
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.validation_split = validation_split
        self.dropout_rate     = dropout_rate
        self.learning_rate    = learning_rate
        self.checkpoint_path  = checkpoint_path  # None during CV, set for final fit
        self.model_           = None
        self.history_         = None

    def fit(self, X, y, **fit_params):
        callbacks = [
            keras.callbacks.EarlyStopping(             # Early stopping ✅
                patience=15, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(         # LR adaptation ✅
                factor=0.5, patience=7, min_lr=1e-6, verbose=0),
        ]
        if self.checkpoint_path:
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            callbacks.append(keras.callbacks.ModelCheckpoint(  # Model checkpoint ✅
                self.checkpoint_path, save_best_only=True, verbose=0))

        self.model_ = self.build_fn(
            X.shape[1], self.dropout_rate, self.learning_rate)
        self.history_ = self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict(self, X):
        return self.model_.predict(X, verbose=0).flatten()

    def get_params(self, deep=True):
        return {
            'build_fn':         self.build_fn,
            'epochs':           self.epochs,
            'batch_size':       self.batch_size,
            'validation_split': self.validation_split,
            'dropout_rate':     self.dropout_rate,
            'learning_rate':    self.learning_rate,
            'checkpoint_path':  self.checkpoint_path,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


# ── 1. Train & compare all candidates via 5-fold CV ──────────────────────────
def train_models(pipelines, X_train, y_train, k_folds=5, random_state=12345, metric="rmse"):
    """
    Evaluate each pipeline with KFold(k_folds) CV. ✅
    Tracks train time and peak memory per model. ✅
    Returns (results dict, best_name).
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    # ── Live comparison chart ─────────────────────────────────────────────────
    plt.ion()
    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 6))
    lbl_dir = "lower is better" if _lower_is_better(metric) else "higher is better"
    ax_cmp.set_title("Training models…")
    ax_cmp.set_xlabel(f"Mean CV {metric.upper()} ({lbl_dir})")
    plt.tight_layout()
    plt.pause(0.05)

    results     = {}
    names_done  = []
    scores_done = []
    errs_done   = []

    for idx, (name, pipeline) in enumerate(pipelines.items()):
        ax_cmp.set_title(f"Training: {name}…")
        fig_cmp.canvas.draw(); fig_cmp.canvas.flush_events(); plt.pause(0.05)

        fold_scores = []
        tracemalloc.start()
        t0 = time.time()

        for train_idx, val_idx in kf.split(X_train):
            X_fold_tr = X_train.iloc[train_idx]
            y_fold_tr = y_train.iloc[train_idx]
            X_fold_vl = X_train.iloc[val_idx]
            y_fold_vl = y_train.iloc[val_idx]

            fold_pipe = clone(pipeline)
            fold_pipe.fit(X_fold_tr, y_fold_tr)
            pred = fold_pipe.predict(X_fold_vl)

            fold_scores.append(_calculate_score(y_fold_vl, pred, metric))

        train_time = time.time() - t0
        _, peak    = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        mean_s = float(np.mean(fold_scores))
        std_s  = float(np.std(fold_scores))

        results[name] = {
            "pipeline":    pipeline,
            "score":       mean_s,
            "fold_scores": fold_scores,
            "train_time":  train_time,
            "peak_mem_mb": peak / 1024 / 1024,   # Training statistics ✅
        }

        names_done.append(name)
        scores_done.append(mean_s)
        errs_done.append(std_s)

        # Update live bar chart
        ax_cmp.clear()
        ax_cmp.set_title(f"Model Comparison — Mean CV {metric.upper()}")
        ax_cmp.set_xlabel(f"{metric.upper()} ({lbl_dir})")
        ax_cmp.barh(names_done, scores_done, xerr=errs_done, capsize=3)
        ax_cmp.invert_yaxis()
        plt.tight_layout(); fig_cmp.canvas.draw(); fig_cmp.canvas.flush_events()
        plt.pause(0.1)

    best_name = _get_best_model(results, metric)

    # Highlight winner
    ax_cmp.clear()
    ax_cmp.set_title(f"CV Results — Winner: {best_name}")
    ax_cmp.set_xlabel(f"Mean CV {metric.upper()} ({lbl_dir})")
    colors = ['orange' if n == best_name else 'tab:blue' for n in names_done]
    ax_cmp.barh(names_done, scores_done, xerr=errs_done, color=colors, capsize=3)
    ax_cmp.invert_yaxis()
    plt.tight_layout(); fig_cmp.canvas.draw(); fig_cmp.canvas.flush_events()
    plt.ioff(); plt.show(block=False)

    # ── Fold-score breakdown chart (training visualization — x=fold) ──────────
    plot_fold_scores(results, metric=metric)

    return results, best_name


def plot_fold_scores(results, metric="rmse"):
    """
    Plot per-fold score for every model. X-axis = fold index. ✅
    (Training visualization: timeseries — iterations of learning)
    """
    n    = len(results)
    fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))
    fig.suptitle(f"CV Fold {metric.upper()} per Model")
    
    # Handle both single plot array and multiplot matrix correctly 
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for ax, (name, r) in zip(axes_flat, results.items()):
        fmt = ",.4f" if metric.lower() == "r2" else ",.0f"
        folds = list(range(1, len(r["fold_scores"]) + 1))
        ax.plot(folds, r["fold_scores"], "o-", linewidth=2, markersize=6)
        mean_str = format(r['score'], fmt)
        ax.axhline(r["score"], color='orange', linestyle="--", linewidth=1.5, label=f"mean={mean_str}")
        ax.set_title(name)
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric.upper())
        ax.set_xticks(folds)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show(block=False)


def plot_keras_history(history, name="KerasNN"):
    """
    Plot epoch vs loss and val_loss for a Keras training history. ✅
    (Training visualization: timeseries — epochs as time axis)
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = range(1, len(history.history['loss']) + 1)
    ax.plot(epochs, history.history['loss'], label='Train loss')
    ax.plot(epochs, history.history['val_loss'], linestyle='--', label='Val loss')
    ax.set_title(f"{name} — Training Curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)


# ── 2. Tune the best model via random search with a live convergence chart ────
def tune_model(pipeline, model_name, param_grid, X_train, y_train,
               n_iter=20, random_state=12345, metric="rmse"):
    """
    Random hyperparameter search over param_grid.
    Uses a fixed 80/20 validation split of training data for speed.
    Skipped if the model class has no entry or an empty grid.
    Returns (best_pipeline, best_params).
    """
    from sklearn.model_selection import train_test_split as tts

    if not param_grid:
        print(f"\n[{model_name}] No hyperparameter grid — skipping tuning.")
        pipeline.fit(X_train, y_train)
        return pipeline, {}

    X_tr, X_vl, y_tr, y_vl = tts(X_train, y_train, test_size=0.2,
                                   random_state=random_state)

    # ── Live convergence chart ─────────────────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"{model_name} — Hyperparameter Search")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Validation {metric.upper()}")
    line_trial, = ax.plot([], [], "o", alpha=0.45, markersize=5, label=f"{metric.upper()} per trial")
    line_best,  = ax.plot([], [], "-", linewidth=2.5, label="Best so far")
    ax.legend()
    plt.tight_layout(); plt.pause(0.05)

    rng        = np.random.default_rng(random_state)
    param_keys = list(param_grid.keys())
    best_score   = np.inf if _lower_is_better(metric) else -np.inf
    best_params = {}
    best_pipe   = pipeline
    xs, trial_scores, best_curve = [], [], []

    for i in range(1, n_iter + 1):
        params   = {k: rng.choice(param_grid[k]).item() for k in param_keys}
        pipe_kw  = {f"model__{k}": v for k, v in params.items()}
        candidate = clone(pipeline)
        candidate.set_params(**pipe_kw)

        candidate.fit(X_tr, y_tr)
        pred = candidate.predict(X_vl)
        score = _calculate_score(y_vl, pred, metric)

        if _is_better(score, best_score, metric):
            best_score = score
            best_params = params
            best_pipe = candidate

        xs.append(i); trial_scores.append(score); best_curve.append(best_score)
        line_trial.set_data(xs, trial_scores)
        line_best.set_data(xs, best_curve)
        ax.relim(); ax.autoscale_view()
        ax.set_title(f"{model_name} — iter {i}/{n_iter}  best {metric.upper()}={best_score:,.4f}")
        fig.canvas.draw(); fig.canvas.flush_events(); plt.pause(0.001)

    plt.ioff(); plt.show(block=False)

    print(f"[{model_name}] Best params:          {best_params}")
    print(f"[{model_name}] Best validation {metric.upper()}: {best_score:,.4f}")

    # Re-fit winner on full training data
    best_pipe.fit(X_train, y_train)

    # Plot Keras training curve if the winner is a Neural Network
    final_step = best_pipe.steps[-1][1]
    if hasattr(final_step, 'history_') and final_step.history_ is not None:
        plot_keras_history(final_step.history_, name=model_name)

    return best_pipe, best_params


# ── 3. Evaluate on held-out test data ─────────────────────────────────────────
def evaluate_model(pipeline, X_test, y_test):
    """
    Apply a fitted pipeline to the held-out test split.
    Returns a metrics dict: RMSE, MSE, MAE, R². ✅
    (Regression-relevant subset of the full metrics checklist)
    """
    t0        = time.time()
    pred      = pipeline.predict(X_test)
    pred_time = time.time() - t0

    mse = float(mean_squared_error(y_test, pred))
    return {
        "rmse":      float(np.sqrt(mse)),
        "mse":       mse,
        "mae":       float(mean_absolute_error(y_test, pred)),
        "r2":        float(r2_score(y_test, pred)),
        "pred_time": pred_time,
    }


# ── 4. Save best model ────────────────────────────────────────────────────────
def save_best_model(pipeline, name, params, metrics, out_dir="models"):
    """
    Save the best pipeline (joblib) or Keras model + preprocessor to disk. ✅
    Also writes a metadata JSON.
    """
    os.makedirs(out_dir, exist_ok=True)
    final_step = pipeline.steps[-1][1]

    if isinstance(final_step, KerasRegressorWrapper):
        model_path = os.path.join(out_dir, "best_model.keras")
        final_step.model_.save(model_path)
        # Save preprocessor steps separately so the API can reload them
        prep_pipeline = clone(pipeline)
        # Just dump the whole pipeline for consistency
        joblib.dump(pipeline, os.path.join(out_dir, "best_pipeline.joblib"))
    else:
        model_path = os.path.join(out_dir, "best_model.joblib")
        joblib.dump(pipeline, model_path)

    metadata = {
        "name":       name,
        "model_path": model_path,
        "params":     params,
        "metrics":    metrics,
        "timestamp":  datetime.now().isoformat(),
    }
    meta_path = os.path.join(out_dir, "best_model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[save]  Best model saved → {model_path}")
    print(f"[save]  Metadata saved   → {meta_path}")


# ── 5. Print summary ──────────────────────────────────────────────────────────
def print_summary(results, best_name, best_params, test_metrics, metric="rmse"):
    """Print CV comparison table and final test metrics."""
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON SUMMARY  (5-fold CV)")
    print("=" * 72)
    for name, r in results.items():
        marker = "  ← best" if name == best_name else ""
        print(f"  {name:<26} CV {metric.upper()}={r['score']:>10,.4f}"
              f"  ±{np.std(r['fold_scores']):>6,.4f}"
              f"  train={r['train_time']:.1f}s"
              f"  mem={r['peak_mem_mb']:.1f}MB"
              f"{marker}")

    print(f"\n  [{best_name}] Test RMSE: {test_metrics['rmse']:>10,.4f}"
          f"  MSE: {test_metrics['mse']:>12,.2f}"
          f"  MAE: {test_metrics['mae']:>10,.4f}"
          f"  R²: {test_metrics['r2']:>7.4f}"
          f"  pred={test_metrics['pred_time']:.4f}s")

    if best_params:
        formatted = "  ".join(f"{k}={v}" for k, v in best_params.items())
        print(f"  [{best_name}] Best params: {formatted}")

    print("=" * 72)


# ── Metric Helpers ────────────────────────────────────────────────────────────
def _lower_is_better(metric):
    """Returns True if a lower score indicates a better model."""
    return metric.lower() in ['rmse', 'mse', 'mae']

def _is_better(score, best_score, metric):
    """Returns True if score is strictly better than best_score."""
    if _lower_is_better(metric):
        return score < best_score
    return score > best_score

def _get_best_model(results, metric):
    """Pick the model with the best cv score based on the chosen metric."""
    if _lower_is_better(metric):
        return min(results, key=lambda k: results[k]["score"])
    return max(results, key=lambda k: results[k]["score"])

def _calculate_score(y_true, y_pred, metric):
    """Compute score based on metric string."""
    m = metric.lower()
    if m == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    elif m == "mse":
        return float(mean_squared_error(y_true, y_pred))
    elif m == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    elif m == "r2":
        return float(r2_score(y_true, y_pred))
    else:
        raise ValueError(f"Unsupported metric: {metric}")
