import os
import sys
import json
import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from datetime import datetime
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import joblib

from src import charts

# Suppress TensorFlow info/warning logs
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ── Silence helpers ───────────────────────────────────────────────────────────
_SILENCE = {
    "LGBMRegressor":     {"verbose": -1},
    "CatBoostRegressor": {"verbose": 0},
    "XGBRegressor":      {"verbosity": 0},
}


@contextmanager
def _devnull():
    """Redirect Python stdout/stderr to /dev/null within the block."""
    old_out, old_err = sys.stdout, sys.stderr
    try:
        with open(os.devnull, "w") as sink:
            sys.stdout = sink
            sys.stderr = sink
            yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


# ── Hyperparameter search spaces ──────────────────────────────────────────────
PARAM_GRIDS = {
    "LinearRegression":    {},
    "KerasRegressorWrapper": {},  # tuned via callbacks (early stopping, LR schedule)
    "LGBMRegressor": {
        "n_estimators":  list(range(50, 300, 10)),
        "max_depth":     list(range(2, 10)),
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
    },
    "RandomForestRegressor": {
        "n_estimators": list(range(50, 300, 10)),
        "max_depth":    list(range(2, 20)),
    },
    "CatBoostRegressor": {
        "n_estimators":  list(range(50, 200, 5)),
        "max_depth":     list(range(2, 8)),
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
    },
    "XGBRegressor": {
        "n_estimators":  list(range(50, 300, 10)),
        "max_depth":     list(range(2, 10)),
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
    },
}


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
def train_candidates(pipelines, X_train, y_train, k_folds=5, random_state=12345):
    """
    Evaluate each pipeline with KFold(k_folds) CV. ✅
    Tracks train time and peak memory per model. ✅
    Returns (results dict, best_name).
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    # ── Live comparison chart ─────────────────────────────────────────────────
    plt.ion()
    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 6))
    fig_cmp.patch.set_facecolor(charts.BG)
    charts.style_axes(ax_cmp, title="Training models…",
                      xlabel="Mean CV RMSE (lower is better)")
    plt.tight_layout()
    plt.pause(0.05)

    results     = {}
    names_done  = []
    scores_done = []
    errs_done   = []
    colors_done = []

    for idx, (name, pipeline) in enumerate(pipelines.items()):
        color = charts.COLORS[idx % len(charts.COLORS)]
        ax_cmp.set_title(f"Training: {name}…", color=charts.TEXT, fontsize=11)
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
            with _devnull():
                fold_pipe.fit(X_fold_tr, y_fold_tr)
                pred = fold_pipe.predict(X_fold_vl)

            fold_scores.append(float(np.sqrt(mean_squared_error(y_fold_vl, pred))))

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
        colors_done.append(color)

        # Update live bar chart
        ax_cmp.clear()
        charts.style_axes(ax_cmp, title="Model Comparison — Mean CV RMSE",
                          xlabel="RMSE (lower is better)")
        bars  = ax_cmp.barh(names_done, scores_done, color=colors_done,
                            edgecolor=charts.BORDER, height=0.5,
                            xerr=errs_done, error_kw={"ecolor": charts.MUTED, "capsize": 3})
        span  = max(scores_done) - min(scores_done) if len(scores_done) > 1 else scores_done[0]
        off   = span * 0.01 if span else scores_done[0] * 0.01
        for bar, val, err in zip(bars, scores_done, errs_done):
            ax_cmp.text(bar.get_width() + off,
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:,.0f} ±{err:,.0f}",
                        va="center", ha="left", color=charts.TEXT, fontsize=9)
        ax_cmp.set_yticks(range(len(names_done)))
        ax_cmp.set_yticklabels(names_done, color=charts.TEXT, fontsize=9)
        ax_cmp.invert_yaxis()
        plt.tight_layout(); fig_cmp.canvas.draw(); fig_cmp.canvas.flush_events()
        plt.pause(0.1)

    best_name = min(results, key=lambda k: results[k]["score"])

    # Highlight winner
    ax_cmp.clear()
    charts.style_axes(ax_cmp, title=f"CV Results — Winner: {best_name}",
                      xlabel="Mean CV RMSE (lower is better)")
    final_colors = ["#f5a623" if n == best_name else c
                    for n, c in zip(names_done, colors_done)]
    bars = ax_cmp.barh(names_done, scores_done, color=final_colors,
                       edgecolor=charts.BORDER, height=0.5,
                       xerr=errs_done, error_kw={"ecolor": charts.MUTED, "capsize": 3})
    span = max(scores_done) - min(scores_done)
    off  = span * 0.01 if span else scores_done[0] * 0.01
    for bar, val, err, n in zip(bars, scores_done, errs_done, names_done):
        label = f"{val:,.0f} ±{err:,.0f}  ← best" if n == best_name else f"{val:,.0f} ±{err:,.0f}"
        ax_cmp.text(bar.get_width() + off, bar.get_y() + bar.get_height() / 2,
                    label, va="center", ha="left", color=charts.TEXT, fontsize=9)
    ax_cmp.set_yticks(range(len(names_done)))
    ax_cmp.set_yticklabels(names_done, color=charts.TEXT, fontsize=9)
    ax_cmp.invert_yaxis()
    plt.tight_layout(); fig_cmp.canvas.draw(); fig_cmp.canvas.flush_events()
    plt.ioff(); plt.show(block=False)

    # ── Fold-score breakdown chart (training visualization — x=fold) ──────────
    plot_fold_scores(results)

    return results, best_name


def plot_fold_scores(results):
    """
    Plot per-fold RMSE for every model. X-axis = fold index. ✅
    (Training visualization: timeseries — iterations of learning)
    """
    n    = len(results)
    fig, axes = charts.new_figure(1, n, title="CV Fold Scores per Model",
                                  figsize=(n * 4, 4))
    axes_flat = np.array(axes).flatten()

    for ax, (name, r) in zip(axes_flat, results.items()):
        folds = list(range(1, len(r["fold_scores"]) + 1))
        ax.plot(folds, r["fold_scores"], "o-",
                color=charts.COLORS[0], linewidth=2, markersize=6)
        ax.axhline(r["score"], color=charts.COLORS[1],
                   linestyle="--", linewidth=1.5, label=f"mean={r['score']:,.0f}")
        charts.style_axes(ax, title=name, xlabel="Fold", ylabel="RMSE")
        ax.set_xticks(folds)
        ax.tick_params(colors=charts.MUTED)
        ax.legend(facecolor=charts.BG, labelcolor=charts.TEXT,
                  edgecolor=charts.BORDER, fontsize=8)

    plt.tight_layout()
    plt.show(block=False)


def plot_keras_history(history, name="KerasNN"):
    """
    Plot epoch vs loss and val_loss for a Keras training history. ✅
    (Training visualization: timeseries — epochs as time axis)
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(charts.BG)
    epochs = range(1, len(history.history['loss']) + 1)
    ax.plot(epochs, history.history['loss'],     color=charts.COLORS[0], label='Train loss')
    ax.plot(epochs, history.history['val_loss'], color=charts.COLORS[1],
            linestyle='--', label='Val loss')
    charts.style_axes(ax, title=f"{name} — Training Curve",
                      xlabel="Epoch", ylabel="MSE Loss")
    ax.legend(facecolor=charts.BG, labelcolor=charts.TEXT, edgecolor=charts.BORDER)
    plt.tight_layout()
    plt.show(block=False)


# ── 2. Tune the best model via random search with a live convergence chart ────
def tune_model(pipeline, model_name, X_train, y_train,
               n_iter=20, random_state=12345):
    """
    Random hyperparameter search over PARAM_GRIDS[model_name].
    Uses a fixed 80/20 validation split of training data for speed.
    Skipped if the model class has no entry or an empty grid.
    Returns (best_pipeline, best_params).
    """
    from sklearn.model_selection import train_test_split as tts

    param_grid = PARAM_GRIDS.get(model_name, {})
    if not param_grid:
        print(f"\n[{model_name}] No hyperparameter grid — skipping tuning.")
        pipeline.fit(X_train, y_train)
        return pipeline, {}

    X_tr, X_vl, y_tr, y_vl = tts(X_train, y_train, test_size=0.2,
                                   random_state=random_state)

    # ── Live convergence chart ─────────────────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(charts.BG)
    charts.style_axes(ax, title=f"{model_name} — Hyperparameter Search",
                      xlabel="Iteration", ylabel="Validation RMSE")
    line_trial, = ax.plot([], [], "o", alpha=0.45, color=charts.COLORS[0],
                          markersize=5, label="RMSE per trial")
    line_best,  = ax.plot([], [], "-",  color=charts.COLORS[1],
                          linewidth=2.5, label="Best so far")
    ax.legend(facecolor=charts.BG, labelcolor=charts.TEXT, edgecolor=charts.BORDER)
    plt.tight_layout(); plt.pause(0.05)

    rng        = np.random.default_rng(random_state)
    param_keys = list(param_grid.keys())
    best_rmse   = np.inf
    best_params = {}
    best_pipe   = pipeline
    xs, trial_rmses, best_curve = [], [], []

    for i in range(1, n_iter + 1):
        params   = {k: rng.choice(param_grid[k]).item() for k in param_keys}
        silence  = _SILENCE.get(model_name, {})
        pipe_kw  = {f"model__{k}": v for k, v in {**silence, **params}.items()}
        candidate = clone(pipeline)
        candidate.set_params(**pipe_kw)

        with _devnull():
            candidate.fit(X_tr, y_tr)
            pred = candidate.predict(X_vl)
        rmse = float(np.sqrt(mean_squared_error(y_vl, pred)))

        if rmse < best_rmse:
            best_rmse, best_params, best_pipe = rmse, params, candidate

        xs.append(i); trial_rmses.append(rmse); best_curve.append(best_rmse)
        line_trial.set_data(xs, trial_rmses)
        line_best.set_data(xs, best_curve)
        ax.relim(); ax.autoscale_view()
        ax.set_title(f"{model_name} — iter {i}/{n_iter}  "
                     f"best RMSE={best_rmse:,.0f}  params={best_params}",
                     color=charts.TEXT, fontsize=9)
        fig.canvas.draw(); fig.canvas.flush_events(); plt.pause(0.001)

    plt.ioff(); plt.show(block=False)

    print(f"[{model_name}] Best params:          {best_params}")
    print(f"[{model_name}] Best validation RMSE: {best_rmse:,.0f}")

    # Re-fit winner on full training data
    with _devnull():
        best_pipe.fit(X_train, y_train)

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
def print_summary(results, best_name, best_params, test_metrics):
    """Print CV comparison table and final test metrics."""
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON SUMMARY  (5-fold CV)")
    print("=" * 72)
    for name, r in results.items():
        marker = "  ← best" if name == best_name else ""
        print(f"  {name:<26} CV RMSE={r['score']:>10,.2f}"
              f"  ±{np.std(r['fold_scores']):>6,.0f}"
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
