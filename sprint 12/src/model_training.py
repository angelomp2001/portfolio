# libraries
import os
import sys
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from contextlib import contextmanager
from sklearn.metrics import mean_squared_error

matplotlib.use("TkAgg")   # interactive backend; falls back gracefully if unavailable

# ── Silence helpers ───────────────────────────────────────────────────────────
# Per-class kwargs that suppress console output at the Python API level.
_SILENCE = {
    "LGBMRegressor":     {"verbose": -1},
    "CatBoostRegressor": {"verbose": 0},
    "XGBRegressor":      {"verbosity": 0},
}


@contextmanager
def _devnull():
    """
    Redirect Python-level stdout/stderr to /dev/null for the duration of the block.
    Catches any remaining print() calls made by model libraries.
    """
    old_out, old_err = sys.stdout, sys.stderr
    try:
        with open(os.devnull, "w") as sink:
            sys.stdout = sink
            sys.stderr = sink
            yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


# ── Chart style helpers ───────────────────────────────────────────────────────
_BG     = "#1a1a2e"
_PANEL  = "#16213e"
_TEXT   = "#e0e0f0"
_MUTED  = "#aaaacc"
_BORDER = "#444466"
_COLORS = ["#6ec6f5", "#f5a623", "#50fa7b", "#ff79c6", "#bd93f9"]


def _style_axes(ax, title="", xlabel="", ylabel=""):
    """Apply the dark theme to an Axes object."""
    ax.set_facecolor(_PANEL)
    ax.set_title(title, color=_TEXT, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=_MUTED, fontsize=10)
    ax.tick_params(colors=_MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(_BORDER)


# ── Hyperparameter search spaces ──────────────────────────────────────────────
# Models with an empty dict are recognised but skipped (no tuning step).
PARAM_GRIDS = {
    "LinearRegression": {},
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


# ── 1. Train & compare all candidate models ───────────────────────────────────
def train_candidates(candidates):
    """
    Fit every model in `candidates`, score it on its validation split, and
    display a live horizontal bar chart that grows as each model finishes.

    Parameters
    ----------
    candidates : dict[str, dict]
        Keys are display names; values are dicts with:
          'model', 'X_train', 'y_train', 'X_valid', 'y_valid'

    Returns
    -------
    results   : dict[str, dict]  – RMSE, train_time, pred_time per model
    best_name : str              – key of the model with the lowest validation RMSE
    """
    # ── Live bar chart setup ──────────────────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(_BG)
    _style_axes(ax, title="Training models…", xlabel="Validation RMSE (lower is better)")
    plt.tight_layout()
    plt.pause(0.05)

    results      = {}
    names_done   = []
    rmses_done   = []
    colors_done  = []

    for idx, (name, spec) in enumerate(candidates.items()):
        model   = spec["model"]
        X_train = spec["X_train"]
        y_train = spec["y_train"]
        X_valid = spec["X_valid"]
        y_valid = spec["y_valid"]
        color   = _COLORS[idx % len(_COLORS)]

        # Show which model is currently training
        ax.set_title(f"Training: {name}…", color=_TEXT, fontsize=12)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.05)

        # Fit — silence all console output from the library
        t0 = time.time()
        with _devnull():
            model.fit(X_train, y_train)
        train_time = time.time() - t0

        # Predict on validation
        t0 = time.time()
        with _devnull():
            pred = model.predict(X_valid)
        pred_time = time.time() - t0

        rmse = np.sqrt(mean_squared_error(y_valid, pred))
        results[name] = {
            "model":      model,
            "rmse":       rmse,
            "train_time": train_time,
            "pred_time":  pred_time,
        }

        names_done.append(name)
        rmses_done.append(rmse)
        colors_done.append(color)

        # Redraw bar chart with each new result
        ax.clear()
        _style_axes(ax,
                    title="Model Comparison — Validation RMSE",
                    xlabel="RMSE (lower is better)")
        bars = ax.barh(names_done, rmses_done, color=colors_done,
                       edgecolor=_BORDER, height=0.5)
        span = max(rmses_done) - min(rmses_done) if len(rmses_done) > 1 else rmses_done[0]
        offset = span * 0.01 if span else rmses_done[0] * 0.01
        for bar, val in zip(bars, rmses_done):
            ax.text(bar.get_width() + offset,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:,.0f}",
                    va="center", ha="left", color=_TEXT, fontsize=9)
        ax.set_yticks(range(len(names_done)))
        ax.set_yticklabels(names_done, color=_TEXT, fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)

    # Final pass: highlight the winner in gold
    best_name = min(results, key=lambda k: results[k]["rmse"])
    ax.clear()
    _style_axes(ax,
                title=f"Model Comparison — Winner: {best_name}",
                xlabel="Validation RMSE (lower is better)")
    final_colors = [
        "#f5a623" if n == best_name else c
        for n, c in zip(names_done, colors_done)
    ]
    bars = ax.barh(names_done, rmses_done, color=final_colors,
                   edgecolor=_BORDER, height=0.5)
    span   = max(rmses_done) - min(rmses_done)
    offset = span * 0.01 if span else rmses_done[0] * 0.01
    for bar, val, n in zip(bars, rmses_done, names_done):
        label = f"{val:,.0f}  ← best" if n == best_name else f"{val:,.0f}"
        ax.text(bar.get_width() + offset,
                bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", color=_TEXT, fontsize=9)
    ax.set_yticks(range(len(names_done)))
    ax.set_yticklabels(names_done, color=_TEXT, fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.ioff()
    plt.show(block=False)   # keep open; plt.show(block=True) in main.py holds all windows

    return results, best_name


# ── 2. Tune the best model via random search with a live convergence chart ────
def tune_model(model, X_train, y_train, X_valid, y_valid,
               n_iter=20, random_state=42):
    """
    Random hyperparameter search against PARAM_GRIDS with a live RMSE
    convergence chart.  Skipped if the model class has no entry or an empty
    grid in PARAM_GRIDS.

    Returns
    -------
    best_model  : estimator re-fitted on X_train with the best found params
    best_params : dict of winning hyperparameter values (empty dict if skipped)
    """
    model_class = type(model)
    model_name  = model_class.__name__
    param_grid  = PARAM_GRIDS.get(model_name, {})

    if not param_grid:
        print(f"\n[{model_name}] No hyperparameter grid defined — skipping tuning.")
        return model, {}

    # ── Live convergence chart setup ──────────────────────────────────────────
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(_BG)
    _style_axes(ax,
                title=f"{model_name} — Hyperparameter Search",
                xlabel="Iteration",
                ylabel="Validation RMSE")
    line_trial, = ax.plot([], [], "o", alpha=0.45, color="#6ec6f5",
                          markersize=5, label="RMSE per trial")
    line_best,  = ax.plot([], [], "-",  color="#f5a623",
                          linewidth=2.5, label="Best so far")
    ax.legend(facecolor=_BG, labelcolor=_TEXT, edgecolor=_BORDER)
    plt.tight_layout()
    plt.pause(0.05)

    # ── Random search loop ────────────────────────────────────────────────────
    rng        = np.random.default_rng(random_state)
    param_keys = list(param_grid.keys())

    best_rmse   = np.inf
    best_params = {}
    best_model  = model
    xs, trial_rmses, best_curve = [], [], []

    for i in range(1, n_iter + 1):
        # Sample one random combination
        params  = {k: rng.choice(param_grid[k]).item() for k in param_keys}
        silence = _SILENCE.get(model_name, {})

        candidate = model_class(**{**silence, **params})
        with _devnull():
            candidate.fit(X_train, y_train)
            pred = candidate.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, pred))

        if rmse < best_rmse:
            best_rmse   = rmse
            best_params = params
            best_model  = candidate

        xs.append(i)
        trial_rmses.append(rmse)
        best_curve.append(best_rmse)

        # Update chart
        line_trial.set_data(xs, trial_rmses)
        line_best.set_data(xs, best_curve)
        ax.relim()
        ax.autoscale_view()
        ax.set_title(
            f"{model_name} — iter {i}/{n_iter}  "
            f"best RMSE={best_rmse:,.0f}  params={best_params}",
            color=_TEXT, fontsize=10,
        )
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    plt.ioff()
    plt.show(block=False)   # keep chart open; main.py calls plt.show(block=True) at exit

    print(f"[{model_name}] Best params:          {best_params}")
    print(f"[{model_name}] Best validation RMSE: {best_rmse:,.0f}")

    # Re-fit on full training data with the best found params
    silence = _SILENCE.get(model_name, {})
    final   = model_class(**{**silence, **best_params})
    with _devnull():
        final.fit(X_train, y_train)

    return final, best_params


# ── 3. Evaluate on held-out test data ─────────────────────────────────────────
def evaluate_model(model, X_test, y_test):
    """
    Apply a fitted model to the held-out test split.

    Returns
    -------
    rmse      : float
    pred_time : float  (seconds)
    """
    t0        = time.time()
    pred      = model.predict(X_test)
    pred_time = time.time() - t0
    rmse      = np.sqrt(mean_squared_error(y_test, pred))
    return rmse, pred_time


# ── 4. Print summary table ────────────────────────────────────────────────────
def print_summary(results, best_name, best_params=None,
                  test_rmse=None, test_pred_time=None):
    """
    Print a formatted comparison table, the final test result, and the
    best hyperparameters.  All values are discovered at runtime — nothing
    is hard-coded.
    """
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 65)
    for name, r in results.items():
        marker = "  ← best" if name == best_name else ""
        print(
            f"  {name:<26} RMSE={r['rmse']:>10,.2f}"
            f"  train={r['train_time']:.3f}s"
            f"  pred={r['pred_time']:.4f}s"
            f"{marker}"
        )

    if test_rmse is not None:
        print(f"\n  [{best_name}] Test RMSE:           {test_rmse:>10,.4f}"
              f"  pred time: {test_pred_time:.4f}s")

    if best_params:
        formatted = "  ".join(f"{k}={v}" for k, v in best_params.items())
        print(f"  [{best_name}] Best hyperparameters: {formatted}")

    print("=" * 65)
