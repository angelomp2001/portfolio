import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.config import TARGET_COL, MODELS
from src.data_preprocessing import (
    load_data, save_stats, save_charts, clean_data, 
    data_types, column_preprocessor
)
from src.model_training import (
    train_model, tune_model, save_model, test_model, save_results
)

def main(
    sample_size: int | None = None,
    k_folds: int = 5,
    random_state: int = 42,
    metric: str = "f1",
):
    # 1. Load and explore data
    df_raw = load_data(sample=sample_size)
    save_stats(df_raw, label="raw")
    save_charts(df_raw, label="raw")

    # 2. Clean data
    df = clean_data(df_raw)
    save_stats(df, label="clean")
    save_charts(df, label="clean")

    # 3. Train / test split
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    # 4. Data types for preprocessing
    num_cols, cat_cols = data_types(X)

    # 5. Build pipelines, run CV, collect results
    pipelines: dict[str, Pipeline] = {}
    hyperparam_grids: dict[str, dict] = {}
    cv_results: dict[str, dict] = {}

    for name, model, is_tree, grid in MODELS:
        preprocessor = column_preprocessor(
            num_cols=num_cols,
            cat_cols=cat_cols,
            is_tree=is_tree,
        )

        pipelines[name] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        hyperparam_grids[name] = grid

        cv_results[name] = train_model(
            pipeline=pipelines[name],
            X_train=X_train,
            y_train=y_train,
            k_folds=k_folds,
            random_state=random_state,
            metric=metric,
            param_grid=hyperparam_grids[name],
        )

    # 6. Pick best model by chosen metric
    best_model_name = max(
        cv_results,
        key=lambda name: cv_results[name][metric]
    )
    best_pipeline = pipelines[best_model_name]

    # 7. Optionally: hyperparameter tuning for the best model
    if hyperparam_grids[best_model_name]:
        best_pipeline_tuned = tune_model(
            pipeline=best_pipeline,
            X_train=X_train,
            y_train=y_train,
            metric=metric,
            param_grid=hyperparam_grids[best_model_name],
        )
    else:
        best_pipeline_tuned = best_pipeline

    # 8. Fit best pipeline on full training data
    best_pipeline_tuned.fit(X_train, y_train)

    # 9. Save model + CV results
    save_model(
        model=best_pipeline_tuned,
        model_name=best_model_name,
        metadata={
            "metric": metric,
            "cv_results": cv_results[best_model_name],
            "random_state": random_state,
            "k_folds": k_folds,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        },
    )

    # 10. Final test evaluation
    test_results = test_model(
        model=best_pipeline_tuned,
        X_test=X_test,
        y_test=y_test,
        metric=metric,
    )

    # 11. Save test results (and optionally all CV results)
    save_results(
        results={
            "test": test_results,
            "cv": cv_results,
            "best_model_name": best_model_name,
            "metric": metric,
        },
        model_name=best_model_name,
    )


if __name__ == "__main__":
    from src.config import RANDOM_STATE
    main(random_state=RANDOM_STATE, metric="accuracy")
