"""Run this file to train and compare Logistic Regression, Decision Tree and Random Forest.

It prints the comparison table, saves the three trained models in models/ and the charts in reports/.
"""
import time

import joblib
import pandas as pd
from sklearn.metrics import classification_report

import config
import evaluate
from data_preparation import clean_data, get_splits, load_data
from models import train_model


def main():
    print("=" * 66)
    print(" CAMPUS PLACEMENT PREDICTION  -  model training and comparison")
    print("=" * 66)
    X_train, X_test, y_train, y_test, report = get_splits()
    print(f"Students read          : {report['rows_in']}")
    print(f"Duplicates removed     : {report['duplicates_removed']}")
    print(f"Missing values (fixed) : {report['missing_values']}  -> median / mode imputation")
    print(f"Train / test split     : {len(X_train)} / {len(X_test)} (80 / 20, stratified)")
    print(f"Placed in training set : {y_train.mean():.1%}\n")

    clean_df, _ = clean_data(load_data())
    evaluate.plot_eda(clean_df)

    config.MODEL_DIR.mkdir(exist_ok=True)
    config.REPORT_DIR.mkdir(exist_ok=True)
    results, preds, probs, fitted, extra = {}, {}, {}, {}, {}
    text_report = []
    for name in config.MODEL_NAMES:
        t0 = time.time()
        print(f"Training {name} (5-fold grid search) ...", end=" ", flush=True)
        model, best_params, cv_f1 = train_model(name, X_train, y_train)
        seconds = time.time() - t0
        metrics, pred, prob = evaluate.compute_metrics(model, X_test, y_test)
        metrics["CV F1"] = cv_f1
        results[name], preds[name], probs[name], fitted[name] = metrics, pred, prob, model
        extra[name] = {"best_params": {k.replace("clf__", ""): v for k, v in best_params.items()},
                       "train_seconds": round(seconds, 1)}
        joblib.dump(model, config.model_path(name))
        text_report.append(f"=== {name} ===\nBest parameters: {extra[name]['best_params']}\n"
                           + classification_report(y_test, pred, target_names=["Not placed", "Placed"]))
        print(f"done ({seconds:.1f}s)")

    table = pd.DataFrame(results).T[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "CV F1"]]
    table.round(4).to_csv(config.REPORT_DIR / "metrics.csv", index_label="Model")
    (config.REPORT_DIR / "classification_reports.txt").write_text("\n".join(text_report), encoding="utf-8")

    evaluate.plot_comparison(results)
    evaluate.plot_confusion(preds, y_test)
    evaluate.plot_roc(probs, y_test)
    importance = evaluate.plot_importance(fitted["Random Forest"])

    print("\nRESULTS ON THE UNSEEN TEST SET")
    print("-" * 66)
    print(table.round(3).to_string())
    best = table["F1-score"].idxmax()
    best_cv = table["CV F1"].idxmax()
    print("-" * 66)
    print(f"Best on test set (F1)      : {best}  (F1 = {table.loc[best, 'F1-score']:.3f}, "
          f"accuracy = {table.loc[best, 'Accuracy']:.3f})")
    print(f"Best in cross-validation   : {best_cv}  (CV F1 = {table.loc[best_cv, 'CV F1']:.3f})")
    print("Top features (Random Forest): " + ", ".join(importance.head(4).index))
    print(f"\nModels saved in : {config.MODEL_DIR}")
    print(f"Charts saved in : {config.REPORT_DIR}")
    return table


if __name__ == "__main__":
    main()
