"""Module 1 - Data Preparation: load, clean, split and build the preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def load_data(path=config.DATA_FILE):
    if not path.exists():
        import generate_dataset
        generate_dataset.main()
    return pd.read_csv(path)


def clean_data(df):
    """Remove duplicate students and impossible values; returns (clean_df, report dict)."""
    report = {"rows_in": len(df)}
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "student_id"]).copy()
    report["duplicates_removed"] = report["rows_in"] - len(df)
    for col, (lo, hi, _) in config.RANGES.items():
        bad = ((df[col] < lo) | (df[col] > hi)) & df[col].notna()
        df.loc[bad, col] = float("nan")          # impossible value -> treated as missing
    report["missing_values"] = int(df[config.FEATURES].isna().sum().sum())
    report["rows_out"] = len(df)
    return df, report


def build_preprocessor():
    """Median-impute + scale numbers, mode-impute + one-hot encode the branch."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, config.NUMERIC_FEATURES),
                              ("cat", categorical, config.CATEGORICAL_FEATURES)])


def get_splits():
    """Returns X_train, X_test, y_train, y_test (stratified 80/20) and the cleaning report."""
    df, report = clean_data(load_data())
    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    return X_train, X_test, y_train, y_test, report
