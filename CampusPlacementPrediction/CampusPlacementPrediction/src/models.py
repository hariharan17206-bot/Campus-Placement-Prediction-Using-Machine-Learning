"""Module 2 - the three algorithms to compare, each with a small hyper-parameter grid."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import config
from data_preparation import build_preprocessor


def candidate_models():
    """name -> (estimator, parameter grid)"""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
            {"clf__C": [0.1, 1, 10]}),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=config.RANDOM_STATE),
            {"clf__max_depth": [3, 5, 7, None], "clf__min_samples_leaf": [1, 5, 10],
             "clf__criterion": ["gini", "entropy"]}),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1),
            {"clf__max_depth": [5, 10, None], "clf__min_samples_leaf": [1, 3, 5]}),
    }


def train_model(name, X_train, y_train, cv=5):
    """Tunes one algorithm with GridSearchCV and returns (fitted pipeline, best params, cv F1)."""
    estimator, grid = candidate_models()[name]
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", estimator)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_
