"""Module 4 - Prediction: validate one student's details and predict with the trained models."""
import joblib
import pandas as pd

import config
from data_preparation import clean_data, load_data

_cache = {}

SAMPLE_PROFILES = {
    "strong": {"branch": "CSE", "cgpa": 8.6, "tenth_pct": 92, "twelfth_pct": 90, "backlogs": 0,
               "aptitude_score": 82, "coding_score": 85, "communication_score": 8, "projects": 3,
               "internships": 2, "certifications": 4},
    "weak": {"branch": "CIVIL", "cgpa": 5.9, "tenth_pct": 62, "twelfth_pct": 60, "backlogs": 3,
             "aptitude_score": 35, "coding_score": 20, "communication_score": 4, "projects": 0,
             "internships": 0, "certifications": 0},
}


def load_models():
    """Loads (and caches) the trained models; trains them first if they do not exist."""
    if not all(config.model_path(n).exists() for n in config.MODEL_NAMES):
        import train
        train.main()
    for name in config.MODEL_NAMES:
        if name not in _cache:
            _cache[name] = joblib.load(config.model_path(name))
    return _cache


def validate(profile):
    """Checks every field; returns a clean dict or raises ValueError with a readable message."""
    clean = {}
    branch = str(profile.get("branch", "")).strip()
    if branch not in config.BRANCHES:
        raise ValueError("Branch must be one of: " + ", ".join(config.BRANCHES))
    clean["branch"] = branch
    for col, (lo, hi, label) in config.RANGES.items():
        raw = profile.get(col, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{label}: please enter a number.")
        if not (lo <= value <= hi):
            raise ValueError(f"{label} must be between {lo} and {hi}.")
        clean[col] = value
    return clean


def _hints(profile):
    """Simple explanations: which skills are below the median of placed students."""
    df, _ = clean_data(load_data())
    median = df[df[config.TARGET] == 1][config.NUMERIC_FEATURES].median()
    hints = []
    if profile["backlogs"] > 0:
        hints.append(f"Clear your {int(profile['backlogs'])} backlog(s) - many companies reject students with backlogs")
    for col in ["cgpa", "coding_score", "aptitude_score", "communication_score",
                "internships", "projects", "certifications"]:
        if profile[col] < median[col] * 0.92:
            hints.append(f"{config.RANGES[col][2]}: yours {profile[col]:g}, placed students' median {median[col]:.1f}")
    return hints[:5]


def predict_student(profile):
    """Returns {'results': {model: {'placed': bool, 'probability': float}}, 'hints': [...]}"""
    clean = validate(profile)
    row = pd.DataFrame([clean])[config.FEATURES]
    results = {}
    for name, model in load_models().items():
        prob = float(model.predict_proba(row)[0, 1])
        results[name] = {"placed": prob >= 0.5, "probability": prob}
    votes = sum(r["placed"] for r in results.values())
    return {"results": results, "votes": votes, "hints": _hints(clean), "profile": clean}


def main():
    """Command-line demo: type the details of one student."""
    print("Enter the student's details (press Enter to use the example value).")
    base = SAMPLE_PROFILES["strong"]
    profile = {}
    for key in ["branch"] + list(config.RANGES):
        label = "Branch (" + "/".join(config.BRANCHES) + ")" if key == "branch" else config.RANGES[key][2]
        value = input(f"{label} [{base[key]}]: ").strip()
        profile[key] = value or base[key]
    try:
        out = predict_student(profile)
    except ValueError as err:
        print("Input error:", err)
        return
    print()
    for name, r in out["results"].items():
        print(f"{name:20s}: {'PLACED' if r['placed'] else 'NOT PLACED':10s} ({r['probability']:.1%})")
    for h in out["hints"]:
        print(" - improve ->", h)


if __name__ == "__main__":
    main()
