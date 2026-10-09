"""Project settings shared by all modules (paths, feature names, random seed)."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent      # project root, works from Eclipse or terminal
DATA_FILE = BASE_DIR / "data" / "placement_data.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "placed"
NUMERIC_FEATURES = [
    "cgpa", "tenth_pct", "twelfth_pct", "backlogs", "aptitude_score",
    "coding_score", "communication_score", "projects", "internships", "certifications",
]
CATEGORICAL_FEATURES = ["branch"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
BRANCHES = ["CSE", "IT", "AI&DS", "ECE", "EEE", "MECH", "CIVIL"]

# Valid input range of every numeric feature: (min, max, label shown to the user)
RANGES = {
    "cgpa": (0.0, 10.0, "CGPA (0-10)"),
    "tenth_pct": (0.0, 100.0, "10th percentage"),
    "twelfth_pct": (0.0, 100.0, "12th percentage"),
    "backlogs": (0, 20, "Current backlogs"),
    "aptitude_score": (0.0, 100.0, "Aptitude score (0-100)"),
    "coding_score": (0.0, 100.0, "Coding score (0-100)"),
    "communication_score": (0.0, 10.0, "Communication (0-10)"),
    "projects": (0, 20, "Projects done"),
    "internships": (0, 10, "Internships done"),
    "certifications": (0, 30, "Certifications"),
}

MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest"]


def model_path(name: str) -> Path:
    return MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib")
