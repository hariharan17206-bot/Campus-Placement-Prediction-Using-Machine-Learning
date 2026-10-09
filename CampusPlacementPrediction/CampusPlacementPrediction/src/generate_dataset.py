"""Creates a realistic SYNTHETIC campus placement dataset (data/placement_data.csv).

The data is generated from a hidden scoring rule plus random noise, so that the
three algorithms can be trained and compared without any real student records.
To use a real dataset, replace data/placement_data.csv with a file that has the
same column names (see config.py) and run train.py again.
"""
import numpy as np
import pandas as pd

import config


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate(n=1200, seed=config.RANDOM_STATE, target_rate=0.58):
    rng = np.random.default_rng(seed)
    branch = rng.choice(config.BRANCHES, n, p=[0.22, 0.14, 0.14, 0.18, 0.12, 0.12, 0.08])
    is_core_it = np.isin(branch, ["CSE", "IT", "AI&DS"])

    cgpa = np.clip(rng.normal(7.3, 0.9, n), 5.0, 9.8)
    tenth = np.clip(rng.normal(80, 8, n), 50, 99)
    twelfth = np.clip(rng.normal(77, 9, n), 50, 99)
    backlogs = rng.choice([0, 1, 2, 3, 4], n, p=[0.70, 0.15, 0.08, 0.04, 0.03])
    aptitude = np.clip(rng.normal(55 + (cgpa - 7.3) * 8, 15), 10, 100)
    coding = np.clip(rng.normal(42 + (cgpa - 7.3) * 6 + np.where(is_core_it, 14, 2), 19), 0, 100)
    comm = np.clip(rng.normal(6.0, 1.6, n), 1, 10)
    projects = np.clip(rng.poisson(1.8, n), 0, 6)
    internships = rng.choice([0, 1, 2, 3], n, p=[0.40, 0.35, 0.18, 0.07])
    certs = np.clip(rng.poisson(2.0, n), 0, 8)

    branch_effect = pd.Series(branch).map(
        {"CSE": 0.4, "IT": 0.4, "AI&DS": 0.4, "ECE": 0.1, "EEE": 0.0, "MECH": -0.3, "CIVIL": -0.4}).to_numpy()

    z = (1.1 * (cgpa - 7.0) + 0.035 * (aptitude - 55) + 0.04 * (coding - 45) + 0.25 * (comm - 6)
         + 0.35 * internships + 0.20 * projects + 0.12 * certs - 0.9 * backlogs
         + 0.012 * (twelfth - 77) + branch_effect
         + 0.0015 * np.maximum(0, coding - 60) * np.maximum(0, aptitude - 60))   # skills work together
    z = z - 1.5 * ((cgpa < 6.0) | (backlogs >= 3))                                # company eligibility cut-off
    z = z + rng.normal(0, 0.8, n)                                                 # luck / interview day

    lo, hi = -10.0, 10.0                    # choose the intercept so about 58 % are placed
    for _ in range(40):
        mid = (lo + hi) / 2
        if _sigmoid(z + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    placed = (rng.random(n) < _sigmoid(z + mid)).astype(int)

    df = pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
        "branch": branch,
        "cgpa": cgpa.round(2),
        "tenth_pct": tenth.round(1),
        "twelfth_pct": twelfth.round(1),
        "backlogs": backlogs,
        "aptitude_score": aptitude.round(1),
        "coding_score": coding.round(1),
        "communication_score": comm.round(1),
        "projects": projects,
        "internships": internships,
        "certifications": certs,
        "placed": placed,
    })
    # a few missing values, as in real records (handled later by the imputer)
    for col in ["aptitude_score", "communication_score", "certifications"]:
        idx = rng.choice(n, size=int(0.02 * n), replace=False)
        df[col] = df[col].astype(float)
        df.loc[idx, col] = np.nan
    return df


def main():
    df = generate()
    config.DATA_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(config.DATA_FILE, index=False)
    print(f"Dataset created: {config.DATA_FILE}  ({len(df)} students, placed = {df.placed.mean():.1%})")


if __name__ == "__main__":
    main()
