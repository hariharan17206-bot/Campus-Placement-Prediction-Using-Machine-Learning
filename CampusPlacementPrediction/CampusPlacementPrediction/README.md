# Campus Placement Prediction Using Machine Learning
Course: AGI1242 - Machine Learning

Predicts whether a student will be placed from CGPA, 10th/12th %, backlogs, aptitude score,
coding score, communication, projects, internships, certifications and branch.
Compares **Logistic Regression, Decision Tree and Random Forest** and gives a live web demo.

## 1. One-time setup
* Install Python 3.9+ and run:  `pip install -r requirements.txt`
* Eclipse: install the **PyDev** plugin (Help > Eclipse Marketplace > search "PyDev").
  Window > Preferences > PyDev > Interpreters > Python Interpreter > Quick Auto-Config.

## 2. Open in Eclipse
1. File > Import > General > **Existing Projects into Workspace** > select this folder (or the zip) > Finish.
2. Open `src/train.py` > right click > **Run As > Python Run**. The Eclipse Console shows the comparison table
   (about 20 seconds) and the models/charts are saved.
3. Open `src/app.py` > **Run As > Python Run**. Your browser opens the demo page (http://localhost:8000).
4. Optional: run `src/self_test.py` - every line should say PASS.

No Eclipse? From a terminal:  `cd src`  then  `python train.py`  and  `python app.py`

## 3. Demo script (about 4 minutes)
1. Run `train.py`: explain the cleaning lines, then the table of the three algorithms.
2. Open the charts in the `reports` folder (model_comparison, confusion_matrices, roc_curves, feature_importance, eda).
3. Run `app.py`. Click **Load strong profile** > Predict. All three models say PLACED.
4. Click **Load weak profile** > Predict. NOT PLACED, with "Where to improve" hints.
5. Change one value (for example coding score from 20 to 80) and predict again to show the probability change.

## 4. Dataset
`data/placement_data.csv` is a **synthetic** dataset of 1200 students created by `src/generate_dataset.py`
(with random noise and about 2 % missing values). To use a real dataset, replace the CSV with a file that has the
same column names (see `src/config.py`) and run `train.py` again.

## 5. Files (src)
config.py - settings | generate_dataset.py - makes the data | data_preparation.py - Module 1 |
models.py - Module 2 (the 3 algorithms + grid search) | evaluate.py - Module 3 (metrics, charts) |
predict.py - Module 4 | app.py - Module 5 (web demo) | train.py - runs the whole pipeline | self_test.py - checks
