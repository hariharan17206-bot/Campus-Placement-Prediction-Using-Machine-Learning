"""Quick checks of the whole project. Run it before the demo; every line should say PASS."""
import threading
import urllib.parse
import urllib.request

import pandas as pd

import app
import config
import predict
from data_preparation import clean_data, get_splits, load_data

passed = failed = 0


def check(name, ok):
    global passed, failed
    print(("PASS  " if ok else "FAIL  ") + name)
    if ok:
        passed += 1
    else:
        failed += 1


def main():
    df = load_data()
    check("dataset has all columns", set(config.FEATURES + [config.TARGET]) <= set(df.columns))
    check("dataset has 2 classes", df[config.TARGET].nunique() == 2)
    clean, report = clean_data(df)
    check("cleaning keeps the rows", report["rows_out"] == len(clean) > 1000)
    X_train, X_test, y_train, y_test, _ = get_splits()
    check("80/20 stratified split", abs(len(X_test) / (len(X_train) + len(X_test)) - 0.2) < 0.01
          and abs(y_train.mean() - y_test.mean()) < 0.02)
    check("no test rows in training set", not set(X_train.index) & set(X_test.index))

    models = predict.load_models()
    check("three models loaded", set(models) == set(config.MODEL_NAMES))
    for name, m in models.items():
        acc = (m.predict(X_test) == y_test).mean()
        check(f"{name}: test accuracy {acc:.2f} > majority baseline",
              acc > max(y_test.mean(), 1 - y_test.mean()) + 0.05)

    strong = predict.predict_student(predict.SAMPLE_PROFILES["strong"])
    weak = predict.predict_student(predict.SAMPLE_PROFILES["weak"])
    check("strong profile is predicted as placed by all models", strong["votes"] == 3)
    check("weak profile is predicted as not placed by all models", weak["votes"] == 0)
    check("weak profile gets improvement hints", len(weak["hints"]) >= 3)
    for bad in [{"cgpa": 12}, {"branch": "XYZ"}]:
        profile = dict(predict.SAMPLE_PROFILES["strong"], **bad)
        try:
            predict.predict_student(profile)
            check(f"invalid input {bad} rejected", False)
        except ValueError:
            check(f"invalid input {bad} rejected", True)

    server = app.start_server(8700)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    page = urllib.request.urlopen(base + "/").read().decode()
    check("web page opens", "Campus Placement Prediction" in page and "Student details" in page)
    body = urllib.parse.urlencode(predict.SAMPLE_PROFILES["strong"]).encode()
    result = urllib.request.urlopen(base + "/predict", data=body).read().decode()
    check("web prediction shows all 3 models", all(n in result for n in config.MODEL_NAMES) and "PLACED" in result)
    body = urllib.parse.urlencode(dict(predict.SAMPLE_PROFILES["strong"], cgpa=15)).encode()
    check("web page shows input error", "must be between" in urllib.request.urlopen(base + "/predict", data=body).read().decode())
    img = urllib.request.urlopen(base + "/reports/model_comparison.png")
    check("chart image is served", img.headers["Content-Type"] == "image/png")
    server.shutdown()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
