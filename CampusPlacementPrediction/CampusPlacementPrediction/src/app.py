"""Module 5 - Web demo. Run this file, a browser page opens at http://localhost:8000

The page lets you type one student's details and shows the prediction of all three
algorithms, simple improvement hints, the model comparison table and the charts.
Uses only the Python standard library for the web part.
"""
import html
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

import config
import predict

CSS = """
body{font-family:Arial,Helvetica,sans-serif;margin:0;background:#f2f5fa;color:#1b2a57}
.head{background:#1b2a57;color:#fff;padding:14px 24px}
.head h1{margin:0;font-size:22px}.head p{margin:3px 0 0;font-size:12px;color:#c5cce0}
.wrap{padding:16px 24px}
table.layout{width:100%;border-collapse:separate;border-spacing:14px 0}
td.col{vertical-align:top;background:#fff;border:1px solid #dde3ee;padding:14px 18px}
h2{font-size:16px;margin:0 0 10px;color:#ff0066}
label{font-size:12px;color:#333}
input,select{width:100%;padding:5px;font-size:13px;border:1px solid #9aa3b5;margin-bottom:6px;box-sizing:border-box}
table.f{width:100%;border-collapse:collapse}
table.f td{padding:0 6px 0 0;width:50%;vertical-align:top}
.btn{background:#1b2a57;color:#fff;border:0;padding:9px 18px;font-size:14px;font-weight:bold;cursor:pointer}
a.s{font-size:12px;margin-right:12px;color:#0b4fd6}
.res{border:1px solid #dde3ee;margin-bottom:8px;padding:8px 10px}
.ok{border-left:6px solid #0e8c85}.no{border-left:6px solid #c62f4b}
.lab{font-weight:bold;font-size:14px}.bar{background:#e4e8f0;height:12px;margin-top:5px}
.fill{height:12px}.err{background:#fde8ec;border:1px solid #c62f4b;color:#8a1030;padding:8px;margin-bottom:10px;font-size:13px}
table.m{border-collapse:collapse;font-size:12px;width:100%}
table.m th{background:#1b2a57;color:#fff;padding:5px}table.m td{border:1px solid #dde3ee;padding:5px;text-align:center}
ul{margin:6px 0 0;padding-left:18px;font-size:12px}img{max-width:100%;border:1px solid #dde3ee}
"""


def pct(p):
    return ">99%" if p > 0.995 else ("<1%" if p < 0.005 else f"{p:.0%}")


def render_page(values=None, outcome=None, error=None):
    v = values or dict(predict.SAMPLE_PROFILES["strong"])
    fields = []
    for key in config.RANGES:
        lo, hi, label = config.RANGES[key]
        step = "1" if key in ("backlogs", "projects", "internships", "certifications") else "0.1"
        fields.append(f'<td><label>{html.escape(label)}</label><input name="{key}" type="number" '
                      f'step="{step}" min="{lo}" max="{hi}" value="{html.escape(str(v.get(key, "")))}"></td>')
    rows = "".join("<tr>" + "".join(fields[i:i + 2]) + "</tr>" for i in range(0, len(fields), 2))
    options = "".join(f'<option{" selected" if v.get("branch") == b else ""}>{html.escape(b)}</option>'
                      for b in config.BRANCHES)
    form = (f'<h2>Student details</h2><form method="post" action="/predict"><label>Branch</label>'
            f'<select name="branch">{options}</select><table class="f">{rows}</table><br>'
            f'<input class="btn" type="submit" value="Predict placement" style="width:auto">'
            f'&nbsp;&nbsp;<a class="s" href="/?profile=strong">Load strong profile</a>'
            f'<a class="s" href="/?profile=weak">Load weak profile</a></form>')

    right = "<h2>Prediction</h2>"
    if error:
        right += f'<div class="err">{html.escape(error)}</div>'
    if outcome:
        for name, r in outcome["results"].items():
            p = r["probability"]
            cls, word, color = ("ok", "PLACED", "#0e8c85") if r["placed"] else ("no", "NOT PLACED", "#c62f4b")
            right += (f'<div class="res {cls}"><span class="lab">{html.escape(name)}: '
                      f'<span style="color:{color}">{word}</span></span> &nbsp; chance of placement {pct(p)}'
                      f'<div class="bar"><div class="fill" style="width:{p * 100:.0f}%;background:{color}"></div></div></div>')
        right += f'<p style="font-size:13px"><b>{outcome["votes"]} of 3 models</b> predict that this student will be placed.</p>'
        if outcome["hints"]:
            right += "<b style='font-size:13px'>Where to improve</b><ul>" + "".join(
                f"<li>{html.escape(h)}</li>" for h in outcome["hints"]) + "</ul>"
    elif not error:
        right += '<p style="font-size:13px;color:#555">Fill the form and click <b>Predict placement</b>.</p>'

    metrics = ""
    mfile = config.REPORT_DIR / "metrics.csv"
    if mfile.exists():
        df = pd.read_csv(mfile)
        head = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
        body = "".join("<tr>" + "".join(
            f"<td>{x if isinstance(x, str) else format(x, '.3f')}</td>" for x in row) + "</tr>"
            for row in df.itertuples(index=False))
        metrics = (f'<h2 style="margin-top:14px">Model comparison (test set)</h2>'
                   f'<table class="m"><tr>{head}</tr>{body}</table>'
                   f'<p><img src="/reports/model_comparison.png" alt="comparison chart"></p>')
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Campus Placement Prediction</title>"
            f"<style>{CSS}</style></head><body><div class='head'><h1>Campus Placement Prediction Using Machine Learning</h1>"
            f"<p>K. Ramakrishnan College of Technology (Autonomous) | AGI1242 - Machine Learning</p></div>"
            f"<div class='wrap'><table class='layout'><tr><td class='col' style='width:48%'>{form}</td>"
            f"<td class='col'>{right}{metrics}</td></tr></table></div></body></html>")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            which = parse_qs(url.query).get("profile", [""])[0]
            self._send(200, render_page(predict.SAMPLE_PROFILES.get(which)))
        elif url.path.startswith("/reports/"):
            f = config.REPORT_DIR / Path(url.path).name
            if f.suffix == ".png" and f.exists():
                self._send(200, f.read_bytes(), "image/png")
            else:
                self._send(404, "Not found", "text/plain")
        else:
            self._send(404, "Not found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            self._send(404, "Not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode("utf-8")).items()}
        try:
            self._send(200, render_page(form, predict.predict_student(form)))
        except ValueError as err:
            self._send(200, render_page(form, error=str(err)))

    def log_message(self, fmt, *args):      # keep the Eclipse console clean
        pass


def start_server(port=8000):
    """Starts the server on the first free port from `port`; returns the server object."""
    predict.load_models()                   # trains first if models are missing
    for p in range(port, port + 15):
        try:
            return ThreadingHTTPServer(("127.0.0.1", p), Handler)
        except OSError:
            continue
    raise RuntimeError("No free port found")


def main():
    server = start_server()
    url = f"http://localhost:{server.server_address[1]}/"
    print(f"Placement prediction demo is running at {url}\nPress Ctrl+C (or the red stop button in Eclipse) to stop.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
