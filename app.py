# app.py
import os
import glob
import time
from typing import Any, Dict, List

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_file,
    flash,
)
from werkzeug.utils import secure_filename
import pandas as pd

# optional: used in demo fallback scoring if model missing (not required if using model_utils)
import numpy as np

# use your model utils (the updated version you already installed)
import model_utils

# Use non-GUI backend for matplotlib (not strictly used here but kept for compatibility)
import matplotlib

matplotlib.use("Agg")

# ---- Paths & Config ----
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace-this-with-a-real-secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---- Small helpers to guarantee JSON-serializable primitives ----
def to_py_list(arr) -> List:
    """Convert pandas / numpy types to plain Python list with native types."""
    if arr is None:
        return []
    if hasattr(arr, "tolist"):
        arr = arr.tolist()
    # now ensure native python types
    def _py(v):
        # pandas NA handling
        try:
            import numpy as _np

            if isinstance(v, (_np.integer,)):
                return int(v)
            if isinstance(v, (_np.floating,)):
                return float(v)
        except Exception:
            pass
        # pandas NaN to None
        try:
            import pandas as _pd

            if _pd.isna(v):
                return None
        except Exception:
            pass
        return v

    if isinstance(arr, list):
        return [_py(v) for v in arr]
    return arr


def jsonify_counts(d: Dict[Any, Any]) -> Dict[str, int]:
    """Ensure dict values are plain ints and keys strings."""
    out = {}
    for k, v in (d or {}).items():
        key = str(k)
        # cast numpy ints/floats
        try:
            import numpy as _np

            if isinstance(v, (_np.integer,)):
                out[key] = int(v)
            elif isinstance(v, (_np.floating,)):
                out[key] = int(v)
            else:
                out[key] = int(v)
        except Exception:
            try:
                out[key] = int(v)
            except Exception:
                out[key] = 0
    return out


# ---- File utilities ----
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---- Routes ----
@app.route("/")
def index():
    """Dashboard home."""
    return render_template("index.html")


@app.route("/train")
def train():
    """Train model (delegates to train_model.train_and_save)."""
    try:
        import train_model

        train_model.train_and_save()
        flash("Model trained and saved successfully.", "success")
    except Exception as e:
        flash(f"Model training failed: {e}", "danger")
    return redirect(url_for("index"))


@app.route("/download_sample")
def download_sample():
    """Return the sample CSV produced by training helper (if present)."""
    sample = os.path.join(DATA_FOLDER, "students_sample.csv")
    if not os.path.exists(sample):
        flash("Sample dataset not found. Run Train to generate it.", "warning")
        return redirect(url_for("index"))
    return send_file(sample, as_attachment=True, download_name="students_sample.csv")


@app.route("/download/<filename>")
def download(filename):
    """Download a result file saved under results/"""
    path = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(path):
        flash("File not found.", "warning")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name=filename)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """
    Upload CSV/XLSX. If file contains required feature columns it will:
      - call model_utils.predict_df (which loads your trained model)
      - save predictions to results/predictions_<timestamp>.xlsx
      - show the preview table and provide a download link
    """
    if request.method == "GET":
        return render_template("upload.html")

    file = request.files.get("file")
    threshold = request.form.get("threshold", 40)

    try:
        threshold = float(threshold)
    except Exception:
        threshold = 40.0

    if not file or file.filename == "":
        flash("No file selected", "warning")
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash("Invalid file type. Use CSV or Excel.", "warning")
        return redirect(request.url)

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # Read file
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(save_path)
        else:
            df = pd.read_excel(save_path)
    except Exception as e:
        flash(f"Error reading file: {e}", "danger")
        return redirect(request.url)

    # Try to predict using trained model
    try:
        # model_utils.predict_df returns DataFrame with Predicted_Score, Risk, Risk_Label, Model_Used
        results_df = model_utils.predict_df(df, threshold=threshold)
    except FileNotFoundError as fnf:
        # If model missing, give helpful message (user can train)
        flash(str(fnf), "warning")
        # As fallback, we can attempt to synthesize scores if the raw features exist
        fallback_ok = all(c in df.columns for c in ["Attendance", "Study_Hours", "Past_Result"])
        if fallback_ok:
            # lightweight fallback calculation (same as earlier demo)
            try:
                df_c = df.copy()
                df_c["Attendance"] = pd.to_numeric(df_c["Attendance"], errors="coerce").fillna(0)
                df_c["Study_Hours"] = pd.to_numeric(df_c["Study_Hours"], errors="coerce").fillna(0)
                df_c["Past_Result"] = pd.to_numeric(df_c["Past_Result"], errors="coerce").fillna(0)
                rng = np.random.default_rng(seed=int(time.time()) % 2**32)
                noise = rng.integers(-5, 6, size=len(df_c))
                df_c["Predicted_Score"] = (
                    df_c["Attendance"] * 0.3 + df_c["Study_Hours"] * 0.4 + df_c["Past_Result"] * 0.3 + noise
                ).clip(0, 100).round(2)
                df_c["Risk"] = df_c["Predicted_Score"] < threshold
                df_c["Risk_Label"] = df_c["Risk"].map({True: "HIGH", False: "LOW"})
                df_c["Model_Used"] = "fallback"
                results_df = df_c
                flash("Used a fallback scorer because a model file was missing. Train model for better predictions.", "info")
            except Exception as e:
                flash(f"Fallback scoring failed: {e}", "danger")
                return redirect(request.url)
        else:
            return redirect(url_for("index"))
    except ValueError as ve:
        # validation errors from model_utils (missing columns / non-numeric)
        flash(f"Prediction error: {ve}", "danger")
        return redirect(request.url)
    except Exception as e:
        flash(f"Unexpected prediction error: {e}", "danger")
        return redirect(request.url)

    # Save results to Excel in results/
    timestamp = int(time.time())
    out_name = f"predictions_{timestamp}.xlsx"
    out_path = os.path.join(RESULTS_FOLDER, out_name)
    try:
        results_df.to_excel(out_path, index=False)
    except Exception as e:
        flash(f"Error saving results file: {e}", "danger")
        return redirect(request.url)

    # Prepare preview table (first 200 rows)
    preview_html = results_df.head(200).to_html(classes="table  bg-purple-400/20", index=False)

    # Render results preview page (with download link)
    return render_template("results.html", table_html=preview_html, download_url=url_for("download", filename=out_name))


@app.route("/model_report")
def model_report():
    """
    Load latest predictions from results/ (predictions_*.xlsx/.csv),
    prepare JSON-safe arrays for Chart.js and render model_report.html.
    """
    # find prediction files (both xlsx and csv)
    prediction_files = (
        glob.glob(os.path.join(RESULTS_FOLDER, "predictions_*.xlsx"))
        + glob.glob(os.path.join(RESULTS_FOLDER, "predictions_*.csv"))
    )

    if not prediction_files:
        flash("No predictions available. Upload a file first.", "warning")
        # Render page with empty data so the template loads
        return render_template(
            "model_report.html",
            file_name="(none)",
            scores=[],
            risk_counts={},
            show_semester_charts=False,
            semesters=[],
            avg_scores_list=[],
            high_counts=[],
            low_counts=[],
        )

    latest_file = max(prediction_files, key=os.path.getctime)

    # read latest file
    try:
        if latest_file.lower().endswith(".csv"):
            df = pd.read_csv(latest_file)
        else:
            df = pd.read_excel(latest_file)
    except Exception as e:
        flash(f"Error reading latest predictions: {e}", "danger")
        return redirect(url_for("index"))

    # Ensure we have columns (try common column names)
    # Accept Predicted_Score or Exam_Score / Predicted_Score
    if "Predicted_Score" not in df.columns and "Exam_Score" in df.columns:
        df = df.rename(columns={"Exam_Score": "Predicted_Score"})

    # If no Predicted_Score present, attempt to compute predictions (fallback)
    if "Predicted_Score" not in df.columns:
        try:
            # Use model_utils.predict_df if model and features present
            df = model_utils.predict_df(df)
        except Exception:
            # Last resort: try simple calculation if raw features exist
            if all(c in df.columns for c in ["Attendance", "Study_Hours", "Past_Result"]):
                df = df.copy()
                df["Attendance"] = pd.to_numeric(df["Attendance"], errors="coerce").fillna(0)
                df["Study_Hours"] = pd.to_numeric(df["Study_Hours"], errors="coerce").fillna(0)
                df["Past_Result"] = pd.to_numeric(df["Past_Result"], errors="coerce").fillna(0)
                df["Predicted_Score"] = (
                    df["Attendance"] * 0.3 + df["Study_Hours"] * 0.4 + df["Past_Result"] * 0.3
                ).clip(0, 100).round(2)
                df["Risk"] = df["Predicted_Score"] < 40
                df["Risk_Label"] = df["Risk"].map({True: "HIGH", False: "LOW"})
                df["Model_Used"] = "fallback"
            else:
                flash("Latest results file doesn't contain scores and cannot be interpreted.", "danger")
                return redirect(url_for("index"))

    # normalize risk labels (High/Low)
    if "Risk" in df.columns:
        # accept boolean or label strings
        if df["Risk"].dtype == bool:
            df["Risk"] = df["Risk"].map({True: "High", False: "Low"})
        else:
            df["Risk"] = df["Risk"].astype(str).apply(lambda s: "High" if "high" in s.lower() else "Low")

    # Prepare chart payload (JSON-safe)
    # Scores (list)
    scores = to_py_list(df["Predicted_Score"] if "Predicted_Score" in df.columns else df.get("Exam_Score", []))

    # Risk counts (dict)
    risk_counts = jsonify_counts(df["Risk"].value_counts().to_dict() if "Risk" in df.columns else {})

    # Semester-based charts (if present)
    show_semester_charts = "Semester" in df.columns and df["Semester"].notna().any()
    semesters = []
    avg_scores_list = []
    high_counts = []
    low_counts = []

    if show_semester_charts:
        # normalize and sort semester values (keep numeric order when possible)
        def sem_key(s):
            # attempt to extract integer
            import re

            m = re.search(r"(\d+)", str(s))
            if m:
                return int(m.group(1))
            return 999

        sem_series = df["Semester"].fillna("").astype(str)
        unique_sems = sorted(set(sem_series.tolist()), key=sem_key)
        for sem in unique_sems:
            sem_df = df[sem_series == sem]
            if sem_df.empty:
                continue
            semesters.append(sem)
            avg_scores_list.append(round(float(sem_df["Predicted_Score"].mean()), 2))
            high_counts.append(int((sem_df["Risk"] == "High").sum()))
            low_counts.append(int((sem_df["Risk"] == "Low").sum()))

    # cast into python-native structures for Jinja's tojson
    ctx = {
        "file_name": os.path.basename(latest_file),
        "scores": to_py_list(scores),
        "risk_counts": risk_counts,
        "show_semester_charts": bool(show_semester_charts),
        "semesters": to_py_list(semesters),
        "avg_scores_list": to_py_list(avg_scores_list),
        "high_counts": to_py_list(high_counts),
        "low_counts": to_py_list(low_counts),
    }

    return render_template("model_report.html", **ctx)


if __name__ == "__main__":
    app.run(debug=True)
