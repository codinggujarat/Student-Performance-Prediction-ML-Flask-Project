# app.py
import os
import glob
import time
from typing import Dict, Any, List

from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Flask

# ---- Paths & Config ----
BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
REQUIRED_COLS = ["Student_ID", "Semester", "Attendance", "Study_Hours", "Past_Result"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'replace-this-with-a-real-secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ---- Utilities ----
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_exam_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Exam_Score and Risk using Attendance/Study_Hours/Past_Result.
    Adds/overwrites: Exam_Score, Predicted_Score, Risk
    Risk normalized to ('High', 'Low')
    """
    # Ensure needed columns exist
    missing = [c for c in ["Attendance", "Study_Hours", "Past_Result"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for scoring: {missing}")

    # numeric coercion
    for c in ["Attendance", "Study_Hours", "Past_Result"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    rng = np.random.default_rng(seed=42)  # deterministic-but-noisy for demo
    noise = rng.integers(-5, 6, size=len(df))  # [-5..5]

    exam = (
        df["Attendance"] * 0.3 +
        df["Study_Hours"] * 0.4 +
        df["Past_Result"] * 0.3 +
        noise
    )
    exam = np.clip(exam, 0, 100)
    df["Exam_Score"] = exam
    df["Predicted_Score"] = df["Exam_Score"]  # unify downstream
    df["Risk"] = np.where(df["Predicted_Score"] < 40, "High", "Low")
    return df


def normalize_semester_labels(series: pd.Series) -> pd.Series:
    """
    Convert semester-like values to 'Sem N' labels (keeps numeric order).
    """
    def _label(v):
        s = str(v)
        m = pd.Series(s).str.extract(r'(\d+)')[0].iloc[0]
        if pd.notna(m):
            return f"Sem {int(m)}"
        return s
    return series.apply(_label)


def to_py_list(arr: Any) -> List:
    """
    Convert numpy/pandas objects to plain Python lists with native types.
    Ensures JSON-serializable structures for Jinja |tojson.
    """
    if isinstance(arr, (pd.Series, pd.Index)):
        arr = arr.tolist()
    elif isinstance(arr, np.ndarray):
        arr = arr.tolist()

    def _py(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if pd.isna(v):
            return None
        return v

    if isinstance(arr, list):
        return [_py(v) for v in arr]
    return arr


def jsonify_counts(d: Dict[Any, Any]) -> Dict[str, int]:
    """
    Convert dict keys to strings and values to plain ints for JSON safety.
    """
    out = {}
    for k, v in d.items():
        key = str(k)
        if isinstance(v, (np.integer,)):
            out[key] = int(v)
        elif isinstance(v, (np.floating,)):
            out[key] = int(v)
        else:
            out[key] = int(v)
    return out


def build_chart_payload_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare all arrays/dicts the templates expect:
    - scores
    - risk_counts
    - show_semester_charts
    - semesters, avg_scores_list, high_counts, low_counts
    """
    # Ensure Predicted_Score & Risk exist (derive if needed)
    if "Predicted_Score" not in df.columns or "Risk" not in df.columns:
        df = calculate_exam_score(df)

    # Numeric coercion
    df["Predicted_Score"] = pd.to_numeric(df["Predicted_Score"], errors="coerce").fillna(0)
    if "Semester" in df.columns:
        df["Semester"] = df["Semester"].fillna("").astype(str)

    # Risk normalization to 'High'/'Low'
    def _norm_risk(v: Any) -> str:
        s = str(v).lower()
        if "high" in s:
            return "High"
        if "low" in s:
            return "Low"
        try:
            # numeric fallback
            val = float(v)
            return "High" if val < 40 else "Low"
        except Exception:
            return "Low"
    df["Risk"] = df["Risk"].apply(_norm_risk)

    # Top-level
    scores = to_py_list(df["Predicted_Score"])
    risk_counts = jsonify_counts(df["Risk"].value_counts().to_dict())

    # Semester charts
    show_semester_charts = False
    semesters: List[str] = []
    avg_scores_list: List[float] = []
    high_counts: List[int] = []
    low_counts: List[int] = []

    if "Semester" in df.columns and df["Semester"].notna().any():
        # normalize labels
        norm = normalize_semester_labels(df["Semester"])
        df = df.copy()
        df["__semester_label"] = norm
        # sort by numeric part
        def _sem_key(s: str) -> int:
            parts = [int(x) for x in str(s).split() if str(x).isdigit()]
            return parts[0] if parts else 999
        semesters_sorted = sorted(df["__semester_label"].unique().tolist(), key=_sem_key)

        grp = df.groupby("__semester_label", dropna=False)
        for sem in semesters_sorted:
            g = grp.get_group(sem)
            semesters.append(sem)
            avg_scores_list.append(round(float(g["Predicted_Score"].mean()), 2))
            high_counts.append(int((g["Risk"] == "High").sum()))
            low_counts.append(int((g["Risk"] == "Low").sum()))

        show_semester_charts = True

    return {
        "scores": scores,
        "risk_counts": risk_counts,
        "show_semester_charts": bool(show_semester_charts),
        "semesters": to_py_list(semesters),
        "avg_scores_list": to_py_list(avg_scores_list),
        "high_counts": to_py_list(high_counts),
        "low_counts": to_py_list(low_counts),
    }


# ---- Routes ----
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train')
def train():
    try:
        import train_model
        train_model.train_and_save()
        flash('Model trained and saved successfully.', 'success')
    except Exception as e:
        flash(f'Model training failed: {e}', 'danger')
    return redirect(url_for('index'))


@app.route('/download_sample')
def download_sample():
    sample = os.path.join('data', 'students_sample.csv')
    if not os.path.exists(sample):
        return 'Sample dataset not found. Run /train to generate it.', 404
    return send_file(sample, as_attachment=True, download_name='students_sample.csv')


@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(path):
        return 'File not found', 404
    return send_file(path, as_attachment=True, download_name=filename)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    Upload a CSV/XLSX. If raw columns are provided, auto-compute Predicted_Score/Risk.
    Saves results to /results and (optionally) shows a results page with table + charts.
    """
    if request.method == 'GET':
        return render_template('upload.html')

    file = request.files.get('file')
    if not file or file.filename == '':
        flash('No file selected', 'warning')
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash('Invalid file type. Use CSV or Excel.', 'warning')
        return redirect(request.url)

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Read file
    try:
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(save_path)
        else:
            df = pd.read_excel(save_path)
    except Exception as e:
        flash(f'Error reading file: {e}', 'danger')
        return redirect(request.url)

    # If user uploaded raw features only, synthesize predictions
    has_pred_cols = all(c in df.columns for c in ["Predicted_Score", "Risk"])
    has_req_cols = all(c in df.columns for c in REQUIRED_COLS)

    try:
        if not has_pred_cols and has_req_cols:
            df = calculate_exam_score(df)
        elif not has_pred_cols and not has_req_cols:
            flash('File does not contain prediction columns or required raw columns.', 'warning')
    except Exception as e:
        flash(f'Could not generate predictions: {e}', 'danger')

    # Save results to Excel in results/
    timestamp = int(time.time())
    out_name = f'predictions_{timestamp}.xlsx'
    out_path = os.path.join(RESULTS_FOLDER, out_name)
    try:
        df.to_excel(out_path, index=False)
    except Exception as e:
        flash(f'Error saving results: {e}', 'danger')

    # Prepare preview table
    preview_html = df.head(200).to_html(classes='table table-striped table-hover', index=False)

    # ---- Choose what to render after upload ----
    SHOW_CHARTS_AFTER_UPLOAD = True  # set False to keep your old 'results.html'

    if SHOW_CHARTS_AFTER_UPLOAD:
        payload = build_chart_payload_from_df(df)
        return render_template(
            "model_report.html",
            file_name=os.path.basename(out_name),
            **payload
        )

    # Old behavior: table + download link
    return render_template(
        'results.html',
        table_html=preview_html,
        download_url=url_for('download', filename=out_name)
    )


@app.route("/model_report")
def model_report():
    """
    Load latest predictions file from /results, normalize for charts, and render model_report.html
    """
    prediction_files = glob.glob(os.path.join(RESULTS_FOLDER, "predictions_*.xlsx")) + \
                       glob.glob(os.path.join(RESULTS_FOLDER, "predictions_*.csv"))

    if not prediction_files:
        flash("No predictions available. Upload a file first.", "warning")
        # Render with empty data so the template still loads
        return render_template(
            "model_report.html",
            file_name="(none)",
            scores=[],
            risk_counts={},
            show_semester_charts=False,
            semesters=[],
            avg_scores_list=[],
            high_counts=[],
            low_counts=[]
        )

    latest_file = max(prediction_files, key=os.path.getctime)
    try:
        if latest_file.lower().endswith(".csv"):
            df = pd.read_csv(latest_file)
        else:
            df = pd.read_excel(latest_file)
    except Exception as e:
        flash(f"Error reading latest predictions: {e}", "danger")
        return redirect(url_for("index"))

    payload = build_chart_payload_from_df(df)

    return render_template(
        "model_report.html",
        file_name=os.path.basename(latest_file),
        **payload
    )


if __name__ == '__main__':
    app.run(debug=True)
