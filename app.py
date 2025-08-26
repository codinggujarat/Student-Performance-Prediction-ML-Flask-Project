# app.py
import os
import time
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
import pandas as pd
import model_utils

BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'replace-this-with-a-real-secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')

    file = request.files.get('file')
    threshold = request.form.get('threshold', 40)
    try:
        threshold = float(threshold)
    except ValueError:
        threshold = 40

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

    # Predict
    try:
        results_df = model_utils.predict_df(df, threshold=threshold)
    except Exception as e:
        flash(f'Prediction error: {e}', 'danger')
        return redirect(request.url)

    # Save results to an excel in results/
    timestamp = int(time.time())
    out_name = f'predictions_{timestamp}.xlsx'
    out_path = os.path.join(RESULTS_FOLDER, out_name)
    results_df.to_excel(out_path, index=False)

    # Show a preview (first 200 rows) and link to download
    preview_html = results_df.head(200).to_html(classes='table table-striped table-hover', index=False)
    return render_template('results.html', table_html=preview_html, download_url=url_for('download', filename=out_name))


@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(path):
        return 'File not found', 404
    return send_file(path, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    app.run(debug=True)
