@echo off
python -m venv venv
call venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
