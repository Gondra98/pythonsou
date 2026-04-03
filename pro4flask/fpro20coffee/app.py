from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for
from db import insert_survey, fetchall_survey

BASE_DIR = Path(__file__).resolve().parent
IMG_PATH = BASE_DIR / 'static' / 'images' / 'vbar.png'

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/coffee/survey")
def survey_view():
    return render_template("coffee/coffeesurvey.html")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
