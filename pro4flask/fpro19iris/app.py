from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
# matplotlib의 rendering engine 중 하나인 Agg : 이미지 저장시 오류 방지 - 차트 출력 없이 저장할 때 사용
import matplotlib.pyplot as plt
import seaborn as sns 
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent       # 현재 파일의 경로
STATIC_DIR = BASE_DIR / 'static' / 'images'

STATIC_DIR.mkdir(parents=True, exist_ok=True)

@app.get('/')
def main():
    return render_template('main.html')

@app.get('/showdata')
def showdata():
    df = sns.load_dataset('iris')
    # print(df.info())

    # pie chart 생성 및 저장
    counts = df['species'].value_counts().sort_index()
    plt.figure()
    counts.plot.pie(autopct='%1.1f%%', startangle=90, ylabel='')
    plt.tight_layout()

    img_path = STATIC_DIR / 'fpro19iris.png'
    plt.savefig(img_path, dpi=130)
    plt.close()

    iris_html = df.head().to_html(
        classes='table table-striped table-sm', index=False
    )


    return render_template(
        'show.html',
        table=iris_html,
        img_path=r"images/fpro19iris.png")

if __name__ == "__main__":
    app.run(debug=True)