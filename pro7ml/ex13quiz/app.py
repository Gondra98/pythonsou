from flask import Flask, render_template, request, jsonify
import pymysql
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import date

app = Flask(__name__)

def get_conn():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="123",
        database="test",
        charset="utf8"
    )

def get_model_and_data():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT jikwonibsail, jikwonpay FROM jikwon")
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    today = date.today()
    X, y = [], []
    for ibsail, pay in rows:
        # 입사일로부터 근무년수 계산
        years = (today - ibsail).days / 365
        X.append([round(years, 1)])
        y.append(pay)

    X = np.array(X)
    y = np.array(y)

    model = LinearRegression()
    model.fit(X, y)
    return model, X, y

model, X, y = get_model_and_data()

@app.get("/")
def home():
    return render_template("main.html")

@app.post("/api/predict")
def predict():
    data = request.get_json()
    years = float(data["years"])

    salary = model.predict([[years]])[0]
    r2 = model.score(X, y) * 100
    coef = model.coef_[0]
    intercept = model.intercept_

    # 직급별 평균 연봉
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT jikwonjik, ROUND(AVG(jikwonpay))
            FROM jikwon
            GROUP BY jikwonjik
        """)
        avg_rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    avg_list = [{"직급": row[0], "평균연봉": int(row[1])} for row in avg_rows]

    return jsonify({
        "salary": round(salary),
        "r2": round(r2, 2),
        "coef": round(coef, 4),
        "intercept": round(intercept, 4),
        "avg_by_rank": avg_list
    })

if __name__ == "__main__":
    app.run(debug=True)