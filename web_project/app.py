from flask import Flask, render_template, request, redirect, session
import mariadb

app = Flask(__name__)
app.secret_key = "drone_secret_key"


# MariaDB 연결 함수
def get_db():
    conn = mariadb.connect(
        user="root",
        password="1234",
        host="localhost",
        port=3306,
        database="drone_db"
    )
    return conn


# 메인 페이지
@app.route("/")
def index():
    return render_template("index.html")


# 비행 안전 조회 페이지
@app.route("/flight")
def flight():
    return render_template("flight.html")


# 로그인 페이지
@app.route("/login")
def login():
    return render_template("login.html")


# 로그인 처리
@app.route("/login", methods=["POST"])
def login_process():

    user_id = request.form["id"]
    user_pw = request.form["pw"]

    conn = get_db()
    cursor = conn.cursor()

    sql = "SELECT * FROM users WHERE id=? AND pw=?"
    cursor.execute(sql, (user_id, user_pw))

    user = cursor.fetchone()

    conn.close()

    if user:
        session["user"] = user_id
        return redirect("/")
    else:
        return "로그인 실패"


# 로그아웃
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")


# 회원가입 페이지
@app.route("/signup")
def signup():
    return render_template("signup.html")


# 회원가입 처리
@app.route("/signup", methods=["POST"])
def signup_process():

    user_id = request.form["id"]
    user_pw = request.form["pw"]

    conn = get_db()
    cursor = conn.cursor()

    sql = "INSERT INTO users (id, pw) VALUES (?, ?)"
    cursor.execute(sql, (user_id, user_pw))

    conn.commit()
    conn.close()

    return redirect("/login")


# 마이페이지
@app.route("/mypage")
def mypage():

    if "user" not in session:
        return redirect("/login")

    return render_template("mypage.html", user=session["user"])


# 서버 실행
if __name__ == "__main__":
    app.run(debug=True)