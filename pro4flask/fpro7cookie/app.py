from flask import Flask, render_template, request, make_response, redirect, url_for

app = Flask("__name__")

# Cookie는 브라우저에 저장되는 작은 키-값 데이터이고, 서버가 클라이언트와 연결 유지
# 유지하는 것 처럼 할 수 있다. 
# 서버가 설정 -> 브라우저가 저장 -> 다음 요청부터 브라우저가 자동으로 함께 전송
COOKIE_AGE = 60 * 60 * 24 * 7

@app.get("/")
def home():
    return render_template("index.html")


@app.get("/login")
def loginfunc():
    name = request.cookies.get("name")
    visits = request.cookies.get("visits")

    if name:
        visits = int(visits or "0") + 1
        msg = f"안녕하세요. {name}님 {visits}번째 방문입니다"
    else:
        visits = None
        msg = "이름을 입력하면 방문 횟수를 쿠키로 기억합니다"

    resp = make_response(render_template("login.html", msg=msg, name=name, visits=visits))

    if name:
        resp.set_cookie("visits", str(visits), max_age=COOKIE_AGE, samesite="Lax")

    return resp


@app.post("/login")
def loginfunc2():
    name = request.form.get(("name") or "").strip()
    resp = make_response(redirect(url_for("loginfunc")))
    resp.set_cookie("name", name, max_age=COOKIE_AGE, samesite="Lax")
    resp.set_cookie("visits", "0", max_age=COOKIE_AGE, samesite="Lax")

    return resp


@app.post("/logout")
def loginfunc2():
    # 쿠키 삭제 후 /login(get)으로 이동
    resp = make_response(redirect(url_for("loginfunc")))
    resp.delete_cookie("name")
    resp.delete_cookie("visits")

    return resp


if __name__ == "__main__":
    app.run(debug=True)