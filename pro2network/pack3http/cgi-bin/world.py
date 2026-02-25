s1 = "자료1"
s2 = "두번째 자료"

print("Content-Type:text/html;charset = utf-8")

print("""
<html lang="kr">
<head>
    <title>메인</title>
</head>
<body>
    <h1>world 페이지</h1>
    자료 출력 : {0}, {1}
    <br/>
    <img src="../images/dog.jpeg" />
    <br/>
    <a href="../index.html">메인으로</a>
</body>
</html>
    """.format(s1, s2))

