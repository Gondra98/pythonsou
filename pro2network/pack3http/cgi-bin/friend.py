# -*- coding: utf-8 -*-   
# 위 명령 안 먹으면 아래 방법 사용
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import urllib.parse

# --- get / past 요청 받을 때 ---------
method = os.environ.get("REQUEST_METHOD", "GET")

if method == "POST":
    length = int(os.environ.get("CONTENT_LENGTH", 0))
    body = sys.stdin.read(length)
else:
    body = os.environ.get("QUERY_STRING", "")

params = urllib.parse.parse_qs(body)

# 값 꺼내기  - 첫 번째 값 꺼내기 [0]
irum = params.get("name", [""])[0] # 없으면 빈 리스트 대신 [""] 사용
phone = params.get("phone", [""])[0]
gen = params.get("gen", [""])[0]

print("Content-Type: text/html; charset=utf-8")
print()
print("""
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>friend</title>
</head> 
<body>
    입력한 값 : 이름은 {0}, 전화번호는 {1}, 성별은 {2}
</body>
</html>
""".format(irum, phone, gen))