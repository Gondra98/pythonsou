# JSON 자료 : XML에 비해 경량, 배열 개념만 있으면 처리 가능 처리 가능
import json

dict = {'name':'tom','age':25, 'score':['90','80','88']}    # dict type
print(dict, type(dict))

print('json 인코딩 : dict -> str ---')
str_val = json.dumps(dict)
# str_val = json.dumps(dict, indent=4)
print(str_val, type(str_val))   # <class 'str'>
# print(str_val['name'])          # TypeError: string indices must be integers, not 'str'
print(str_val[0:20])    # 문자열 관련 함수만 사용 가능

print('json 디코딩 : dict -> dict ---')
json_val = json.loads(str_val)
print(json_val, type(json_val))     # <class 'dict'>
print(json_val['name'])             # dict 관련 명령 사용 가능

for k in json_val.keys():
    print(k)

for v in json_val.values():
    print(v)

print('\n서울시 제공 도서관 정보 JSON 샘플 자료(5개) 읽기 ---')
import urllib.request as req

url = "http://openapi.seoul.go.kr:8088/sample/json/SeoulLibraryTimeInfo/1/5/"
plainText = req.urlopen(url).read().decode()
print(plainText, type(plainText))   # <class 'str'>

jsonData = json.loads(plainText)
# print(jsonData, type(jsonData))     # <class 'dict'>
print(jsonData["SeoulLibraryTimeInfo"]["row"][0]["LBRRY_NAME"])

# dict의 get() 사용
print()
libData = jsonData.get("SeoulLibraryTimeInfo").get("row")
print(libData)                          # [{'LBRRY_SEQ_NO': '2413', 'LBRRY_NAME': 'BIBLIOTECA', 'GU_CODE': '0008', 'CODE_VALUE': '강남구', 'ADRES': '서울특별시 강남구 도산대로 25길 15-4 1층', 'FDRM_CLOSE_DATE': '휴관중', 'TEL_NO': '', 'XCNTS': '37.5207695', 'YDNTS': '127.0245651'}
name = libData[0].get('LBRRY_NAME')
print(name)


print()
datas = []
for ele in libData:
    name = ele.get('LBRRY_NAME')
    tel = ele.get('TEL_NO')
    addr = ele.get('ADRES')
    print(name, ' ', tel, ' ', addr)
    datas.append([name, tel, addr])

import pandas as pd
df = pd.DataFrame(datas, columns=['도서관명','전화','주소'])
print(df)
