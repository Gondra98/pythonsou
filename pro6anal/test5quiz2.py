import pandas as pd
import scipy.stats as stats
import pymysql

config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '123',
    'db': 'test',
    'port': 3306,
    'charset': 'utf8'
}

conn = pymysql.connect(**config)
query = "SELECT jikwonjik, jikwonpay FROM jikwon"
data = pd.read_sql(query, conn)
conn.close()

# NA 행 제외
df_jikwon = data.dropna(subset=['jikwonjik', 'jikwonpay'])

# (조건: 1000~2999:1, 3000~4999:2, 5000~6999:3, 7000~:4)
bins = [1000, 3000, 5000, 7000, 1000000] 
labels = [1, 2, 3, 4]
df_jikwon['pay_group'] = pd.cut(df_jikwon['jikwonpay'], bins=bins, labels=labels, right=False)

ctab_jik = pd.crosstab(index=df_jikwon['jikwonjik'], columns=df_jikwon['pay_group'])
chi2, p, dof, expected = stats.chi2_contingency(ctab_jik)

print(f"카이제곱: {chi2}")          # 카이제곱 : 37.40349394195548
print(f"p-value: {p}")              # p-value : 0.00019211533885350577

if p < 0.05:
    print(f"판정: p-value({p}) < 0.05 이므로 귀무가설을 기각 => 통계적으로 유의미")
else:
    print(f"판정: p-value({p}) >= 0.05 이므로 귀무가설을 채택 => 통계적 관련 없음")