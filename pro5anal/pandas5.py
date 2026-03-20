# pandas file i/o
import pandas as pd
import numpy as np

df = pd.read_csv('ex1.csv')
print(df, type(df))
print()
df = pd.read_table('ex1.csv', sep=',')
df = pd.read_table('ex1.csv', sep=',', skip_blank_lines=True)
# skip_blank_lines : 칼럼명, 데이터의 앞에 공백을 제거
print(df)
print()
pd.set_option('display.max_columns', None)   # 모든 칼럼 표시 옵션
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv')
print(df)
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', header=None)
print(df)
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', header=None, skiprows=1)
print(df)
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', header=None, names=['a','b','c','d','e'])
print(df)
print()
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt')
print(df)
df = pd.read_table('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt', sep='\s+')
print(df)
print(df.iloc[:, 0])
df = pd.read_table('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt', sep='\s+', skiprows=[1, 3])
print(df)
df = pd.read_fwf('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/data_fwt.txt', header=None, widths=(10, 3, 5), names=('data', 'name', 'price'), encoding='utf8')
print(df)
print(df.iloc[:, 0])
print(df['data'])

print('\nchunk : 대량의 데이터를 부분씩 메모리로 읽어 처리')
# 대용량 자료 로딩시 초과 오류 발생 방지 : 메모리를 절약
# 스트리밍 방식(일부만 순차 처리)으로 읽음
# 분산처리의 효과
# 여러 번 반복해 읽어야 하므로 속도는 느리다.
import time

n_rows = 10000
data = {
    'id':range(1, n_rows + 1),
    'name':[f'Student_{i}' for i in range(1, n_rows + 1)],
    'score1':np.random.randint(50, 101, size=n_rows),
    'score2':np.random.randint(50, 101, size=n_rows)
}
df = pd.DataFrame(data)
print(df.head())
print(df.tail(3))

csv_fname = 'students.csv'
df.to_csv(csv_fname, index=False)   # 파일 저장

print('-------------------------------------------')
# csv 파일 읽기 : 전체 한 번에 읽기
start_all = time.time()
df_all = pd.read_csv(csv_fname)
average_all_1 = df_all['score1'].mean()
average_all_2 = df_all['score2'].mean()
time_all = time.time() - start_all

# chunk로 읽기
chunk_size = 1000
total_score1 = 0
total_score2 = 0
total_count = 0
start_chunk_total = time.time()

for i, chunk in enumerate(pd.read_csv(csv_fname, chunksize=chunk_size)):
    start_chunk = time.time()
    # 청크 처리 중 첫번째 학생 정보 출력
    first_student = chunk.iloc[0]
    print(f"Chunk {i + 1} 첫번째 학생:ID={first_student['id']}, 이름={first_student['name']}", f"score1={first_student['score1']}", f"score2={first_student['score2']}")

    total_score1 += chunk['score1'].sum()
    total_score2 += chunk['score2'].sum()
    total_count += len(chunk)

    end_chunk = time.time()
    elapsed = end_chunk - start_chunk
    print(f"    처리 시간: {elapsed:7f}")   # 청크 단위 처리 시간


time_chunk_total = time.time() - start_chunk_total
average_chunk1 = total_score1 / total_count
average_chunk2 = total_score2 / total_count


print('\n처리 결과')
print(f'전체 학생 수 : {total_count}')
print(f"score1 총합 : {total_score1}, 평균 : {average_chunk1:3f}")
print(f"score2 총합 : {total_score1}, 평균 : {average_chunk2:3f}")
print(f"전체 한 번에 처리 시간 : {time_all:7f}초")
print(f"청크로 처리한 총 시간 : {time_chunk_total:7f}초")


# 청크 처리 시간 시각화
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
labels = ['전체 한번에 처리', '청크로 처리']
times = [time_all, time_chunk_total]

plt.figure(figsize=(6, 4))
bars = plt.bar(labels, times, color=['skyblue', 'red'])
for bar, time_val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{time_val:3f}초', ha='center', va='bottom', fontsize=10)

plt.ylabel('처리 시간(초)')
plt.grid(linestyle='--')
plt.tight_layout()
plt.show()

