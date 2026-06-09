"""
TensorFlow DQN 기반 창고 물류 로봇 예제
이 소스 코드는 강화학습의 대표 알고리즘 중 하나인 DQN을 이용하여, 창고 안의 물류 로봇이 상품을 찾아 출고 지점까지 배송하는 과정을 학습하는 예제다.
로봇은 격자 형태의 창고 환경에서 움직이며, 상품 위치로 이동한 뒤 상품을 집고, 출고 지점으로 이동하여 상품을 내려놓는 것이 목표다.
코드에는 창고 환경 정의, DQN 모델 생성, 경험 저장소인 Replay Buffer, 행동 선택 방식인 epsilon-greedy, 학습 과정, 학습 결과 시각화, 평가 실행 기능이 포함되어 있습니다. 업로드된 코드도 WarehouseRobotEnv, ReplayBuffer, create_dqn_model, select_action, train_step, train_dqn, evaluate_agent로 기능이 나뉘어 구성되어 있다.

1. 전체 코드의 목적 : 로봇이 다음 순서를 스스로 학습하도록 만드는 것.
  1) 시작 위치에서 출발한다.
  2) 장애물을 피하면서 상품 위치로 이동한다.
  3) 상품 위치에서 상품을 집는다.
  4) 출고 지점으로 이동한다.
  5) 출고 지점에서 상품을 내려놓는다.
즉, 단순히 최단 거리로 이동하는 문제가 아니라, 상품을 집은 상태와 집지 않은 상태를 구분하면서 올바른 행동 순서를 학습하는 문제다.

2. 창고 환경 구조 : 창고는 5x5 격자 공간으로 구성되어 있다.
 R . . . .
 . X . X .
 . . P . .
 . X . . .
 . . . . D

기호	의미
 R	 로봇 위치
 R*	 상품을 들고 있는 로봇
 P	 상품 위치
 D	 출고 지점
 X	 장애물
 .	 빈 공간

"""


# !pip install gymnasium tensorflow numpy matplotlib

# TensorFlow/Keras 기반 창고 물류 로봇 DQN 예제
"""
창고 물류 로봇 환경
 목표: 로봇이 상품 위치로 이동 -> 상품 집기 -> 출고 지점으로 이동 -> 상품 내려놓기
 행동 -- 0:위,  1:아래,  2:왼쪽,  3:오른쪽,  4:상품 집기,  5:상품 내려놓기
"""

import random
import time
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

# 1. 창고 물류 로봇 환경  # 강화학습에서 에이전트가 상호작용할 환경을 정의하는 부분
class WarehouseRobotEnv(gym.Env):  # Gymnasium의 Env 클래스를 상속받아 사용자 정의 환경 생성
    metadata = {"render_modes": ["human"]}  # 환경을 사람이 볼 수 있는 방식으로 출력할 수 있음을 표시

    def __init__(self, grid_size=5, max_steps=100):  # 환경 객체가 생성될 때 실행되는 초기화 함수
        super().__init__()

        self.grid_size = grid_size  # 창고 격자의 크기를 저장, 기본값은 5x5
        self.max_steps = max_steps  # 한 에피소드에서 허용할 최대 행동 횟수 저장

        # 행동 공간: 위, 아래, 왼쪽, 오른쪽, 집기, 내려놓기
        self.action_space = spaces.Discrete(6)  # 행동이 총 6개인 이산 행동 공간 정의

        # 상태: [로봇 row, 로봇 col, 상품 보유 여부]
        self.observation_space = spaces.Box(  # 상태 공간을 연속값 범위로 정의
            low=0.0,  # 상태값의 최소값을 0.0으로 설정
            high=1.0,  # 상태값의 최대값을 1.0으로 설정
            shape=(3,),  # 상태는 3개의 값으로 구성됨
            dtype=np.float32  # 상태값의 자료형을 float32로 설정
        )  # 상태 공간 정의 종료

        # 위치 설정
        self.start_pos = (0, 0)  # 로봇의 시작 위치를 0행 0열로 설정
        self.product_pos = (2, 2)  # 상품이 놓여 있는 위치를 2행 2열로 설정
        self.dropoff_pos = (4, 4)  # 상품을 내려놓을 출고 지점을 4행 4열로 설정

        # 장애물 위치  : 로봇이 지나갈 수 없는 위치를 정의
        self.obstacles = {(1, 1), (1, 3), (3, 1)}  # 장애물 좌표들을 집합 자료형으로 저장

        self.robot_pos = None  # 현재 로봇 위치를 저장할 변수, reset에서 실제 위치가 설정됨
        self.has_product = False  # 로봇이 상품을 들고 있는지 여부, 처음에는 False
        self.step_count = 0  # 현재 에피소드에서 수행한 행동 횟수

    def reset(self, seed=None, options=None):  # 에피소드 시작 시 환경을 초기 상태로 되돌리는 함수
        super().reset(seed=seed)

        self.robot_pos = self.start_pos  # 로봇 위치를 시작 위치로 초기화
        self.has_product = False
        self.step_count = 0

        return self._get_obs(), {}  # 초기 상태와 추가 정보를 반환

    def _get_obs(self):  # 현재 환경 상태를 DQN 모델 입력 형태로 반환하는 내부 함수
        """
        DQN에 입력할 상태를 반환한다.
        row, col은 0~1 사이 값으로 정규화한다.
        has_product는 상품을 들고 있으면 1, 아니면 0이다.
        """
        row, col = self.robot_pos  # 현재 로봇 위치를 row와 col로 분리

        state = np.array([
            row / (self.grid_size - 1),  # 행 위치를 0~1 사이 값으로 정규화
            col / (self.grid_size - 1),  # 열 위치를 0~1 사이 값으로 정규화
            1.0 if self.has_product else 0.0  # 상품을 들고 있으면 1.0, 아니면 0.0
        ], dtype=np.float32)

        return state  # 현재 상태 배열 반환

    def step(self, action):  # 에이전트가 선택한 행동을 환경에 적용하는 함수
        self.step_count += 1  # 행동을 한 번 수행했으므로 step 수를 1 증가

        reward = 0  # 이번 행동으로 받을 보상을 0으로 초기화
        terminated = False  # 목표 달성으로 에피소드가 끝났는지 여부
        truncated = False  # 최대 step 초과로 에피소드가 중단됐는지 여부

        row, col = self.robot_pos  # 현재 로봇 위치를 row, col로 분리

        # 이동 행동  # 행동이 상하좌우 이동인 경우 처리
        if action in [0, 1, 2, 3]:
            reward -= 1  # 이동할 때마다 이동 비용으로 -1 보상 부여
            next_row, next_col = row, col  # 이동 후 위치를 계산하기 위해 현재 위치를 복사

            if action == 0:      # 위
                next_row -= 1
            elif action == 1:    # 아래
                next_row += 1
            elif action == 2:    # 왼쪽
                next_col -= 1
            elif action == 3:    # 오른쪽
                next_col += 1

            # 벽 충돌  # 이동하려는 위치가 격자 밖인지 검사
            if not self._is_inside_grid(next_row, next_col):  # 격자 범위를 벗어나면
                reward -= 5
            # 장애물 충돌  - 이동하려는 위치가 장애물인지 검사
            elif (next_row, next_col) in self.obstacles:  # 다음 위치가 장애물 위치라면
                reward -= 5
            # 정상 이동  - 벽도 아니고 장애물도 아닌 경우
            else:
                self.robot_pos = (next_row, next_col)  # 로봇의 현재 위치를 다음 위치로 변경

        # 상품 집기
        elif action == 4:  # 행동 번호 4는 상품 집기
            if self.robot_pos == self.product_pos and not self.has_product:  # 상품 위치에 있고 아직 상품을 들고 있지 않다면
                self.has_product = True  # 상품을 들고 있는 상태로 변경
                reward += 10  # 상품 집기 성공 보상 +10 부여
            else:  # 상품 위치가 아니거나 이미 상품을 들고 있다면
                reward -= 10  # 잘못된 집기 행동이므로 -10 패널티 부여

        # 상품 내려놓기  # 행동이 상품 내려놓기인 경우 처리
        elif action == 5:  # 행동 번호 5는 상품 내려놓기
            if self.robot_pos == self.dropoff_pos and self.has_product:  # 출고 지점에 있고 상품을 들고 있다면
                self.has_product = False  # 상품을 내려놓았으므로 상품 보유 여부를 False로 변경
                reward += 30  # 배송 성공 보상 +30 부여
                terminated = True  # 목표를 달성했으므로 에피소드 종료
            else:  # 출고 지점이 아니거나 상품을 들고 있지 않다면
                reward -= 10  # 잘못된 내려놓기 행동이므로 -10 패널티 부여

        # 최대 step 초과  # 너무 오래 움직인 경우 에피소드 종료 처리
        if self.step_count >= self.max_steps:  # 현재 step 수가 최대 step 이상이면
            truncated = True  # 시간 초과로 에피소드 중단 처리

        next_state = self._get_obs()  # 행동 후의 다음 상태를 가져옴

        return next_state, reward, terminated, truncated, {}  # 다음 상태, 보상, 종료 여부, 중단 여부, 추가 정보 반환

    def _is_inside_grid(self, row, col):  # 좌표가 격자 안에 있는지 확인하는 함수
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size  # 행과 열이 모두 격자 범위 안이면 True 반환

    def render(self):  # 현재 창고 상태를 콘솔에 출력하는 함수
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]  # 빈 칸으로 채워진 2차원 격자 생성

        # 장애물 표시  # 장애물 위치를 화면 표시용 격자에 반영
        for r, c in self.obstacles:  # 장애물 좌표들을 하나씩 꺼냄
            grid[r][c] = "X"  # 해당 위치에 장애물 표시 X 입력

        # 상품 표시  # 상품 위치를 화면 표시용 격자에 반영
        if not self.has_product:  # 로봇이 상품을 들고 있지 않은 경우에만
            pr, pc = self.product_pos  # 상품 위치 좌표를 분리
            grid[pr][pc] = "P"  # 상품 위치에 P 표시

        # 출고 지점 표시  # 출고 위치를 화면 표시용 격자에 반영
        dr, dc = self.dropoff_pos  # 출고 지점 좌표를 분리
        grid[dr][dc] = "D"  # 출고 지점에 D 표시

        # 로봇 표시  # 로봇 위치를 화면 표시용 격자에 반영
        rr, rc = self.robot_pos  # 현재 로봇 위치 좌표를 분리
        grid[rr][rc] = "R*" if self.has_product else "R"  # 상품을 들고 있으면 R*, 아니면 R 표시

        print("\n현재 창고 상태")
        print(f"step: {self.step_count}, has_product: {self.has_product}")  # 현재 step 수와 상품 보유 여부 출력

        for row in grid:  # 격자의 각 행을 하나씩 반복
            print(" ".join(f"{cell:2}" for cell in row))  # 각 칸을 보기 좋게 정렬해서 출력

        print()


# 2. Replay Buffer : DQN은 과거 경험을 저장해두고,무작위로 뽑아서 학습한다.
class ReplayBuffer:  # 경험 데이터를 저장하고 샘플링하는 클래스
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)  # 최대 capacity 개수만큼 경험을 저장하는 큐 생성

    def push(self, state, action, reward, next_state, done):  # 하나의 경험을 저장하는 함수
        self.buffer.append(
            (state, action, reward, next_state, done)  # 현재 상태, 행동, 보상, 다음 상태, 종료 여부를 하나의 튜플로 저장
        )

    def sample(self, batch_size):  # 버퍼에서 무작위로 batch_size만큼 경험을 뽑는 함수
        batch = random.sample(self.buffer, batch_size)  # 저장된 경험 중 무작위로 batch_size개 선택
        states, actions, rewards, next_states, dones = zip(*batch)  # 경험 묶음을 항목별로 분리

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)  # 종료 여부를 float32 배열로 변환, True는 1.0, False는 0.0처럼 사용

        return states, actions, rewards, next_states, dones  # 학습에 사용할 배치 데이터 반환

    def __len__(self):
        return len(self.buffer)


# 3. DQN 모델 생성 함수 : 상태를 입력받아 각 행동의 Q값을 출력하는 신경망
def create_dqn_model(state_dim, action_dim):  # 상태 차원 수와 행동 개수 입력
    """
    입력: [robot_row, robot_col, has_product]
    출력:  행동 6개에 대한 Q값
    """
    model = models.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(action_dim)
    ])

    return model


# 4. epsilon-greedy 행동 선택
def select_action(state, model, epsilon, action_dim):  # 현재 상태, 모델, epsilon 값, 행동 개수를 입력받음
    # epsilon 확률로 랜덤 행동을 선택하고, 1 - epsilon 확률로 Q값이 가장 큰 행동을 선택한다.
    if random.random() < epsilon:
        return random.randrange(action_dim)

    state_input = np.expand_dims(state, axis=0)  # 모델 입력 형태에 맞게 상태 차원을 하나 추가
    q_values = model(state_input, training=False)  # 현재 상태에 대한 각 행동의 Q값 예측
    action = np.argmax(q_values.numpy()[0])
    return int(action)


# 5. DQN 한 번 학습
def train_step(policy_model, target_model, optimizer,  # 학습 모델, 타깃 모델, 최적화 도구 입력
               states, actions, rewards, next_states, dones, gamma):  # 배치 데이터와 할인율 입력
    # DQN의 핵심 학습 단계 : target = reward + gamma * max(Q(next_state))
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)  # 행동 배열을 TensorFlow 텐서로 변환
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)  # 보상 배열을 TensorFlow 텐서로 변환
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)  # 종료 여부 배열을 TensorFlow 텐서로 변환

    states = tf.convert_to_tensor(states, dtype=tf.float32)  # 현재 상태 배열을 TensorFlow 텐서로 변환
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)  # 다음 상태 배열을 TensorFlow 텐서로 변환

    with tf.GradientTape() as tape:  # 자동 미분을 기록하기 위한 GradientTape 시작
        # 현재 상태의 Q값  # policy_model이 예측한 현재 상태의 행동별 Q값 계산
        q_values = policy_model(states, training=True)  # 현재 상태 배치를 입력해 Q값 예측

        # 선택한 action에 해당하는 Q값만 추출  # 여러 행동 Q값 중 실제 수행한 행동의 Q값만 뽑기 위한 인덱스 생성
        action_indices = tf.stack(  # gather_nd에 사용할 2차원 인덱스 생성
            [tf.range(tf.shape(actions)[0]), actions],  # 배치 번호와 행동 번호를 묶음
            axis=1  # 열 방향으로 쌓아 [배치인덱스, 행동인덱스] 형태 생성
        )

        current_q = tf.gather_nd(q_values, action_indices)  # 실제 선택한 행동에 해당하는 현재 Q값만 추출

        # 다음 상태의 Q값  # target_model이 예측한 다음 상태의 행동별 Q값 계산
        next_q_values = target_model(next_states, training=False)  # 다음 상태 배치를 입력해 Q값 예측

        # 다음 상태에서 가장 큰 Q값  # 다음 상태에서 가장 좋아 보이는 행동의 Q값 선택
        max_next_q = tf.reduce_max(next_q_values, axis=1)  # 각 샘플별로 가장 큰 다음 Q값 계산

        # DQN target  # 정답에 가까운 목표 Q값 계산
        target_q = rewards + gamma * max_next_q * (1 - dones)  # 보상 + 미래 최대 Q값을 이용해 목표값 계산, 종료 상태면 미래 보상 제외

        # 손실 함수  # 현재 Q값과 목표 Q값의 차이를 계산
        loss = tf.reduce_mean(tf.square(target_q - current_q))  # 평균제곱오차 방식으로 손실 계산

    gradients = tape.gradient(loss, policy_model.trainable_variables)  # 손실을 기준으로 모델 파라미터의 기울기 계산
    optimizer.apply_gradients(zip(gradients, policy_model.trainable_variables))  # 계산된 기울기를 이용해 모델 가중치 업데이트

    return loss.numpy()  # 손실값을 numpy 값으로 변환해 반환


# 6. DQN 학습 함수
def train_dqn():
    env = WarehouseRobotEnv(grid_size=5, max_steps=100)  # 5x5 창고 환경 생성, 최대 100 step 제한

    state_dim = int(env.observation_space.shape[0])  # 상태 차원 수를 정수로 저장
    action_dim = int(env.action_space.n)  # 행동 개수를 정수로 저장

    policy_model = create_dqn_model(state_dim, action_dim)  # 실제 학습되는 DQN 모델 생성
    target_model = create_dqn_model(state_dim, action_dim)  # 목표 Q값 계산에 사용할 타깃 모델 생성

    # target network 초기화  # 타깃 모델의 초기 가중치를 policy 모델과 동일하게 맞춤
    target_model.set_weights(policy_model.get_weights())  # policy_model의 가중치를 target_model에 복사

    optimizer = optimizers.Adam(learning_rate=0.001)  # Adam 최적화 알고리즘 생성, 학습률은 0.001

    replay_buffer = ReplayBuffer(capacity=10000)  # 최대 10000개의 경험을 저장할 Replay Buffer 생성

    # 하이퍼파라미터
    episodes = 800  # 총 학습 에피소드 수
    batch_size = 64  # 한 번 학습할 때 사용할 경험 데이터 개수
    gamma = 0.99  # 미래 보상을 얼마나 중요하게 볼지 결정하는 할인율

    epsilon = 1.0  # 초기 탐험 확률, 처음에는 대부분 랜덤 행동
    epsilon_min = 0.05  # epsilon이 줄어들 수 있는 최소값
    epsilon_decay = 0.995  # 에피소드마다 epsilon을 줄이는 비율

    target_update_interval = 20  # 몇 에피소드마다 target_model을 업데이트할지 설정

    episode_rewards = []  # 각 에피소드의 총 보상을 저장할 리스트
    success_history = []  # 각 에피소드의 성공 여부를 저장할 리스트
    loss_history = []  # 각 에피소드의 평균 손실을 저장할 리스트

    for episode in range(1, episodes + 1):  # 1번 에피소드부터 episodes번까지 반복
        state, _ = env.reset()  # 환경을 초기화하고 시작 상태를 받음

        total_reward = 0  # 현재 에피소드의 총 보상 초기화
        success = False  # 현재 에피소드 성공 여부 초기화
        episode_loss = []  # 현재 에피소드에서 발생한 손실값들을 저장할 리스트

        while True:  # 에피소드가 끝날 때까지 반복
            # 행동 선택
            action = select_action(  # epsilon-greedy 방식으로 행동 선택
                state=state,
                model=policy_model,
                epsilon=epsilon,
                action_dim=action_dim
            )

            # 환경에서 행동 실행
            next_state, reward, terminated, truncated, _ = env.step(action)  # 다음 상태, 보상, 종료 여부를 받음

            done = terminated or truncated  # 목표 달성 또는 최대 step 초과이면 에피소드 종료로 판단

            # 경험 저장  # 현재 경험을 Replay Buffer에 저장
            replay_buffer.push(  # 경험 저장 함수 호출
                state,
                action,
                reward,
                next_state,
                done
            )

            state = next_state  # 다음 반복을 위해 현재 상태를 next_state로 갱신
            total_reward += reward  # 현재 에피소드의 총 보상에 이번 보상을 더함

            # Replay Buffer가 충분히 쌓이면 학습
            if len(replay_buffer) >= batch_size:  # 저장된 경험 개수가 batch_size 이상이면
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)  # 무작위 배치 샘플링

                loss = train_step(  # 샘플링한 배치로 DQN 모델을 한 번 학습
                    policy_model=policy_model,  # 학습 대상 모델 전달
                    target_model=target_model,  # 목표값 계산용 모델 전달
                    optimizer=optimizer,  # 최적화 도구 전달
                    states=states,  # 현재 상태 배치 전달
                    actions=actions,  # 행동 배치 전달
                    rewards=rewards,  # 보상 배치 전달
                    next_states=next_states,  # 다음 상태 배치 전달
                    dones=dones,  # 종료 여부 배치 전달
                    gamma=gamma  # 할인율 전달
                )

                episode_loss.append(loss)  # 이번 학습에서 나온 손실값을 저장

            if terminated:  # 목표 달성으로 에피소드가 끝났다면
                success = True

            if done:  # 에피소드가 종료되었다면
                break

        # epsilon 감소  # 에피소드가 끝날 때마다 탐험 확률을 줄임
        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # epsilon이 최소값보다 작아지지 않도록 감소

        # target network 업데이트
        if episode % target_update_interval == 0:  # 현재 에피소드가 업데이트 주기에 해당하면
            target_model.set_weights(policy_model.get_weights())  # policy_model의 가중치를 target_model에 복사

        episode_rewards.append(total_reward)  # 현재 에피소드의 총 보상을 기록
        success_history.append(1 if success else 0)  # 성공이면 1, 실패면 0으로 기록

        if episode_loss:  # 현재 에피소드에서 학습 손실값이 존재하면
            loss_history.append(np.mean(episode_loss))  # 평균 손실값을 기록
        else:  # 아직 학습이 일어나지 않아 손실값이 없다면
            loss_history.append(0)  # 손실값을 0으로 기록

        # 로그 출력  # 학습 진행 상황을 일정 주기마다 출력
        if episode % 50 == 0:  # 50 에피소드마다 로그 출력
            avg_reward = np.mean(episode_rewards[-50:])  # 최근 50개 에피소드의 평균 보상 계산
            success_rate = np.mean(success_history[-50:]) * 100  # 최근 50개 에피소드의 성공률 계산
            avg_loss = np.mean(loss_history[-50:])  # 최근 50개 에피소드의 평균 손실 계산

            print(
                f"Episode {episode:4d} | "   # 현재 에피소드 번호 출력
                f"Avg Reward: {avg_reward:7.2f} | "  # 최근 평균 보상 출력
                f"Success Rate: {success_rate:5.1f}% | "  # 최근 성공률 출력
                f"Loss: {avg_loss:8.4f} | "  # 최근 평균 손실 출력
                f"Epsilon: {epsilon:.3f}"    # 현재 epsilon 값 출력
            )

    return policy_model, episode_rewards, success_history, loss_history  # 학습된 모델과 학습 기록 반환


# 학습 결과 그래프
def plot_training_result(episode_rewards, success_history, loss_history):  # 보상, 성공률, 손실 기록을 입력받음
    # 보상 그래프
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

    # 성공률 이동 평균  # 최근 일정 구간의 성공률 변화를 계산
    window_size = 50  # 이동 평균을 계산할 에피소드 개수 설정
    moving_success_rate = []  # 이동 평균 성공률을 저장할 리스트

    for i in range(len(success_history)):  # 성공 기록 길이만큼 반복
        start = max(0, i - window_size + 1)  # 이동 평균 시작 위치 계산
        recent = success_history[start:i + 1]  # 현재 위치 기준 최근 성공 기록 추출
        moving_success_rate.append(np.mean(recent) * 100)  # 최근 성공률 평균을 백분율로 저장

    plt.figure(figsize=(10, 5))
    plt.plot(moving_success_rate)
    plt.title("Success Rate Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.grid(True)
    plt.show()

    # 손실 그래프  # DQN 학습 손실값 변화를 그래프로 표시
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("DQN Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


# 학습된 모델 평가
def evaluate_agent(model, delay=0.5):  # 학습된 모델과 출력 지연 시간을 입력받음
    env = WarehouseRobotEnv(grid_size=5, max_steps=100)  # 평가용 창고 환경 생성
    state, _ = env.reset()  # 환경 초기화 후 시작 상태 받기
    total_reward = 0  # 평가 중 누적 보상 초기화

    print("학습된 로봇 평가 시작")  # 평가 시작 메시지 출력
    env.render()  # 초기 창고 상태 출력
    time.sleep(delay)  # 사람이 보기 쉽도록 잠시 대기

    while True:
        state_input = np.expand_dims(state, axis=0)  # 모델 입력 형태에 맞게 상태 차원 추가

        q_values = model(state_input, training=False)  # 학습된 모델로 현재 상태의 행동별 Q값 예측
        action = np.argmax(q_values.numpy()[0])  # 가장 Q값이 큰 행동 선택

        next_state, reward, terminated, truncated, _ = env.step(action)  # 선택한 행동을 환경에 적용
        total_reward += reward  # 평가 누적 보상에 현재 보상을 더함
        print(f"선택 행동: {action_to_text(action)}, 보상: {reward}")  # 선택한 행동과 받은 보상 출력
        env.render()  # 행동 후 창고 상태 출력

        time.sleep(delay)

        state = next_state

        if terminated:  # 목표 달성으로 종료되었다면
            print("성공: 상품을 출고 지점에 내려놓았습니다.")  # 성공 메시지 출력
            break

        if truncated:  # 최대 step 초과로 종료되었다면
            print("실패: 최대 step을 초과했습니다.")  # 실패 메시지 출력
            break

    print("평가 총 보상:", total_reward)  # 평가 과정에서 받은 총 보상 출력


def action_to_text(action):  # 행동 번호를 사람이 읽기 쉬운 문자로 바꾸는 함수
    action_map = {  # 행동 번호와 설명 문자열을 매핑하는 딕셔너리
        0: "위로 이동",
        1: "아래로 이동",
        2: "왼쪽으로 이동",
        3: "오른쪽으로 이동",
        4: "상품 집기",
        5: "상품 내려놓기"
    }

    return action_map.get(action, "알 수 없는 행동")


# 모델 저장 / 불러오기
def save_model(model, path="warehouse_dqn_tf_model.keras"):
    model.save(path)
    print(f"모델 저장 완료: {path}")

def load_model(path="warehouse_dqn_tf_model.keras"):
    model = tf.keras.models.load_model(path)  # 지정된 경로에서 Keras 모델 불러오기
    print(f"모델 불러오기 완료: {path}")
    return model

if __name__ == "__main__":
    trained_model, rewards, success_history, loss_history = train_dqn()  # DQN 학습 실행 후 모델과 학습 기록 받기

    plot_training_result(
        episode_rewards=rewards,  # 에피소드별 보상 기록 전달
        success_history=success_history,  # 성공 여부 기록 전달
        loss_history=loss_history  # 손실 기록 전달
    )

    save_model(trained_model)
    evaluate_agent(trained_model, delay=0.5)  # 학습된 모델로 창고 로봇 평가 실행