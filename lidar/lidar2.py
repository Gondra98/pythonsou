# pip install gymnasium

import math
import numpy as np
import gymnasium as gym         # 실습환경 제공(현재상태 제공 -> 행동선택 -> 환경이 행동이 반영)
from gymnasium import spaces    # 행동 공간과 관측 공간을 정의
import matplotlib.pyplot as plt

# 환경/장애물/라이다 설정
WORLD_W, WORLD_H = 20.0, 15.0
OBSTACLES = [(6.0, 4.0, 0.5),(8.0, 10.0, 1.5),(15.0, 5.0, 1.0)]
NUM_RAYS = 20
FOV = np.deg2rad(150)
MAX_RANGE = 8.0     # 라이다 최대 감지 거리
STEP_MARCH = 0.05   # 레이 전진 단위 거리

def inside_worldFunc(x, y):     # 좌표값이 시뮬레이터 공간 경계 내에 있는지 여부
    return (0.0 <= x <= WORLD_W) and (0.0 <= y <= WORLD_H)

# 라이다 광선의 끝점이 장애물과 충돌했는지 여부
def hit_circleFunc(px, py, cx, cy, r):
    return (px - cx) ** 2 + (py - cy) ** 2 < r ** 2     # ... > r ** 2

# 에이전트(x,y,theta)에서 시야각(FOV)으로 NUM_RAYS개의 광선을 쏴,
# 각 레이가 처음 부딪히는 지점까지 거리구하기
def cast_lidar(x, y, theta, num_rays = NUM_RAYS, fov = FOV,
               max_range=MAX_RANGE, step=STEP_MARCH):
    start = theta - fov / 2 # 전방 시약각 왼쪽 끝 각도(첫번째 레이아웃)
    angles = start + np.arange(num_rays) * (fov / max(num_rays - 1, 1))     # 균등 분포 각도 배열
    print('angles : ', angles)
    dists = np.full(num_rays, max_range, dtype=np.float32)  # 초기 거리 배열 초기화 = 최대거리
    print('dists :', dists)

    for i, ang in enumerate(angles):
        dist = 0.0      # 레이(관선 전진값)
        hit = False     # 에이전트가 환경 영역 밖 또는 장애물 충돌 여부
        while dist < max_range:     # 최대거리까지 전진
            px = x + math.cos(ang) * dist   # 레이 끝점 x 좌표
            py = y + math.sin(ang) * dist   # 레이 끝점 y 좌표
            if not inside_worldFunc(px, py):
                hit = True
                break
            for (cx, cy, r) in OBSTACLES:
                if hit_circleFunc(px, py, cx, cy, r):
                    hit = True
                    break
            if hit:
                break

            dist += step    # 충돌이 없으면 레이 전진

        dists[i] = min(dist, max_range)     # 충돌 거리 기록

    return dists, angles    # 거리 배열, 각도

# Gymnasium의 기존환경을 상속받은 클래스 작성
class SimpleLidarEnv(gym.Env):
    def __init__(self, render_mode = "human"):
        super().__init__()
        self.render_mode = render_mode
        # 강화학습 환경 설정시 두 가지는 반드시 선언
        self.action_space = spaces.Discrete(3)      # 가능한 해동 3가지(0:좌회전, 1:직진, 2:우회전)
        self.observation_space = spaces.Box(
            low=0.0,
            high=MAX_RANGE,
            shape=(NUM_RAYS,),
            dtype=np.float32
            # 관측값은 길이 20짜리 배열, 각 값의 범위는 0.0 ~ MAX_RANGE
        )

        self.v = 0.25   # 전진 속도
        self.steer_delta = np.deg2rad(8)    # 회전각도 각 8도
        self.goal = np.array([18.0, 12.0], dtype=np.float32)    # 에이전트의 최종 목표
        self.goal_radius = 0.6  # 목표 판정 반경
        self.max_steps = 400    # 하나의 에피소드에서 허용되는 최대 행동 횟수

        self.fig, self.ax = None, None     # 렌더링용 객체
        self._state = None      # [x, y, theta]
        self._prev_goal_dist = None # 이전 목표 거리
        self._steps = 0         # step counter

    # 현재 환경 상태를 강화학습 모델이 이해할 수 있는 숫자 배열로 만들어 반환
    def _get_obs(self):
        x, y, th = self._state  # 환경 객체가 가진 상태
        obs, _ = cast_lidar(x, y, th)   # 라이다 거리 관측
        return obs.astype(np.float32)

    # gymnasium 환경에 추가 정보를 제공
    def _get_info(self):
        x, y, _ = self._state
        d = np.linalg.norm(np.array([x, y]) - self.goal)    # 목표까지의 거리
        return {'goal_dist':float(d), 'steps':self._steps}

    def _collision(self):
        x, y, _ = self._state
        if not inside_worldFunc(x, y):
            return True

        for (cx, cy, r) in OBSTACLES:
            if hit_circleFunc(x, y, cx, cy, r + 0.25):
                return True
            
        return False
    
    def reset(self, seed = None, options = None):
        super().reset(seed = seed)

        self._state = np.array([2.0, 2.0, np.deg2rad(30.0)], dtype=np.float32)
        self._steps = 0
        self._prev_goal_dist = np.linalg.norm(self._state[:2] - self.goal)  # 초기 목표 거리
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
    
    def step(self, action):
        self._steps += 1
        x, y, th = self._state
        
        # 행동 적용
        if action == 0:
            th += self.steer_delta  # 좌회전
        elif action == 2:
            th -= self.steer_delta  # 우회전

        x += math.cos(th) * self.v
        y += math.sin(th) * self.v
        self._state = np.array([x, y, th], dtype=np.float32)    # 새위치 새 방향으로

        goal_dist = np.linalg.norm(self._state[:2] - self.goal)
        process = self._prev_goal_dist - goal_dist  # 접근 변화량
        self._prev_goal_dist = goal_dist

        reward = 1.0 * process - 0.01   # 보상 계산 : 매 스텝마다 패널티(- 0.01) 부여
        terminated, truncated = False, False

        if goal_dist < self.goal_radius:    # 목표 도달
            reward += 1.0
            terminated = True
        
        if self._collision():
            reward -= 1.0;
            terminated = True

        if self._steps >= self.max_steps:
            terminated = True   # 스탭 초과로 종료

        # 관측 정보 반환
        obs = self._get_obs()   # 라이더 거리 벡터
        info = self._get_info() # goal_dist, steps

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print('현재 상태 : ', self._state)

            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(8, 6))
                plt.ion()  # interactive mode (pause 사용 위해)

            ax = self.ax
            ax.clear()
            ax.set_xlim(0, WORLD_W)
            ax.set_ylim(0, WORLD_H)
            ax.set_aspect('equal', adjustable = 'box')  # 가로 세로 비율은 1:1, 왜곡방지
            ax.set_title('Simple Lidar Env')
            ax.plot([0, WORLD_W, WORLD_W, 0, 0], [0, 0, WORLD_H, WORLD_H, 0], lw=2) # 경계사각형
            
            for (cx, cy, r) in OBSTACLES:
                circ = plt.Circle((cx, cy), r, edgecolor='tab:red', facecolor='none', lw=2)
                ax.add_patch(circ)

            goal = plt.Circle(tuple(self.goal), self.goal_radius, edgecolor='tab:blue', facecolor='none')
            ax.add_patch(goal)

            # 에이전트 : 삼각형 형태로 그리기
            x, y, th = self._state
            L = 0.6     # 삼각형 길이(크기)

            tri = np.array([
                [x + np.cos(th) * L,       y + np.sin(th) * L      ],
                [x + np.cos(th + 2.5) * L, y + np.sin(th + 2.5) * L],
                [x + np.cos(th - 2.5) * L, y + np.sin(th - 2.5) * L],
                ])
            
            ax.fill(tri[:, 0], tri[:, 1], alpha=0.85, color='tab:blue', label='agent')

            obs, angs = cast_lidar(x, y, th)    # 라이다 밤 시각화

            for d, a in zip(obs, angs):
                # 두 점을 연결하는 선 그리기
                ax.plot([x, x + np.cos(a) * d], [y, y + np.sin(a) * d], lw=1, alpha=0.9)
            ax.legend(loc='upper right')
            plt.pause(0.001)        # frame 갱신. rander() 내에서 매번 그림이 다시 그려지게 확인

                

            def close(self):
                if self.ifg is not None:
                    plt.close(self.fig)
                    self.fig = None
                    self.ax = None



if __name__ == '__main__':
    env = SimpleLidarEnv()
    obs, info = env.reset()

    total_reward = 0.0

    for t in range(500):
        action = env.action_space.sample()  # 환경에서 가능한 행동범위중 무작위로 행동 하나를 선택(직진, 좌, 우)
        obs, reward, terminated, truncated, info = env.step(action)     # 환경 단계
        # obs: 새로운 상태, reward:보상, terminated, truncated: 시간제한 종료 여부, info
        total_reward += reward
        env.render()

    if terminated or truncated:     # 에피소드 종료 조건
        print(f'Episode and at step={t}, total_reward={total_reward:.3f}, info={info}')
        obs, info = env.reset()
        total_reward = 0.0

    env.close()









