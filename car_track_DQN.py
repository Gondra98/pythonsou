# -*- coding: utf-8 -*-
"""
Mini Car - DQN (Centerline Progress, Anti-stuck, Slightly Slower, CCW)
- Actions(4): 0=Left+Throttle, 1=Straight+Throttle, 2=Right+Throttle, 3=Brake
- Start: left straight, facing DOWN (+pi/2, CCW). Start lane biased to OUTER side.
- Brake allowed, NO reverse (v >= 0)
"""
# 강화학습에서 하나의 에피소드(episode) 란
# 에이전트가 환경(env) 속에서 시작 상태(reset) → 행동 선택(step) → 보상 받음(reward) → 다음 상태(next state)
# 를 반복하다가 종료(done=True)가 되는 한 사이클입니다.
# 즉, “에이전트가 한 번 트랙에 들어가서, 부딪히거나 목표를 달성하거나 제한시간이 끝날 때까지의 한 번의 주행”

import math, random, sys, time
import numpy as np

try:
    import pygame
except ImportError:
    print("pygame 미설치: pip install pygame")
    raise

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("PyTorch 미설치: pip install torch")
    raise


# 0) 글로벌 설정
class G:
    SEED = 42
    WIDTH, HEIGHT = 960, 640
    FPS = 60
    HEADLESS = False
    SHOW_SENSORS = True

    # 트랙(도넛)
    OUTER_MARGIN = 40
    INNER_MARGIN = 180
    CORNER_RADIUS = 200

    # 센서
    SENSOR_COUNT = 5
    SENSOR_FOV_DEG = 120
    SENSOR_MAX_DIST = 220.0

    # 차량 (요청대로 약간 느리게)
    CAR_LENGTH = 26
    CAR_WIDTH  = 14
    MAX_STEER  = math.radians(28)
    MAX_ACCEL  = 0.11          # ↓ 조금 낮춤
    FRICTION   = 0.012
    TURN_GAIN  = 0.062
    SPEED_CLAMP = 3.4          # ↓ 최고속 소폭 하향

    # 에피소드/학습
    EPISODE_STEPS = 500
    N_EPISODES    = 400

    # DQN
    OBS_DIM   = SENSOR_COUNT + 1
    N_ACTIONS = 4
    HIDDEN    = 128
    GAMMA     = 0.99
    LR        = 2.5e-4
    BATCH     = 128
    BUFFER    = 50_000
    START_LEARN = 1_000
    TARGET_SYNC = 500
    EPS_START = 1.0
    EPS_END   = 0.05
    EPS_DECAY_STEPS = 20_000

random.seed(G.SEED)
np.random.seed(G.SEED)
torch.manual_seed(G.SEED)


# 1) 기하 유틸
def line_intersection(p, r, q, s):
    rxs = r[0]*s[1] - r[1]*s[0]
    if abs(rxs) < 1e-9: return None
    qmp = (q[0]-p[0], q[1]-p[1])
    t = (qmp[0]*s[1]-qmp[1]*s[0]) / rxs
    u = (qmp[0]*r[1]-qmp[1]*r[0]) / rxs
    if 0 <= t <= 1 and 0 <= u <= 1:
        return (p[0]+t*r[0], p[1]+t*r[1]), t
    return None

def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%n]
        if (y1 > y) != (y2 > y):
            xints = (x2 - x1)*(y - y1)/(y2 - y1 + 1e-9) + x1
            if x < xints: inside = not inside
    return inside

def rounded_rect_polygon(w, h, r, cx, cy, n_corner=10):
    r = min(r, w/2 - 2, h/2 - 2)
    corners = [
        (cx - w/2 + r, cy - h/2 + r),
        (cx + w/2 - r, cy - h/2 + r),
        (cx + w/2 - r, cy + h/2 - r),
        (cx - w/2 + r, cy + h/2 - r),
    ]
    pts = []
    for i,(cxk, cyk) in enumerate(corners):
        start = i * math.pi/2 + math.pi
        for j in range(n_corner+1):
            th = start + j*(math.pi/2)/n_corner
            pts.append((cxk + r*math.cos(th), cyk + r*math.sin(th)))
    return pts

def polygon_edges(poly):
    return list(zip(poly, poly[1:]+poly[:1]))


# 2) 트랙 (중심선/접선 포함, 시작선은 OUTER 쪽으로 오프셋)
class Track:
    def __init__(self):
        W, H = G.WIDTH, G.HEIGHT
        outer_w = W - 2*G.OUTER_MARGIN
        outer_h = H - 2*G.OUTER_MARGIN
        inner_w = W - 2*G.INNER_MARGIN
        inner_h = H - 2*G.INNER_MARGIN

        self.outer = rounded_rect_polygon(outer_w, outer_h, G.CORNER_RADIUS, W/2, H/2, 18)
        self.inner = rounded_rect_polygon(inner_w, inner_h, G.CORNER_RADIUS*0.6, W/2, H/2, 18)
        self.outer_edges = polygon_edges(self.outer)
        self.inner_edges = polygon_edges(self.inner)

        # --- 중심선(outer/inner 대응 점의 중간점)과 접선(unit tangent) 미리 계산 ---
        assert len(self.outer) == len(self.inner)
        self.center = [((ox+ix)/2.0, (oy+iy)/2.0) for (ox,oy),(ix,iy) in zip(self.outer, self.inner)]
        self.center_tan = []
        for i in range(len(self.center)):
            x1,y1 = self.center[i]
            x2,y2 = self.center[(i+1) % len(self.center)]
            v = np.array([x2-x1, y2-y1], dtype=np.float32)
            n = np.linalg.norm(v) + 1e-9
            self.center_tan.append((v[0]/n, v[1]/n))

        # --- 왼쪽 직선: OUTER쪽으로 65% 쪽에 시작선 배치(안쪽 벽과 여유) ---
        outer_left = W/2 - outer_w/2
        inner_left = W/2 - inner_w/2
        lane_x = outer_left * 0.65 + inner_left * 0.35  # OUTER에 더 가깝게

        self.start_base = (lane_x, H/2)
        self.start_angle = +math.pi/2  # 아래쪽, CCW

    def start_pose(self):
        dl  = np.random.uniform(-40, 40)                           # 진행(y)
        dn  = np.random.uniform(-6, 6)                             # 횡(x)
        dth = np.random.uniform(-math.radians(4), math.radians(4)) # 각도
        x0, y0 = self.start_base
        return (x0 + dn, y0 + dl, self.start_angle + dth)

    def on_track(self, p):
        return point_in_polygon(p, self.outer) and (not point_in_polygon(p, self.inner))

    def raycast(self, p, ang, maxdist):
        dx = math.cos(ang)*maxdist
        dy = math.sin(ang)*maxdist
        best = None
        for (a,b) in self.outer_edges + self.inner_edges:
            hit = line_intersection(p, (dx,dy), a, (b[0]-a[0], b[1]-a[1]))
            if hit is not None:
                _, t = hit
                dist = t*maxdist
                if best is None or dist < best:
                    best = dist
        return best if best is not None else maxdist

    def nearest_center_idx(self, p):
        x,y = p
        d2_min, idx = 1e18, 0
        for i,(cx,cy) in enumerate(self.center):
            d2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
            if d2 < d2_min:
                d2_min, idx = d2, i
        return idx

    def tangent_at(self, idx):
        return self.center_tan[idx % len(self.center)]


# 3) 차량 (브레이크 허용, 후진 금지)
class Car:
    def __init__(self, track):
        self.track = track
        self.reset()

    def reset(self):
        self.x, self.y, self.angle = self.track.start_pose()
        self.v = 0.0
        self.alive = True
        self.time_alive = 0

    def step(self, steer_cmd, throttle):
        """steer_cmd ∈ [-1,1], throttle ∈ [-1,1]; v>=0 (no reverse)"""
        steer = float(np.clip(steer_cmd, -1.0, 1.0)) * G.MAX_STEER
        acc   = float(np.clip(throttle,  -1.0, 1.0)) * G.MAX_ACCEL
        self.v = np.clip(self.v + acc, 0.0, G.SPEED_CLAMP)                 # NO reverse
        self.angle += steer * (1.0 + 0.15*self.v) * G.TURN_GAIN
        if self.v > 1e-6: self.v = max(0.0, self.v - G.FRICTION)
        else: self.v = 0.0
        self.x += math.cos(self.angle)*self.v
        self.y += math.sin(self.angle)*self.v
        if not self.track.on_track((self.x, self.y)): self.alive = False
        self.time_alive += 1

    def sensor_readings(self):
        readings = []
        span = math.radians(G.SENSOR_FOV_DEG)
        for i in range(G.SENSOR_COUNT):
            a = -span/2 + span*(i/(G.SENSOR_COUNT-1))
            ang = self.angle + a
            d = self.track.raycast((self.x, self.y), ang, G.SENSOR_MAX_DIST)
            readings.append(d / G.SENSOR_MAX_DIST)
        return np.array(readings, dtype=np.float32)

    def rect_points(self):
        L, W = G.CAR_LENGTH, G.CAR_WIDTH
        pts = [(L/2,0), (-L/2,-W/2), (-L/2, W/2)]
        rot = lambda X,Y: (self.x + X*math.cos(self.angle) - Y*math.sin(self.angle),
                           self.y + X*math.sin(self.angle) + Y*math.cos(self.angle))
        return [rot(*p) for p in pts]


# 4) DQN 구성요소
class QNet(nn.Module):
    def __init__(self, in_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, G.HIDDEN), nn.ReLU(),
            nn.Linear(G.HIDDEN, G.HIDDEN), nn.ReLU(),
            nn.Linear(G.HIDDEN, n_actions)
        )
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, cap):
        self.cap = cap; self.ptr = 0; self.full = False
        self.s  = np.zeros((cap, G.OBS_DIM), dtype=np.float32)
        self.a  = np.zeros((cap,), dtype=np.int64)
        self.r  = np.zeros((cap,), dtype=np.float32)
        self.ns = np.zeros((cap, G.OBS_DIM), dtype=np.float32)
        self.d  = np.zeros((cap,), dtype=np.float32)
    def push(self, s,a,r,ns,d):
        i = self.ptr
        self.s[i]=s; self.a[i]=a; self.r[i]=r; self.ns[i]=ns; self.d[i]=float(d)
        self.ptr = (self.ptr+1)%self.cap
        if self.ptr==0: self.full=True
    def __len__(self): return self.cap if self.full else self.ptr
    def sample(self, batch):
        idx = np.random.randint(0, len(self), size=batch)
        return (
            torch.from_numpy(self.s[idx]),
            torch.from_numpy(self.a[idx]),
            torch.from_numpy(self.r[idx]),
            torch.from_numpy(self.ns[idx]),
            torch.from_numpy(self.d[idx]),
        )

# 5) 환경 (중심선 진행 보상 포함)
class SimpleCarEnv:
    def __init__(self):
        self.track = Track()
        self.car = Car(self.track)
        self.steps = 0
        self.prev_xy = None  # 진행 보상 계산용

    def reset(self):
        self.car = Car(self.track)
        self.steps = 0
        self.prev_xy = np.array([self.car.x, self.car.y], dtype=np.float32)
        return self._obs()

    def _obs(self):
        sensors = self.car.sensor_readings()
        speed = np.array([self.car.v / G.SPEED_CLAMP], np.float32)
        return np.concatenate([sensors, speed], axis=0)

    def step(self, action):
        """
        0: Left+Throttle     (steer=-1, throttle=+1)
        1: Straight+Throttle (steer= 0, throttle=+1)
        2: Right+Throttle    (steer=+1, throttle=+1)
        3: Brake             (steer= 0, throttle=-1)
        """
        steer  = [-1.0,  0.0, +1.0, 0.0][action]
        thr    = [ +1.0, +1.0, +1.0,-1.0][action]
        self.car.step(steer, thr)

        sensors    = self.car.sensor_readings()
        left, mid, right = sensors[0], sensors[2], sensors[-1]
        min_clear  = float(np.min(sensors))
        mean_clear = float(np.mean(sensors))
        v_norm     = float(self.car.v / G.SPEED_CLAMP)

        # ---- (A) 중심선 진행 보상 ----
        cur_xy = np.array([self.car.x, self.car.y], dtype=np.float32)
        disp = cur_xy - self.prev_xy
        self.prev_xy = cur_xy
        # 현재 위치에서 가장 가까운 중심선 인덱스와 접선
        idx = self.track.nearest_center_idx((self.car.x, self.car.y))
        tx, ty = self.track.tangent_at(idx)
        t_hat = np.array([tx, ty], dtype=np.float32)
        progress = float(np.dot(disp, t_hat))            # 접선 방향 성분(+ 전진, - 후진)

        # ---- (B) 보상 구성 ----
        reward  = 0.20
        reward += 0.05 * max(0.0, progress)             # ← 진행 보상(스케일은 작게, 그러나 결정적)
        reward -= 0.03 * max(0.0, -progress)            # ← 반대 방향 패널티(약하게)

        reward += 0.50 * mid                             # 정면 여유
        reward += 0.20 * mean_clear                      # 전체 여유
        reward += 0.30 * v_norm * max(0.0, mid - 0.15)   # 너무 가깝지 않을 때만 속도 보상

        # 중앙 정렬 & 회피
        center = 1.0 - abs(right - left)
        reward += 0.20 * center
        turn_dir = (-1 if action==0 else (1 if action==2 else 0))
        reward  += 0.25 * (right - left) * turn_dir

        # 안전버퍼
        near = max(0.0, 0.22 - min(left, mid, right))
        reward -= 0.80 * near

        # 브레이크는 정말 막혔을 때만
        if action == 3:
            reward += (0.18 if mid < 0.14 else -0.10)

        # 조기 종료(충돌 임박/이탈)
        done = False
        if min_clear < 0.03 or not self.car.alive:
            reward -= 50.0
            done = True

        self.steps += 1
        if self.steps >= G.EPISODE_STEPS:
            done = True

        return self._obs(), reward, done

    def draw(self, screen):
        draw_track(screen, self.track)
        draw_car(screen, self.car)


# 6) 렌더링
def draw_track(screen, track):
    screen.fill((18,18,22))
    pygame.draw.polygon(screen, (70,70,70), track.outer)
    pygame.draw.polygon(screen, (18,18,22), track.inner)
    pygame.draw.lines(screen, (220,220,220), True, track.outer, 2)
    pygame.draw.lines(screen, (220,220,220), True, track.inner, 2)

def draw_car(screen, car):
    pts = car.rect_points()
    pygame.draw.polygon(screen, (0,170,255), pts)
    if G.SHOW_SENSORS and car.alive:
        span = math.radians(G.SENSOR_FOV_DEG)
        for i in range(G.SENSOR_COUNT):
            a = -span/2 + span*(i/(G.SENSOR_COUNT-1))
            ang = car.angle + a
            d = car.track.raycast((car.x, car.y), ang, G.SENSOR_MAX_DIST)
            x2 = car.x + math.cos(ang)*d
            y2 = car.y + math.sin(ang)*d
            pygame.draw.line(screen, (255,200,50), (car.x,car.y), (x2,y2), 1)
            pygame.draw.circle(screen, (255,200,50), (int(x2),int(y2)), 2)


# 7) 학습 루프 (DQN)
def linear_eps(step):
    t = min(1.0, step / G.EPS_DECAY_STEPS)
    return G.EPS_START + (G.EPS_END - G.EPS_START) * t

def train_dqn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SimpleCarEnv()

    q  = QNet(G.OBS_DIM, G.N_ACTIONS).to(device)
    tq = QNet(G.OBS_DIM, G.N_ACTIONS).to(device)
    tq.load_state_dict(q.state_dict())
    opt = optim.Adam(q.parameters(), lr=G.LR)
    buf = ReplayBuffer(G.BUFFER)

    if not G.HEADLESS:
        pygame.init()
        screen = pygame.display.set_mode((G.WIDTH, G.HEIGHT))
        pygame.display.set_caption("Mini Car - DQN (Centerline Progress)")

    global_step = 0
    returns = []

    for ep in range(1, G.N_EPISODES+1):
        s = env.reset()
        ep_ret, done = 0.0, False

        # 워밍업(탐색 데이터 확보)
        warm = 60
        for _ in range(warm):
            a = np.random.randint(0, G.N_ACTIONS)
            ns, r, d = env.step(a)
            buf.push(s, a, r, ns, d)
            s = ns; ep_ret += r
            if d: s, ep_ret = env.reset(), 0.0

        while not done:
            if not G.HEADLESS:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit(); sys.exit(0)

            eps = linear_eps(global_step)
            if random.random() < eps:
                a = np.random.randint(0, G.N_ACTIONS)
            else:
                with torch.no_grad():
                    qs = q(torch.from_numpy(s).float().unsqueeze(0).to(device))
                    a = int(qs.argmax(dim=1).item())

            ns, r, done = env.step(a)
            buf.push(s, a, r, ns, done)
            s = ns; ep_ret += r; global_step += 1

            if len(buf) >= G.START_LEARN:
                bs, ba, br, bns, bd = buf.sample(G.BATCH)
                bs, ba, br, bns, bd = bs.to(device), ba.to(device), br.to(device), bns.to(device), bd.to(device)
                with torch.no_grad():
                    next_act = q(bns).argmax(dim=1, keepdim=True)
                    next_q   = tq(bns).gather(1, next_act).squeeze(1)
                    target   = br + (1.0 - bd) * G.GAMMA * next_q
                cur_q = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss = nn.SmoothL1Loss()(cur_q, target)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
                if global_step % G.TARGET_SYNC == 0:
                    tq.load_state_dict(q.state_dict())

            if not G.HEADLESS:
                env.draw(screen)
                font = pygame.font.SysFont("consolas", 18)
                hud  = font.render(f"Ep {ep}  Step {global_step}  Ret {ep_ret:.1f}  eps {eps:.2f}", True, (240,240,240))
                screen.blit(hud, (16, 12))
                pygame.display.flip()
                pygame.time.Clock().tick(G.FPS)

        returns.append(ep_ret)
        if ep % 10 == 0:
            print(f"[Ep {ep:04d}] return(avg10) = {np.mean(returns[-10:]):.1f}, buffer={len(buf)}")

    if not G.HEADLESS:
        done = False
        while not done:
            for e in pygame.event.get():
                if e.type == pygame.QUIT: done = True
            time.sleep(0.02)
        pygame.quit()



if __name__ == "__main__":
    train_dqn()
