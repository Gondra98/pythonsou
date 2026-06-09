"""
BipedalWalker-v3는 OpenAI Gym에서 제공하는 강화학습용 물리 시뮬레이션 환경 중 하나로,
에이전트가 두 다리로 걷는 로봇(Bipedal Walker)를 조작하여 지면 위를 안정적으로 이동하도록 학습하는 문제다.
이 환경은 연속 행동 공간(Continuous Action Space)을 다루며, SAC, PPO, DDPG 등 정책기반 알고리즘 학습 실험에 자주 사용된다.

Soft Actor-Critic (SAC) 알고리즘으로 BipedalWalker-v3에서 2D 로봇을 학습시키는 강화학습 실습 코드입니다. 핵심 구조와 목적을 정확하게 정리해주면 다음과 같습니다.

1. 이 코드의 실습 목적
  목표 : 2족 로봇이 쓰러지지 않고 앞으로 걷기
  행동(action)을 연속적으로 제어 (토크 값)
  강화학습으로 걷는 정책(policy)을 자동 학습

2. 전체 구조 설명 : 4개 핵심 파트로 구성.
① 환경 (Gym Environment)
env = gym.make("BipedalWalker-v3", render_mode="human")
이 환경은 4개의 다리 관절 제어, 연속 행동 공간 (continuous action space), 목표: 앞으로 이동 + 균형 유지

특징: reward 기반 학습, 넘어지면 패널티, 앞으로 갈수록 보상 증가

② Replay Buffer (경험 저장소)
class ReplayBuffer:
역할: (state, action, reward, next_state) 저장, 랜덤 샘플링으로 학습 안정화
왜 필요?
  → 연속 데이터 그대로 학습하면 불안정해짐
  → '경험 재사용'으로 학습 안정화

③ Actor / Critic 네트워크
Actor (행동 생성) : a = tanh(mu + std * noise)
  상태 → 행동 생성, 확률적 정책 (stochastic policy), exploration 자동 수행
Critic (가치 평가)
  Q(s, a) : 이 행동이 얼마나 좋은가? 평가, SAC는 Q network 2개 사용 (double Q-learning)

④ SAC 알고리즘 핵심
SAC = '보상 + 엔트로피(랜덤성)' 최대화, 즉 최고의 행동 + 적당한 랜덤성 유지
"""

import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Cfg:
    ENV_ID = "BipedalWalker-v3"
    SEED = 42

    MAX_EPISODES = 300

    START_STEPS = 1000
    UPDATE_AFTER = 1000
    UPDATE_EVERY = 10
    GRAD_STEPS_PER_ITER = 20

    BATCH_SIZE = 256
    REPLAY_SIZE = 1_000_000

    GAMMA = 0.99
    TAU = 0.005

    LR_ACTOR = 3e-4
    LR_CRITIC = 3e-4
    LR_ALPHA = 3e-4
    TARGET_ENTROPY_COEF = 1.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(Cfg.SEED)
np.random.seed(Cfg.SEED)
random.seed(Cfg.SEED)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size, 1), dtype=np.float32)
        self.done = np.zeros((size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.obs2[self.ptr] = o2
        self.done[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        return dict(
            obs=torch.tensor(self.obs[idx], device=device),
            obs2=torch.tensor(self.obs2[idx], device=device),
            act=torch.tensor(self.act[idx], device=device),
            rew=torch.tensor(self.rew[idx], device=device),
            done=torch.tensor(self.done[idx], device=device),
        )


# Networks
def mlp(in_dim, out_dim, hidden=(256, 256)):
    layers = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim, 256)
        self.mu = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

    def forward(self, x):
        x = self.net(x)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        eps = torch.randn_like(mu)
        z = mu + std * eps
        a = torch.tanh(z)

        logp = -0.5 * (((z - mu) / (std + 1e-8)) ** 2)
        logp = logp - torch.log(std + 1e-8) - 0.5 * np.log(2 * np.pi)
        logp = logp.sum(dim=-1, keepdim=True)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return a, logp


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1)

    def forward(self, o, a):
        return self.net(torch.cat([o, a], dim=-1))


# SAC Agent
class SAC:
    def __init__(self, obs_dim, act_dim):
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.q1 = Critic(obs_dim, act_dim).to(device)
        self.q2 = Critic(obs_dim, act_dim).to(device)
        self.q1_t = Critic(obs_dim, act_dim).to(device)
        self.q2_t = Critic(obs_dim, act_dim).to(device)

        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=Cfg.LR_ACTOR)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=Cfg.LR_CRITIC)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=Cfg.LR_CRITIC)

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=Cfg.LR_ALPHA)

        self.target_entropy = -Cfg.TARGET_ENTROPY_COEF * act_dim

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def soft_update(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(t.data * (1 - Cfg.TAU) + s.data * Cfg.TAU)

    def train(self, batch):
        o, a, r, o2, d = batch["obs"], batch["act"], batch["rew"], batch["obs2"], batch["done"]

        with torch.no_grad():
            a2, logp2 = self.actor.sample(o2)
            q = torch.min(self.q1_t(o2, a2), self.q2_t(o2, a2)) - self.alpha * logp2
            backup = r + Cfg.GAMMA * (1 - d) * q

        q1_loss = F.mse_loss(self.q1(o, a), backup)
        q2_loss = F.mse_loss(self.q2(o, a), backup)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        a_new, logp = self.actor.sample(o)
        q_pi = torch.min(self.q1(o, a_new), self.q2(o, a_new))

        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self.soft_update(self.q1_t, self.q1)
        self.soft_update(self.q2_t, self.q2)

    # SAVE / LOAD 추가
    def save(self, path="sac_actor.pth"):
        torch.save(self.actor.state_dict(), path)

    def load(self, path="sac_actor.pth"):
        self.actor.load_state_dict(torch.load(path, map_location=device))
        self.actor.eval()


# TRAIN
def main():
    env = gym.make(Cfg.ENV_ID)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SAC(obs_dim, act_dim)
    buf = ReplayBuffer(obs_dim, act_dim, Cfg.REPLAY_SIZE)

    total_steps = 0

    for episode in range(Cfg.MAX_EPISODES):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if total_steps < Cfg.START_STEPS:
                act = env.action_space.sample()
            else:
                with torch.no_grad():
                    o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    act, _ = agent.actor.sample(o)
                    act = act.cpu().numpy()[0]

            next_obs, reward, done, trunc, _ = env.step(act)

            buf.store(obs, act, reward, next_obs, float(done or trunc))

            obs = next_obs
            episode_reward += reward
            total_steps += 1

            if total_steps > Cfg.UPDATE_AFTER and total_steps % Cfg.UPDATE_EVERY == 0:
                for _ in range(Cfg.GRAD_STEPS_PER_ITER):
                    batch = buf.sample(Cfg.BATCH_SIZE)
                    agent.train(batch)

            if done or trunc:
                break

        print(f"Episode {episode} | Reward: {episode_reward:.2f}")

    # SAVE MODEL
    agent.save("sac_actor.pth")
    print("Model saved: sac_actor.pth")

    env.close()


# TEST (LOAD MODEL)
def test():
    env = gym.make(Cfg.ENV_ID, render_mode="human")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SAC(obs_dim, act_dim)
    agent.load("sac_actor.pth")

    for episode in range(5):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                act, _ = agent.actor.sample(o)
                act = act.cpu().numpy()[0]

            obs, reward, done, trunc, _ = env.step(act)
            episode_reward += reward

            env.render()

            if done or trunc:
                break

        print(f"[TEST] Episode {episode} Reward: {episode_reward:.2f}")

    env.close()


if __name__ == "__main__":
    # main()   # 학습용
    test()   # 학습된 정책 검증용
    # test() 실행 시: 로봇이 걷는 장면, 넘어지거나 균형 잡는 움직임, forward reward 행동 결과를 봄