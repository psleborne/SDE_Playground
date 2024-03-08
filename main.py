import math
import random

import matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar

import simpleQ
import simpleQ as sq
import Simulation
import agent_class as ac

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from itertools import chain
matplotlib.rcParams.update({'font.size': 42})


# %matplotlib inline

DEFAULT_ENV_NAME = "ControlingBM"


def dW(delta_t: float) -> float:
    """Sample a random number at each call."""
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))


def a1(x, i):
    return 1.0


def b(x, i, theta0, theta1):
    delta = 1 / (2 * theta0) + 1 / (2 * theta1)
    if x > delta:
        return theta0
    else:
        return theta1


# Action is 0 or 1, either drift theta0 or theta1
def action(current_state, eps, Q):
    if random.random() < eps:
        return round(random.random())
    else:
        # print("Choosing Q value")
        return np.argmin(Q[current_state, :])


def control(theta0, theta1, a, t):
    #     a = 0: action is to turn left
    #     a = 1: action is to turn right
    if a == 0:
        # print("Theta0 = ", theta0)
        b = theta0 - 1.0 / (-1 * math.sqrt(t) - 1 / math.sqrt(-1.0 * theta0))
        # b = theta0 + 1.0 / ((-1 * t - 1 / math.sqrt(-1.0 * theta0)) * (-1 * t - 1 / math.sqrt(-1.0 * theta0)))
        # b = theta0
    else:
        b = theta1 - 1.0 / (math.sqrt(t) + 1 / math.sqrt(theta1))
        # b = theta1 - 1.0 / ((t + 1 / math.sqrt(theta1)) * (t + 1 / math.sqrt(theta1)))
        # b = theta1
    return b


def getbounds(X):
    maxx = max(X)
    minx = min(X)

    return minx, maxx


def get_state(minx: float, maxx: float, n: int, x: float):
    if n == 0:
        dt = 0
    else:
        dt = abs(maxx - minx) / n
    if x >= maxx:
        return n + 1
    elif x <= minx:
        return 0
    else:
        for i in range(n + 1):

            if x >= (i) * dt + minx and x < (i + 1) * dt + minx:
                return i + 1

    print("NoneFound x =", x)
    return 1


def b_Q(x, t, Q, n_states, minx, maxx, theta0, theta1):
    state = get_state(minx, maxx, n_states, x)
    a = action(state, 0, Q)
    c = control(theta0, theta1, a, 50)
    return c


def simulation(ita, dt, T, x0, n_states, theta0, theta1):
    n = int(T / dt)
    dB = np.zeros(n)
    X = np.zeros(n)

    for i in range(n - 1):
        dB[i] = dW(dt)
        X[i + 1] = X[i] + b(X[i], i, theta0, theta1) * dt + dB[i]

    # Initialize the Q-table to 0. We have two actions and T/dt observations.
    Q = np.zeros((n_states + 2, 2))
    print(Q)
    print("n =", n)

    # initialize the exploration probability to 1
    exploration_proba = 1

    # exploartion decreasing decay for exponential decreasing
    exploration_decreasing_decay = 0.001

    # minimum of exploration proba
    min_exploration_proba = 0.01

    # discounted factor
    gamma = 0.99

    # learning rate
    lr = 0.1

    rewards_per_episode = list()
    switching_times = list()
    b1 = np.zeros(n)
    delta = 0
    states = np.zeros(n)
    a = action(x0, exploration_proba, Q)
    minx, maxx = getbounds(X)
    # minx = -2
    # maxx = 2
    print("Intervals for ", (n_states + 2), "states:")
    dtt = abs(maxx - minx) / n_states
    print("(-inf, ", minx, "]")
    for ii in range(n_states):
        print("[", ii * dtt + minx, ", ", (ii + 1) * dtt + minx, "]")
    print("[, ", maxx, ", inf)")
    print("min = ", minx, "max =", maxx)
    Q[0, 0] = 1
    Q[n_states + 1, 1] = 1
    for i in progressbar(range(ita)):
        # we initialize the first state of the episode
        current_c_state = x0  # continuous state
        current_state = get_state(minx, maxx, n_states, x0)
        total_episode_reward = 0
        states = np.zeros(n)
        pre_t = 0
        count = 0
        avg_delta = 0
        min_theta0 = 0
        max_theta1 = 0
        switching_times.clear()
        for t in range(n - 1):
            pre_a = a
            a = action(current_state, exploration_proba, Q)

            states[t] = current_c_state

            # The environment runs the chosen action and returns
            # the next state, a reward and true if the epiosed is ended.
            # next_state, reward, done, _ = env.step(action)

            if pre_a != a:
                switching_times.append(t)
                count += 1
                delta = current_c_state
                avg_delta += delta
                pre_t = 0

            b2 = control(theta0, theta1, a, pre_t)
            if b2 < 0.0 and min_theta0 > b2:
                min_theta0 = b2
            if b2 > 0.0 and max_theta1 < b2:
                max_theta1 = b2

            pre_t += 1

            # next_c_state = X[t] + b2 * dt + dB[t]
            next_c_state = current_c_state + b2 * dt + dB[t]
            next_state = get_state(minx, maxx, n_states, next_c_state)
            reward = next_c_state * next_c_state
            # if current_state < n_states+1 and 0 < current_state:
            Q[current_state, a] = (1 - lr) * Q[current_state, a] + lr * (reward + gamma * min(Q[next_state, :]))

            current_c_state = next_c_state
            current_state = next_state

        # We update the exploration proba using exponential decay formula
        exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay * i))
        total_episode_reward = expected_cost(states, dt, T)
        rewards_per_episode.append(total_episode_reward)
        avg_delta = avg_delta / count

    return delta, avg_delta, rewards_per_episode, states, Q, dB, X, min_theta0, max_theta1, switching_times


def euler_maruyama(a, b, dt, T, x0):
    n = int(T / dt)  # Number of time steps.
    print(T)
    # n = 1000
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = x[i] + b(x[i], i) * dt + a(x[i], i) * dW(dt)
    return x


def euler_maruyama2(b, dt, T, x0, dB):
    n = int(T / dt)  # Number of time steps.
    print(T)
    # n = 1000
    x = np.zeros(n)
    x[0] = x0
    for i in range(n - 1):
        x[i + 1] = x[i] + b(x[i], i) * dt + dB[i]
    return x


def expected_cost(y, dt, T):
    y1 = list(map(lambda number: number ** 2, y))
    c = np.trapz(y1)
    c = c / T
    return np.mean(c) * dt


def optimal_cost(theta0, theta1):
    return 0.25 * (1 / theta0 ** 2 + 1 / theta1 ** 2)


def optimal_delta(theta0, theta1):
    return 0.5 * (1 / theta0 + 1 / theta1)

def eval_last_steps(trajectory, steps, dt):
    i = len(trajectory) - steps
    new_trajectory = np.zeros(steps)
    print("steps:", i, steps, len(new_trajectory))
    for s in range(steps-1):
        new_trajectory[s] = trajectory[i+s]
    y = list(map(lambda number: number ** 2, new_trajectory))
    cost = np.trapz(y)
    cost = cost / steps
    print("Cost of ", steps, "is ", cost)
    fig, ax = plt.subplots(1, 1, figsize=(32, 16))
    x = np.linspace(0., int(steps*dt), steps)
    line, = ax.plot(x, new_trajectory, lw =2)
    fig.show()
    plt.close()
    return new_trajectory, cost




def main():
    print("Hello World!")
    sigma = 1.  # Standard deviation.
    mu = 10.  # Mean.
    tau = .05  # Time constant.

    dt = 0.01  # Time step.
    T = 100  # Total time.
    n = int(T / dt)  # Number of time steps.
    t = np.linspace(0., T, n)  # Vector of times.
    ita = 50000
    ii = np.linspace(0., ita / 10, ita)
    x0 = 0
    n_states = 20

    theta0 = -1.0
    theta1 = 1.0
    b1 = lambda x, i: b(x, i, theta0, theta1)

    # x2 = euler_maruyama(a1, b1, dt, T, 0)

    delta, avg_delta, rewards_per_epi, states, Q, dB, X, min_theta0, max_theta1, switching_times = simulation(ita, dt,
                                                                                                              T, x0,
                                                                                                              n_states,
                                                                                                              theta0,
                                                                                                              theta1)
    b2 = lambda x, i: b_Q(x, i, Q, n_states, min(X), max(X), theta0, theta1)
    x3 = euler_maruyama2(b2, dt, T, x0, dB)
    print("delta =", delta)
    print("avg_delta =", avg_delta)
    print("Optimal delta =", optimal_delta(theta0, theta1))
    fig1, ax1 = plt.subplots(1, 1, figsize=(32, 16))
    fig2, ax2 = plt.subplots(1, 1, figsize=(32, 16))
    fig3, ax3 = plt.subplots(1, 1, figsize=(32, 16))
    fig4, ax4 = plt.subplots(1, 1, figsize=(32, 16))
    ax1.plot(t, states, lw=2)
    ax2.plot(t, X, lw=2)
    ax3.plot(ii, rewards_per_epi, lw=2)
    ax4.plot(t, x3, lw=2)
    plt.show()
    print("Q = ", Q)
    print("Expected Cost with Q-learning: ", expected_cost(states, dt, T))
    print("Expected Cost with optimal b: ", expected_cost(X, dt, T))
    print("Q-learning Cost: ", expected_cost(x3, dt, T))
    print("Optimal Cost:", optimal_cost(theta0, theta1))
    print("thetas = ", theta0, theta1, min_theta0, max_theta1)
    # print("switching times:", switching_times)


if __name__ == "__main__":
    # main()

    path = '/Users/philip/Library/Mobile Documents/com~apple~CloudDocs/Documents/Vorlesungen/Skript/Masterarbeit/SDE_Playground/net1.pt'
    dt = 0.0001  # Time step.
    T = 1000  # Total time.
    frameskip = 100
    n = int(T / (dt*frameskip))  # Number of time steps.

    t = np.linspace(0., T, n)  # Vector of times.
    ita = n
    ii = np.linspace(0., ita / 10, ita)
    x0 = 0
    n_states = 20
    theta0 = -6
    theta1 = 1
    eps = 0.01
    r_buff_size = 20000

    total_rewards = []
    total_reward = 0
    minCost = 1000


    record = True

    done = False


    parser = argparse.ArgumentParser()
    parser.add_argument("--mps", default=False, action="store_true", help="activate M1 GPU")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Name of the env, Strandard=" + DEFAULT_ENV_NAME)
    args = parser.parse_args()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS device found.")
    else:
        print("MPS device not found.")
        device = torch.device("mps" if args.mps else "cpu")
    # device = torch.device("cpu")
    device2 = torch.device("cpu")
    # print("mps" if args.mps else "cpu")
    print("device = ", device)

    env = Simulation.Env(0, T, frameskip=frameskip, dt=dt, start=x0, theta0=theta0, theta1=theta1)
    buffer = ac.ExperienceBuffer(r_buff_size)
    pbuffer = ac.ExperienceBuffer(0)
    net = ac.DQN().to(device)
    tgt_net = ac.DQN().to(device)
    writer = SummaryWriter(comment="-" + args.env)
    print(net)
    trajectory = []
    agent = ac.Agent(env, exp_buffer=buffer)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    act = 0

    # plt.ion()
    # fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    # x = np.linspace(0., 100, ita)
    # line, = ax.plot(x, np.zeros(ita), lw =2)
    # fig.show()
    agent.eps = 1.0

    epsilon = 1.0

    for s in progressbar(range(ita)):
        # print(s, env.trajectory)
        # epsilon = max(eps, 1 - s / 10000)
        if 10002 > s:
            agent.eps = max(eps, 1 - s / 10000)






        if env.t >= env.max_T:
            print("reseting env")
        reward, trajectory = agent.play_step(net, device)

        if reward is not None:
            print("reward = ", reward)
            total_rewards.append(reward)

        if len(buffer) < r_buff_size:
            continue

        if s %1000 == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(2**10)
        loss_t = agent.cal_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()


    print("env tr = ", env.trajectory)
    print("tr = ", trajectory)
    if len(trajectory) == 0:
        trajectory = list(chain.from_iterable(env.trajectory))
    else:
        trajectory = list(chain.from_iterable(trajectory))
    # print(trajectory)
    fig1, ax1 = plt.subplots(1, 1, figsize=(32, 16))
    ax1.plot(np.linspace(0., int(T), len(trajectory)), trajectory, lw=2,label="$X_t$")
    plt.rcParams['text.usetex'] = True
    ax1.legend()
    plt.show()
    print(total_reward)
    print("loss = ", loss_t)
    print("cost = ", expected_cost(trajectory, dt, T=100))
    eval_last_steps(trajectory, 10000, dt)
    # costs, cost, traj = env.eval_steps(trajectory = trajectory, s = 10000)
    # print("cost = ", costs, cost)
    print("Timesteps = ", n)
    # torch.save(net, path)
