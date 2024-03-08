import math
import random

import matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar
import Simulation
from itertools import chain

matplotlib.rcParams.update({'font.size': 22})


# %matplotlib inline

class SimpleQ:

    def __init__(self, ita, env, num_of_states, upperb, lowerb):
        self.ita = ita
        self.env = env
        self.num_of_states = num_of_states
        self.upperb = upperb
        self.lowerb = lowerb

    def dW(self, delta_t: float) -> float:
        """Sample a random number at each call."""
        return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

    def a1(self, x, i):
        return 1.0

    def b(self, x, i):
        theta0, theta1 = self.env.theta0, self.env.theta1
        delta = 1 / (2 * theta0) + 1 / (2 * theta1)
        if x > delta:
            return theta0
        else:
            return theta1

    # Action is 0 or 1, either drift theta0 or theta1
    def action(self, current_state, eps, Q):
        r = random.random()
        if current_state > 100:
            return 0
        elif current_state < -100:
            return 1
        elif r < eps:
            return round(r)
        else:
            return np.argmin(Q[current_state, :])

    def control(self, a, t):
        theta0, theta1 = self.env.theta0, self.env.theta1
        #     a = 0: action is to turn left
        #     a = 1: action is to turn right
        if a == 0:
            # print("Theta0 = ", theta0)
            # b = theta0 - 1.0 / (-1 * math.sqrt(t) - 1 / math.sqrt(-1.0 * theta0))
            # b = theta0 + 1.0 / ((-1 * t - 1 / math.sqrt(-1.0 * theta0)) * (-1 * t - 1 / math.sqrt(-1.0 * theta0)))
            b = theta0
        else:
            # b = theta1 - 1.0 / (math.sqrt(t) + 1 / math.sqrt(theta1))
            # b = theta1 - 1.0 / ((t + 1 / math.sqrt(theta1)) * (t + 1 / math.sqrt(theta1)))
            b = theta1
        return b

    def getbounds(self, X):
        maxx = max(X)
        minx = min(X)

        return minx, maxx

    def get_state(self, n: int, x: float):

        minx = self.lowerb
        maxx = self.upperb
        if n == 0:
            ds = 0
        else:
            ds = abs(maxx - minx) / n
        if x >= maxx:
            return n + 1
        elif x <= minx:
            return 0
        else:
            for i in range(n + 1):

                if x >= (i) * ds + minx and x < (i + 1) * ds + minx:
                    return i + 1

        print("NoneFound x =", x)
        return 1

    def b_Q(self, x, t, Q, minx, maxx):
        state = self.get_state(minx, maxx, self.num_of_states, x)
        a = self.action(state, 0, Q)
        c = self.control(a, 50)
        return c

    def q_learn(self, start, Q):

        # initialize the exploration probability to 1
        exploration_proba = 1.0

        # exploartion decreasing decay for exponential decreasing
        exploration_decreasing_decay = 0.001

        # minimum of exploration proba
        min_exploration_proba = 0.01

        # discounted factor
        gamma = 0.9

        # learning rate
        lr = 0.01

        rewards_per_episode = list()

        delta = 0



        print("Intervals for ", (self.num_of_states + 2), "states:")
        dtt = abs(self.upperb - self.lowerb) / self.num_of_states
        print("(-inf, ", self.lowerb, "]")
        for ii in range(self.num_of_states):
            print("[", round(ii * dtt + self.lowerb, 2), ", ", round((ii + 1) * dtt + self.lowerb, 2), "]")
        print("[, ", self.upperb, ", inf)")
        print("min = ", self.lowerb, "max =", self.upperb)

        Q_old = Q
        for i in progressbar(range(self.ita)):
            # we initialize the first state of the episode
            current_state = self.get_state(self.num_of_states, start)
            total_episode_reward = 0

            while not self.env.done:
                action = self.action(current_state, exploration_proba, Q)

                # The environment runs the chosen action and returns
                # the next state, a reward and true if the episode is ended.

                X, reward, _ = self.env.step(action, True)

                next_state = self.get_state(self.num_of_states, X)

                reward = self.env.pos_cost

                Q[current_state, action] = ((1 - lr) * Q[current_state, action] +
                                            lr * (reward + gamma * min(Q[next_state, :])))

                current_state = next_state

            # We update the exploration proba using exponential decay formula
            trajectory = self.env.trajectory
            t_reward = self.env.total_cost
            self.env.reset(start)
            exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay * i))
            m=1
            if i > 900:
                m = np.min(rewards_per_episode)
                if t_reward > m:
                    Q = Q_old
                    # print("total_episode_reward > m", total_episode_reward, m)
                    continue

            if t_reward > 0:
                if i > 899:
                    print("m = ", m,t_reward)
                    rewards_per_episode.append(t_reward)

        states = list(chain.from_iterable(self.env.trajectory))

        return t_reward, rewards_per_episode, states, Q, trajectory

    # def euler_maruyama(self, a, b, dt, T, x0):
    #     n = int(T / dt)  # Number of time steps.
    #     print(T)
    #     # n = 1000
    #     x = np.zeros(n)
    #     x[0] = x0
    #     for i in range(n - 1):
    #         x[i + 1] = x[i] + b(x[i], i) * dt + a(x[i], i) * dW(dt)
    #     return x

    def euler_maruyama2(self, b, dt, T, x0, dB):
        n = int(T / dt)  # Number of time steps.
        print(T)
        # n = 1000
        x = np.zeros(n)
        x[0] = x0
        for i in range(n - 1):
            x[i + 1] = x[i] + b(x[i], i) * dt + dB[i]
        return x

    def expected_cost(self, y, dt, T):
        y1 = list(map(lambda number: number ** 2, y))
        c = np.trapz(y1)
        c = c / T
        return np.mean(c) * dt

    def estimate_delta(self, Q):
        n_states = len(Q)
        if Q[0][0] > Q[0][1]:
            a = 1
        else:
            return False, 0, 0, 0
        count = 0
        d = 0

        minx = self.lowerb
        maxx = self.upperb
        if n == 0:
            ds = 0
        else:
            ds = abs(maxx - minx) / (n_states - 2)

        for s in range(n_states - 2):

            if a == 1 and Q[s + 1][0] < Q[s + 1][1]:
                count += 1
                a = 0
                if count == 1:
                    print(s + 1, ds, n_states)

                    x_min = minx + math.floor((s + 2) * 0.5) * ds
                    x_max = minx + math.floor((s + 5) * 0.5) * ds
                    d = (x_min + x_max) / 2.0
                    print("test1:", True, d, x_min, x_max)

            if a == 0 and Q[s + 1][0] > Q[s + 1][1]:
                count += 1
                a = 1
        if count == 1:
            return True, d, x_min, x_max
        else:
            return False, d, 0, 0

    def exploitation(self, Q, start):
        found_delta, delta, xmin, xmax = self.estimate_delta(Q)
        if found_delta:
            self.env.reset(start)
            self.env.delta = delta
            done = False
            while not done:
                _, _, done = self.env.step_d(record=True)
            return self.env.pos_cost, found_delta
        return 0, found_delta

    def iteration(self, start, Q, max_iter):

        done = False
        count = 0
        t_reward, rewards_per_epi, states, Q, trajectory = self.q_learn(start, Q)
        expl_cost = self.exploitation(Q, start)
        delta_est = 0
        while not done:
            count += 1
            if expl_cost != -1:
                delta_est = self.estimate_delta(Q)
                print("Exploitation cost = ", expl_cost)

    def run(self, start):
        print("Running normal Q-learning")

        dt = self.env.dt  # Time step.
        T = self.env.max_T  # Total time.
        n = int(T / dt)  # Number of time steps.
        t = np.linspace(0., T, n)  # Vector of times.
        ita = self.ita
        ii = np.linspace(0., ita / 10, ita)
        n_states = self.num_of_states

        theta0 = self.env.theta0
        theta1 = self.env.theta1
        b1 = lambda x, i: self.b(x, i)

        # x2 = euler_maruyama(a1, b1, dt, T, 0)
        # Initialize the Q-table to 0. We have two actions and T/dt observations.
        Q = np.zeros((n_states + 2, 2))
        print(Q)

        t_reward, rewards_per_epi, states, Q, trajectory = self.q_learn(start, Q)
        # b2 = lambda x, i: self.b_Q(x, i, Q, min(X), max(X))
        # x3 = self. euler_maruyama2(b2, dt, T, x0, dB)
        # print("delta =", delta)

        print("Optimal delta =", self.env.optimal_delta())

        trajectory = list(chain.from_iterable(trajectory))
        fig1, ax1 = plt.subplots(1, 1, figsize=(32, 16))
        # fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
        fig3, ax3 = plt.subplots(1, 1, figsize=(32, 16))
        # fig4, ax4 = plt.subplots(1, 1, figsize=(8, 4))
        # ax1.plot(t, states, lw=2)
        fig2, ax2 = plt.subplots(1, 1, figsize=(32, 16))
        ax2.plot(np.linspace(0., int(T), len(trajectory)), trajectory, lw=2, label="X_t")
        ii = np.linspace(0., len(rewards_per_epi), len(rewards_per_epi))
        ax3.plot(ii, rewards_per_epi, lw=2)
        # ax4.plot(t, x3, lw=2)
        plt.show()
        print("test3", rewards_per_epi[0], np.min(rewards_per_epi))
        print("Q = ", Q)
        print("Expected Cost with Q-learning: ", self.expected_cost(trajectory, dt, T), t_reward)
        # print("Expected Cost with optimal b: ", self.optimal_cost())
        # print("Q-learning Cost: ", self.expected_cost(x3, dt, T))
        print("Optimal Cost:", self.env.optimal_cost())
        print("thetas = ", theta0, theta1)

        print("estimating delta with Q-values:")
        print("Q-est delta = ", self.estimate_delta(Q))
        # print("switching times:", switching_times)
        expl_cost, found_delta = self.exploitation(Q, start)

        stop = False
        rounds = 0
        while not stop:

            print("rounds = ", rounds)
            if rounds > 20:
                stop = True

            if found_delta:
                print("Exploitation cost = ", expl_cost)
                _, self.env.delta, _, _ = self.estimate_delta(Q)
                self.env.test_env(True)
                rounds += 10
                _, _, lowerb, upperb = self.estimate_delta(Q)
                self.lowerb = lowerb - 0.1
                self.upperb = upperb + 0.1
                print("estimating delta with Q-values:")
                print("Q-est delta = ", self.env.delta)
                print("Refining bounds to ", self.lowerb, self.upperb)

                n_states = 3
                self.num_of_states = n_states
                Q = np.zeros((n_states + 2, 2))
                # self.env.dt = 0.001
                delta, rewards_per_epi, states, Q, _ = self.q_learn(start, Q)
                expl_cost, found_delta = self.exploitation(Q, start)

            else:
                print("No conclusive delta found")
                print("Iterating again")


                # Q = 0.0001 * Q
                print(Q)
                delta, rewards_per_epi, states, Q, _ = self.q_learn(start, Q)
                expl_cost, found_delta = self.exploitation(Q, start)
                print("estimating delta with Q-values:")
                print("Q-est delta = ", self.estimate_delta(Q))
                print("Exploitation cost = ", expl_cost)
                rounds += 2


if __name__ == "__main__":
    dt = 0.01  # Time step.
    T = 100  # Total time.
    n = int(T / dt)  # Number of time steps.
    t = np.linspace(0., T, n)  # Vector of times.
    ita = 5000
    # ii = np.linspace(0., T, ita*T/dt)
    x0 = 0
    n_states = 5
    theta0 = -6.0
    theta1 = 1.0
    eps = 0.001
    lowerb = -2
    upperb = 2

    total_rewards = []
    total_reward = 0
    minCost = 1000

    record = True

    done = False

    # Testing env:

    env = Simulation.Env(0, T, frameskip=1, dt=dt, start=x0, theta0=theta0, theta1=theta1)
    env.test_env()

    sQ = SimpleQ(ita, env, n_states, upperb=upperb, lowerb=lowerb)
    sQ.run(0)
