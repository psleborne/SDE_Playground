import numpy as np
import matplotlib.pyplot as plt
import math
import random
from itertools import chain
import asyncio

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


class Env:

    def __init__(self, delta, max_T, cost=np.square, frameskip=10000, dt=10.0 ** (-5), start=0.0, theta0=-1, theta1=1):
        self.delta = delta
        self.max_T = max_T
        self.cost = cost
        self.frameskip = frameskip
        self.dt = dt
        self.X = start
        self.theta0 = theta0
        self.theta1 = theta1

        self.done = False
        self.pos_cost = 0
        self.total_cost = 0
        self.t = 0
        self.trajectory = []
        # self.optimal_trajectory = []
        self.regrets = np.zeros(int(max_T))

    def step(self, action, record=True):
        X, pos_cost, trajectory, opt_trajectory = self.simulate(dt=self.dt, cost=self.cost, steps=self.frameskip,
                                                                start=self.X,
                                                                action=action, record=record)

        self.X = X
        self.t += self.dt * float(self.frameskip)
        self.pos_cost = pos_cost
        self.total_cost += pos_cost
        if record:
            self.trajectory.append(trajectory)
            # self.optimal_trajectory.append(opt_trajectory)
        if self.t > self.max_T:
            self.done = True

        return X, pos_cost, self.done

    def step_d(self, record=True):
        X, pos_cost, trajectory = self.simulate_d(dt=self.dt, cost=self.cost, steps=self.frameskip, start=self.X,
                                                  record=record)
        self.X = X
        self.t += self.dt * float(self.frameskip)
        self.pos_cost += pos_cost
        if record:
            self.trajectory.append(trajectory)
        if self.t > self.max_T:
            self.done = True

        return X, pos_cost, self.done

    def reset(self, start):
        self.X = start
        self.done = False
        self.pos_cost = 0
        self.total_cost = 0
        self.t = 0
        self.trajectory = []

        return start, False

    def _drift_delta(self, X):
        theta0, theta1 = self.theta0, self.theta1
        return theta0 if self.delta < X else theta1

    def optimal_delta(self):
        return 0.5 * (1.0 / self.theta0 + 1.0 / self.theta1)

    def cal_delta(self, t0, t1):
        return 0.5 * (1.0 / t0 + 1.0 / t1)

    def optimal_cost(self):
        theta0 = self.theta0
        theta1 = self.theta1
        return 0.25 * (1.0 / (theta0 ** 2) + 1.0 / (theta1 ** 2))

    def _drift_action(self, action):
        theta0, theta1 = self.theta0, self.theta1
        return theta0 if action == 0 else theta1

    def simulate(self, dt, cost, steps, start, action, record):
        effort = 0
        X = start
        # X_opt = start
        pos_cost = 0
        trajectory = []
        opt_trajectory = []

        delta = self.optimal_delta()

        for i in range(steps):
            dY = np.random.normal(loc=0.0, scale=np.sqrt(dt))
            Y = X + self._drift_action(action) * dt + dY
            # Y_opt = X_opt + self._drift_delta(X_opt) * dt + dY
            if record:
                trajectory.append(Y)
                # opt_trajectory.append(Y_opt)
            X = Y
            # X_opt = Y_opt
            # pos_cost += dt * cost(X)
            pos_cost += cost(X)/steps

        return X, pos_cost, trajectory, opt_trajectory

    def simulate_d(self, dt, cost, steps, start, record):
        effort = 0
        X = start
        pos_cost = 0
        trajectory = []

        for i in range(steps):
            dY = np.random.normal(loc=0.0, scale=np.sqrt(dt))
            Y = X + self._drift_delta(X) * dt + dY
            if record:
                trajectory.append(Y)
            X = Y
            pos_cost = dt * cost(X) * dt

        return X, pos_cost, trajectory

    def plot_sim(self):
        print(self.trajectory)

    # Evaluates the steps trajectory from time s until terminal time T
    def eval_steps(self, trajectory = None, s=-1.0, T=-1.0):
        if trajectory == None:
            trajectory = list(chain.from_iterable(self.trajectory))
        late_start = False
        early_end = False
        if s == -1.0:
            s = self.max_T
        else:
            late_start = True
        if T != -1.0:

            early_end = True

        steps1 = int(math.floor(s) / self.dt)
        if steps1 <= 0:
            return -1.0, 0.0, 0.0
        if late_start:
            i = len(trajectory) - steps1
            new_trajectory = np.zeros(steps1)
            for t in range(steps1):
                new_trajectory[t] = trajectory[i + t]
            new_trajectory = list(map(lambda number: number ** 2, new_trajectory))

            cost = np.trapz(new_trajectory)
            return cost / steps1, cost*self.dt, new_trajectory
        elif early_end:
            steps1 = int(math.floor(T) / self.dt)
            new_trajectory = np.zeros(steps1)
            for t in range(steps1):
                new_trajectory[t] = trajectory[t]
            new_trajectory = list(map(lambda number: number ** 2, new_trajectory))
            cost = np.trapz(new_trajectory)
            return cost / steps1, cost*self.dt, new_trajectory

        else:
            trajectory = list(map(lambda number: number ** 2, trajectory))
            cost = np.trapz(trajectory)
        return cost / steps1, cost*self.dt, trajectory

    @background
    def _aux_func(self, t, optimal_trajectory, trajectory, s):
        d_expected_cost, expected_cost, _ = self.eval_steps(optimal_trajectory, s, T=t+1)
        d_actual_cost, actual_cost, _ = self.eval_steps(trajectory, s, T=t+1)
        self.regrets[t] = math.fabs(expected_cost - actual_cost)

    def get_mean_cost(self):
        return self.total_cost * self.dt

    def get_regrets(self,optimal_trajectory, trajectory, s):
        # for t in range(int(self.max_T)):
        #     d_expected_cost, expected_cost, _ = self.eval_steps(optimal_trajectory, s, T=t+1)
        #     d_actual_cost, actual_cost, _ = self.eval_steps(trajectory, s, T=t+1)
        #     regrets[t] = math.fabs(d_expected_cost - d_actual_cost)

        loop = asyncio.get_event_loop()                                              # Have a new event loop

        looper = asyncio.gather(*[self._aux_func(t, optimal_trajectory, trajectory, s) for t in range(self.max_T)])         # Run the loop

        results = loop.run_until_complete(looper)


# return regrets

    def regret(self, s=-1.0, show=False):

        n = len(self.optimal_trajectory)
        if n != len(self.trajectory):
            print("Problem! Lengths of trajectories are unequal")
            return -1, []
        else:
            # trajectory = list(chain.from_iterable(self.env.trajectory))

            optimal_trajectory = list(chain.from_iterable(self.optimal_trajectory))
            trajectory = list(chain.from_iterable(self.trajectory))

            # regrets = list(map(lambda a, b: math.fabs(a - b), optimal_trajectory, trajectory))
            # regrets = np.zeros(self.max_T + 2)
            # regrets[0] = 0

            # regrets = self.get_regrets(optimal_trajectory, trajectory)

            self.get_regrets(optimal_trajectory, trajectory, s)



            d_expected_cost, expected_cost, _ = self.eval_steps(optimal_trajectory, s)
            d_actual_cost, actual_cost, _ = self.eval_steps(trajectory, s)
            r = math.fabs(expected_cost - actual_cost)

            return r, self.regrets

    def test_env(self, set_delta = False):
        print("Testing environment with parameters: ")
        print("Theta0 = ", self.theta0)
        print("Theta1 = ", self.theta1)
        if not set_delta:
            temp = self.delta
            self.delta = self.optimal_delta()


        d_line = [self.delta] * int(self.max_T)
        self.reset(0)
        while not self.done:
            # act = 0 if delta < self.X else 1
            X, pos_cost, _ = self.step_d()

        trajectory = list(chain.from_iterable(self.trajectory))
        n_traject = trajectory[::100]
        n_t = np.linspace(0., self.max_T, len(n_traject))
        t = np.linspace(0., self.max_T, len(trajectory))
        fig1, ax1 = plt.subplots(1, 1, figsize=(32, 16))
        ax1.plot(n_t, n_traject, lw=2)
        ax1.plot(np.linspace(0, int(self.max_T), len(d_line)), d_line, color='b', lw=4, label="delta="+str(round(self.delta,2)))
        ax1.legend()
        plt.show()
        if not set_delta:
            self.delta = temp

        print(pos_cost)
