import math
import random

import matplotlib
import tensorflow as tf
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar
import Simulation
from itertools import chain
from scipy.optimize import curve_fit
matplotlib.rcParams.update({'font.size': 42})


def func(x, a, b): # Sigmoid A With Offset from zunzun.com
    # return a * np.exp(-b * x) + c
    # return  1.0 / (1.0 + np.exp(-a * (x-b))) + c
    # return
    return a*x + b

class ExporeFirst:

    def __init__(self, K, env, record = True):
        self.K = K
        self.n = math.floor(math.sqrt(env.max_T))
        self.env = env
        self.record = record
        # self.tau = np.zeros(2*self.n+1)
        self.tau0 = 0
        self.tau1 = 0
        self.xData = []

    def reset(self):
        self.env.reset(-1.0*self.K)
        # self.K = K
        # self.tau = np.zeros(2*self.n+1)


    def exploration(self):
        h = 1
        tau0, tau1, tau_temp= 0,0,0
        # self.tau[0] = 0
        drift = 1
        X,_,_ = self.env.step(drift, self.record)
        # print("Starting exploration phase")
        while h <= 2*self.n: #We iterate until we have returned from one barrier to the other 2n times
            if drift == 1:
                if X >= self.K:
                    tau1 += self.env.t - tau_temp
                    tau_temp = self.env.t
                    # self.tau[h] = self.env.t - self.env.dt
                    drift = 0
                    h += 1
            else:
                if X <= -1.0 * self.K:
                    drift = 1
                    tau0 += self.env.t - tau_temp
                    tau_temp = self.env.t
                    # self.tau[h] = self.env.t - self.env.dt
                    h += 1

            X,_,_ = self.env.step(drift, self.record)
            self.tau0 = tau0
            self.tau1 = tau1
            if self.env.done:
                break
        return self.env.t


    # def estimation(self):
    #     tau_0, tau_1 = 0.0, 0.0
    #
    #     for k in range(self.n):
    #         tau_1 += (self.tau[2*k + 1] - self.tau[2*k])
    #         tau_0 += (self.tau[2*k + 2] - self.tau[2*k + 1])
    #
    #     return tau_0/self.n, tau_1/self.n

    def get_delta(self):
        # tau0, tau1 = self.estimation()
        # print("1: ", tau0, tau1)
        # print("2:", self.tau0, self.tau1)
        return (self.tau1 - self.tau0) / (self.n * 4.0 * self.K)

    def get_theta(self, i):
        tau0, tau1 = self.tau0, self.tau1
        if i == 0:
            return -2.0 * self.K / tau0
        elif i == 1:
            return 2.0 * self.K / tau1







    def exploitation(self):
        # print("Starting exploitation phase")
        done = self.env.done
        delta = self.get_delta()
        self.env.delta = delta

        while not done:
            _, _, done = self.env.step_d(record=True)
        return self.env.pos_cost

    def eval_last_steps(self, trajectory, steps):
        steps1 = int(math.floor(steps)/self.env.dt)
        if steps1 <= 0:
            return -1
        i = len(trajectory) - steps1
        # print("i = ", i)
        new_trajectory = np.zeros(steps1)
        for s in range(steps1):
            new_trajectory[s] = trajectory[i+s]
        y = list(map(lambda number: number ** 2, new_trajectory))
        cost = np.trapz(y)

        cost = cost / steps1
        return cost


    def optimal_cost(self):
        return 0.25 * (1.0 / (self.env.theta0 ** 2) + 1.0 / (self.env.theta1 ** 2))
    def optimal_delta(self):
        return 0.5 * (1.0 / self.env.theta0 + 1.0 / self.env.theta1)
    def _aux_mult(self, a, b):
        opt_cost = self.optimal_cost()
        return b*math.fabs(a - opt_cost)

    def log_result(self, result):
        # This is called whenever foo_pool(i) returns a result.
        # result_list is modified only by the main process, not the pool workers.
        self.xData.append(result)

    def _aux_mult2(self, a, b):
        opt_cost = self.optimal_cost()
        return a

    def expect_cost_sim(self, T):
        ita = 100
        exp_cost = 0
        max_T = self.env.max_T
        self.env.max_T = T
        # manager = multiprocessing.Manager()
        # return_value = manager.dict()
        # processes = []
        print("t = ", T)

        for i in range(ita):

            self.reset()
            self.exploitation()
            self.exploitation()
            c,u_c,_ = self.env.eval_steps()

            exp_cost += c
        exp_cost =  exp_cost / ita
        self.env.max_T = max_T
        # print("t = ", T, "(", exp_cost, ",", T, ")")
        return exp_cost, T

    def run(self, output):
        T = self.env.max_T


        # env = Simulation.Env(0, T, frameskip=frameskip, dt=dt, start=start, theta0=theta0, theta1=theta1)
        # expl_first = ExporeFirst(K=K, n=n, env=env, record=record)
        self.exploration()
        t = self.env.t
        if output:
            print("t = ", int(t))
            print("T = ", self.env.max_T)
            print("is done =", self.env.done)

        self.exploitation()



        trajectory = list(chain.from_iterable(self.env.trajectory))
        temp = int(round(T - t))
        if output:
            print("t = ", self.env.t)
            print("is done =", self.env.done)
            print("temp = ", temp)


            self.xData = []
            costs = []

            pool = multiprocessing.Pool()
            # for i in range(ita):
            #     pool.apply(self._aux_expcost, args = (i, ), callback = self.log_result)
            pool.map_async(self.expect_cost_sim, range(int(T)), callback = self.log_result)
            pool.close()
            pool.join()
            print('results = ', self.xData)
            costs = [self._aux_mult2(a, b) for a,b in self.xData[0]]
            regrets = [self._aux_mult(a, b) for a,b in self.xData[0]]
            print('regrets = ', regrets)
            print("costs = ", costs)
            regrets = np.sqrt(regrets)

            t2 = np.arange(0.1, len(regrets), 1)

            fittedParameters, pcov = curve_fit(func, t2, regrets)
            # fittedParameters2 ,_ = curve_fit(func2, t3, d_regret)
            modelPredictions = func(t2, *fittedParameters)
            # modelPredictions2 = func2(t2, *fittedParameters2)
            print(fittedParameters)

            K_line0 = [self.K]*int(t)
            K_line1 = [-1.0*self.K]*int(t)
            fig1, ax1 = plt.subplots(1, 1, figsize=(32, 16))
            fig3, ax3 = plt.subplots(1, 1, figsize=(32, 16))
            ax1.plot(np.linspace(0., int(T), len(trajectory)), trajectory, lw=2, label="$X_t$")
            ax1.plot(np.linspace(0., int(t), int(t)), K_line0, color='r', lw=3, label="$K=$"+str(self.K))
            ax1.plot(np.linspace(0., int(t), int(t)), K_line1, color='r', lw=3)
            if temp >= 0:
                d_line = [self.get_delta()]*temp
                ax1.plot(np.linspace(int(t), int(T), temp), d_line, color='b', lw=4, label="$\hat{\delta}=$"+str(round(self.get_delta(),2)))
            # ax3.plot(np.linspace(0., len(regrets), len(regrets)), (regrets), lw=2, label="regret")
            # ax3.plot(np.linspace(0., len(regrets), len(modelPredictions)), (modelPredictions), color ='magenta' , lw=6, label="fitted r")
            ax3.legend()
            ax1.legend()
            plt.rcParams['text.usetex'] = True
            plt.show()
            print("n = ", self.n)
            print("delta = ", self.get_delta())

            print("est_theta0 = ", self.get_theta(0))
            print("est_theta1 = ", self.get_theta(1))

        cost = self.eval_last_steps(trajectory, temp)
        tot_cost = self.eval_last_steps(trajectory, T)


        return cost, tot_cost, self.get_delta(), self.get_theta(0), self.get_theta(1), t

    def test_K(self, minK, maxK, ds, iter):
        r = int((math.floor(maxK - minK)+1)/ds)
        s = minK
        rcost = np.zeros(r+1)
        tot_rcost = np.zeros(r+1)
        rdelta = np.zeros(r+1)
        i = 0
        s_times = np.zeros(r+1)

        i_costs = np.zeros((3, iter))

        min_cost = [1000.0, 0, -1]
        opt_K = 0.0
        i=0
        count = 0

        while s <= maxK:
            self.K = s

            count = 0
            ave_cost = 0
            ave_tot_c=0
            ave_d = 0
            for ii in range(iter):
                temp_c, temp_tot_c, temp_d,_ ,_, t = self.run(False)
                if temp_c >0:
                    count += 1
                    i_costs[0][ii], i_costs[1][ii], i_costs[2][ii] = temp_c, temp_tot_c, temp_d
                    ave_tot_c += temp_tot_c
                    ave_cost += temp_c
                    ave_d += temp_d

                self.env.reset(0)

            # print(i_costs[0])
            rcost[i] = ave_cost/count
            rdelta[i] = ave_d/count
            tot_rcost[i] = ave_tot_c/count
            print(i,s, round(rcost[i],4), round(tot_rcost[i],4), round(rdelta[i],4), round(t,4))
            s_times[i] = t
            if rcost[i] >= 0 and rcost[i] < min_cost[0]:
                min_cost = [rcost[i], rdelta[i], s_times[i]]
                opt_K = s
            s += ds
            i += 1
        return min_cost, opt_K









if __name__ == "__main__":
    theta0, theta1 = -6, 1
    T = 100
    dt = 0.01
    n = math.floor(math.sqrt(T))

    K = 3


    env = Simulation.Env(0, T, frameskip=1, dt=dt, start=(-1.0*K), theta0=theta0, theta1=theta1)
    expl_first = ExporeFirst(K=K, env=env, record=True)
    # delta = env.optimal_delta()
    # env.test_env()

    cost, tot_cost, delta, etheta0, etheta1, t = expl_first.run(True)

    print("cost = ", cost)
    # print("total cost = ", env.pos_cost)
    print("total cost = ", tot_cost)
    print("n = ", n)
    print("optimal cost = ", expl_first.optimal_cost())
    print("delta = ", delta)
    print("optimal delta = ", expl_first.optimal_delta())
    print("est. theta0 = ", etheta0, " theta1 = ", etheta1)

    # min_cost, opt_K = expl_first.test_K(0.2, 7.0, 0.2, 5)
    # print("[cost, delta, stime]", min_cost)
    # print("K = ", opt_K)









