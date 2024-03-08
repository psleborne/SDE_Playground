import math
import random
import asyncio
import multiprocessing

import scipy
from scipy.optimize import curve_fit

import matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from progressbar import progressbar
import Simulation
from itertools import chain
from scipy.optimize import differential_evolution
import warnings
matplotlib.rcParams.update({'font.size': 42})


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

def func(x, a, b): # Sigmoid A With Offset from zunzun.com
    # return a * np.exp(-b * x) + c
    # return  1.0 / (1.0 + np.exp(-a * (x-b))) + c
    # return
    return a*x + b

def func2(x, a, b, c): # Sigmoid A With Offset from zunzun.com
    return a * np.exp(-b * x) + c
    # return  1.0 / (1.0 + np.exp(-a * (x-b))) + c
    # return a/np.log(b*x) - c



class Adaptive_A:

    def __init__(self, K, env, record=True):
        self.K = K
        self.n = math.floor(math.sqrt(env.max_T))
        self.env = env
        self.record = record
        self.xData,self.yData = [], []

    def reset(self):
        self.env.reset(-1.0 * self.K)
        # self.K = K
        # self.tau = []

    def exploration(self):
        n_theta0 = 0
        d_theta0 = 1
        n_theta1 = 0
        d_theta1 = 1
        theta0 = 0
        theta1 = 0
        d = []
        K_listu = []
        K_listl = []

        tau0, tau1, tau2 = 0, 0, 0

        n = 0
        k_v = self.K
        delta_est = 0
        drift = 1

        X0 = self.env.X
        X, _, _ = self.env.step(drift, self.record)
        prev = k_v
        barrieru = k_v + delta_est
        barrierl = -1.0 * k_v+ delta_est

        while not self.env.done:

            barrieru = k_v + delta_est
            barrierl = -1.0 * k_v+ delta_est


            if drift == 1:
                if X >= barrieru:
                    prev = k_v + delta_est
                    n += 1
                    X1 = self.env.X
                    n_theta1 += X1 - X0
                    X0 = X1
                    tau1 = self.env.t
                    d_theta1 += tau1 - tau0
                    K_listu.extend([barrieru] * int((tau1 - tau0) / self.env.dt))
                    K_listl.extend([barrierl] * int((tau1 - tau0) / self.env.dt))
                    tau0 = tau1
                    drift = 0
                    theta1 = n_theta1 / d_theta1

                    if theta1 != 0 and theta0 != 0:
                        delta_est = self.compute_delta(theta0, theta1)
                        if k_v > self.env.dt + 0.01:
                            k_v -= 0.1*self.K
                            # k_v = 1.0 / n
                            # barrieru = k_v + delta_est
                            # barrierl = -1.0 * k_v+ delta_est
                            if prev < -1.0 * k_v + delta_est:
                                drift = 1
                                print("Error: ", prev, "<", -1.0 * k_v + delta_est, "t = ", self.env.t)

            else:
                if X <= barrierl:
                    prev = -1.0 * k_v + delta_est
                    n += 1
                    X1 = self.env.X
                    n_theta0 += X1 - X0
                    X0 = X1
                    tau1 = self.env.t
                    d_theta0 += tau1 - tau0
                    K_listu.extend([barrieru] * int((tau1 - tau0) / self.env.dt))
                    K_listl.extend([barrierl] * int((tau1 - tau0) / self.env.dt))
                    tau0 = tau1
                    drift = 1
                    theta0 = n_theta0 / d_theta0
                    if theta1 != 0 and theta0 != 0:
                        delta_est = self.compute_delta(theta0, theta1)
                        # 0.01 better that dt
                        if k_v > self.env.dt + 0.01:
                            k_v -= 0.1*self.K
                            # k_v = 1.0 / n
                            # barrieru = k_v + delta_est
                            # barrierl = -1.0 * k_v+ delta_est
                            if prev > k_v + delta_est:
                                drift = 0
                                print("Error: ", prev, ">", k_v + delta_est, "t = ", self.env.t)

            X, _, _ = self.env.step(drift, self.record)
            d.append(delta_est)

        return delta_est, d, K_listu, K_listl

    def exploration_light(self):
        n_theta0 = 0
        d_theta0 = 1
        n_theta1 = 0
        d_theta1 = 1
        theta0 = 0
        theta1 = 0

        tau0, tau1, tau2 = 0, 0, 0

        n = 0
        k_v_light = self.K
        delta_est = 0
        drift = 1
        X0 = self.env.X
        X, _, _ = self.env.step(drift, self.record)
        barrieru = k_v_light + delta_est
        barrierl = -1.0 * k_v_light + delta_est


        while not self.env.done:

            if drift == 1:

                if X >= barrieru:
                    n += 1
                    X1 = self.env.X
                    n_theta1 += X1 - X0
                    X0 = X1
                    tau1 = self.env.t
                    d_theta1 += tau1 - tau0
                    tau0 = tau1
                    drift = 0
                    theta1 = n_theta1 / d_theta1

                    if theta1 != 0 and theta0 != 0:
                        delta_est = self.compute_delta(theta0, theta1)
                        if k_v_light > self.env.dt + 0.01:
                            # self.K -= 0.1*self.K
                            k_v_light = 1.0 / n
                            barrieru = k_v_light + delta_est
                            barrierl = -1.0 * k_v_light + delta_est


            else:
                if X <= barrierl:
                    n += 1
                    X1 = self.env.X
                    n_theta0 += X1 - X0
                    X0 = X1
                    tau1 = self.env.t
                    d_theta0 += tau1 - tau0
                    tau0 = tau1
                    drift = 1
                    theta0 = n_theta0 / d_theta0
                    if theta1 != 0 and theta0 != 0:
                        delta_est = self.compute_delta(theta0, theta1)
                        # 0.01 better that dt
                        if k_v_light > self.env.dt + 0.01:
                            # self.K -= 0.1*self.K
                            k_v_light = 1.0 / n
                            barrieru = k_v_light + delta_est
                            barrierl = -1.0 * k_v_light + delta_est

            X, _, _ = self.env.step(drift, self.record)

        return delta_est

    def get_delta(self):
        theta0, theta1 = self.estimation()
        return 0.5 * (1.0 / theta0 + 1.0 / theta1)

    def compute_delta(self, t0, t1):
        return 0.5 * (1.0 / t0 + 1.0 / t1)

    def optimal_cost(self):
        return 0.25 * (1.0 / (self.env.theta0 ** 2) + 1.0 / (self.env.theta1 ** 2))

    def optimal_delta(self):
        return 0.5 * (1.0 / self.env.theta0 + 1.0 / self.env.theta1)

    def eval_last_steps(self, trajectory, steps):
        steps1 = int(math.floor(steps) / self.env.dt)
        if steps1 <= 0:
            return -1
        i = len(trajectory) - steps1
        new_trajectory = np.zeros(steps1)
        for s in range(steps1):
            new_trajectory[s] = trajectory[i + s]
        y = list(map(lambda number: number ** 2, new_trajectory))
        cost = np.trapz(y)

        cost = cost / steps1
        return cost

    # @background
    def _aux_expcost(self, i):
        self.reset()
        self.exploration_light()
        c,_,_ = self.env.eval_steps()
        print(" i = ", i)
        return c


    def log_result(self, result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
        self.xData.append(result)

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
            self.exploration_light()
            c,u_c,_ = self.env.eval_steps()

            exp_cost += c
        exp_cost =  exp_cost / ita
        self.env.max_T = max_T
        return exp_cost, T





    def run(self, output):
        T = self.env.max_T

        delta_est, d, K_listu, K_listl = self.exploration()
        cost, u_cost, _ = self.env.eval_steps()
        print("total cost env = ", cost, u_cost, self.env.max_T)
        t = self.env.t
        r = 0

        trajectory = list(chain.from_iterable(self.env.trajectory))

        filename_K_listu = 'K_listu.npy'
        filename_K_listl = 'K_listl.npy'
        filename_d = 'd.npy'
        filename_trajectory = 'trajectory.npy'
        filename_regrets = 'regrets.npy'
        filename_modelPredictions = 'modelPredictions.npy'

        np.save(filename_K_listl, K_listl)
        np.save(filename_K_listu, K_listu)
        np.save(filename_d, d)
        np.save(filename_trajectory, trajectory)



        if output:
            # print("t = ", self.env.t)
            # print("is done =", self.env.done)
            odelta = self.optimal_delta()
            d_regret = list(map(lambda x: math.fabs(x - odelta), d))
            print("starting Evaluation")
            cost, u_cost,_ = self.env.eval_steps()
            # r, regrets = self.env.regret()

            regrets = []
            opt_cost = self.optimal_cost()

            self.xData = []

            pool = multiprocessing.Pool()
            # for i in range(ita):
            #     pool.apply(self._aux_expcost, args = (i, ), callback = self.log_result)
            pool.map_async(self.expect_cost_sim, range(int(env.max_T)), callback = self.log_result)
            pool.close()
            pool.join()
            ## self.xData[0][0] = (0,0)
            print('results = ', self.xData)
            costs = [self._aux_mult2(a, b) for a,b in self.xData[0]]
            regrets = [self._aux_mult(a, b) for a,b in self.xData[0]]
            regrets[0] = 0.1
            regrets = np.log(regrets)

            print('regrets = ', regrets)
            print('costs = ', costs)
            np.save(filename_regrets, regrets)


            for t in progressbar(range(100)):
                opt_cost_t = opt_cost
                expect_cost = self.expect_cost_sim(100, t)
                r = expect_cost - opt_cost_t
                regrets.append(t*math.fabs(r))

            t1 = np.arange(0.1, len(regrets), self.env.dt)
            # t1 = np.exp(t1)
            t2 = np.arange(0.1, len(regrets), 1)
            # t3 = np.arange(0.1, len(d_regret), 1)

            # self.xData = t1
            # self.yData = regrets
            # geneticParameters = self.generate_Initial_Parameters()



            # fittedParameters, pcov = curve_fit(func, t2, regrets)
            # # fittedParameters2 ,_ = curve_fit(func2, t3, d_regret)
            # modelPredictions = func(t2, *fittedParameters)
            # # modelPredictions2 = func2(t2, *fittedParameters2)
            # print(fittedParameters)
            # np.save(filename_modelPredictions, modelPredictions)

            # K_line0 = [self.K] * int(t)
            # K_line1 = [-1.0 * self.K] * int(t)
            fig1, ax1 = plt.subplots(1, 1, figsize=(32, 16))
            fig2, ax2 = plt.subplots(1, 1, figsize=(32, 16))
            fig3, ax3 = plt.subplots(1, 1, figsize=(32, 16))
            ax1.plot(np.linspace(0., int(T), len(trajectory)), trajectory, lw=1, label="$X_t$")
            ax1.plot(np.linspace(0., int(T), len(K_listu)), K_listu, color='r', lw=3, label="$K_n + \hat{\delta}_n$")
            ax1.plot(np.linspace(0., int(T), len(K_listl)), K_listl, color='g', lw=3, label="$K_n + \hat{\delta}_n$")
            ax1.plot(np.linspace(0., int(T), len(d)), d, lw=6, label="$\hat{\delta}$", color='magenta')
            ax2.plot(np.linspace(0., int(T), len(d_regret)), d_regret, lw=2, label="$|\delta - \hat{\delta}|$")
            # ax2.plot(np.linspace(0., int(T), len(modelPredictions2)), modelPredictions2, lw=2, label="fitted")
            # ax3.plot(np.linspace(0., len(regrets), len(regrets)), (regrets), lw=2, label="regret")
            # ax3.plot(np.linspace(0., len(regrets), len(modelPredictions)), (modelPredictions), color ='magenta' , lw=6, label="fitted r")

            # ax3.plot(t1, np.log(t1), lw=2, color = 'r', label="log(t)")
            # ax3.set_ylim([0,10])
            ax1.legend()
            ax2.legend()
            ax3.legend()
            plt.show()
            
            # print("n = ", self.n)
            # print("delta = ", delta_est)

        # cost, u_cost, _ = self.env.eval_steps()
        # r, regrets = self.env.regret()
        # print("total cost env = ", cost, u_cost, self.env.max_T)
        # print("total cost adapt = ", tot_cost, u_cost )





        return cost, delta_est, t, r


if __name__ == "__main__":
    # theta0, theta1 = -0.54, 0.25
    # T = 200.0
    # dt = 0.0001
    theta0, theta1 = -6, 1
    T = 100.0
    dt = 0.001
    n = math.floor(math.sqrt(T))

    K = 6

    env = Simulation.Env(0, T, frameskip=1, dt=dt, start=(-1.0 * K), theta0=theta0, theta1=theta1)
    adaptiv = Adaptive_A(K=K, env=env, record=True)

    tot_cost, delta, t, r = adaptiv.run(True)


    print("optimal cost = ", adaptiv.optimal_cost())
    print("total cost = ", tot_cost)
    print("Regret = ", r)

    print("delta = ", delta)
    print("optimal delta = ", adaptiv.optimal_delta())



    # filename_K_listu = 'K_listu.npy'
    # filename_K_listl = 'K_listl.npy'
    # filename_d = 'd.npy'
    # filename_trajectory = 'trajectory.npy'
    # filename_regrets = 'regrets.npy'
    # filename_modelPredictions = 'modelPredictions.npy'
    #
    # regrets = np.load(filename_regrets)
    # K_listu = np.load(filename_K_listu)
    # K_listl = np.load(filename_K_listl)
    # d = np.load(filename_d)
    # trajectory = np.load(filename_trajectory)
    # modelPredictions = np.load(filename_modelPredictions)
    #
    # odelta = adaptiv.optimal_delta()
    # d_regret = list(map(lambda x: math.fabs(x - odelta), d))
    #
    #
    # t1 = np.arange(0.1, len(regrets), dt)
    #
    # fig1, ax1 = plt.subplots(1, 1, figsize=(32, 16))
    # fig2, ax2 = plt.subplots(1, 1, figsize=(32, 16))
    # fig3, ax3 = plt.subplots(1, 1, figsize=(32, 16))
    # ax1.plot(np.linspace(0., int(T), len(trajectory)), trajectory, lw=2, label="X_t")
    # ax1.plot(np.linspace(0., int(T), len(K_listu)), K_listu, color='r', lw=3, label="K_n + delta")
    # ax1.plot(np.linspace(0., int(T), len(K_listl)), K_listl, color='g', lw=3, label="-K_n + delta")
    # ax1.plot(np.linspace(0., int(T), len(d)), d, lw=2, label="delta", color='magenta')
    # ax2.plot(np.linspace(0., int(T), len(d_regret)), d_regret, lw=2, label="|d-d_est|")
    # # ax2.plot(np.linspace(0., int(T), len(modelPredictions2)), modelPredictions2, lw=2, label="fitted")
    # ax3.plot(np.linspace(0., len(regrets), len(regrets)), regrets, lw=2, label="regret")
    # ax3.plot(np.linspace(0., len(regrets), len(modelPredictions)), modelPredictions, color ='magenta' , lw=6, label="fitted r")
    #
    # # ax3.plot(t1, np.log(t1), lw=2, color = 'r', label="log(t)")
    # # ax3.set_ylim([0,10])
    # ax1.legend()
    # ax2.legend()
    # ax3.legend()
    # plt.show()


