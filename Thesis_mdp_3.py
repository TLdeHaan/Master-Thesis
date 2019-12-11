import numpy as np
import random
import matplotlib.pyplot as plt
from math import factorial, exp
import pandas as pd

import itertools
import scipy.sparse as sp

import datetime

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


class MDP:
    planning_horizon = 20
    days_week = 5
    warm_up = 260
    total_period = 12740
    prob_arr = 0.05
    lim_u = 6
    lim_n = 20

    def __init__(self, fup, urg, new, horizon):
        self.labda_fup = fup
        self.labda_urg = urg
        self.labda_new = new
        self.labda_sum = fup + urg + new
        self.N = horizon
        self.K = 1
        while (1 + self.labda_sum / self.K) * (exp(-self.labda_sum / self.K)) < 1 - self.prob_arr:
            self.K += 1
        self.B = int(np.ceil(self.labda_sum + 0.01))
        self.Patients = 1
        if self.labda_fup > 0:
            self.Patients += 1
        if self.labda_urg > 0:
            self.Patients += 1
        if self.labda_new > 0:
            self.Patients += 1

        self.p_no = (((self.labda_sum / self.K) ** 0) / factorial(0)) * exp(-self.labda_sum / self.K)
        self.p_arr = 1 - self.p_no
        self.p_fup = (self.p_arr / self.labda_sum) * self.labda_fup
        self.p_urg = (self.p_arr / self.labda_sum) * self.labda_urg
        self.p_new = (self.p_arr / self.labda_sum) * self.labda_new
        self.p_all = [self.p_no, self.p_fup, self.p_urg, self.p_new]

    def state_space(self):
        # Define number of states:
        S_num_small = ((self.B + 1) ** self.N) * self.Patients     # Excludes intervals
        S_num = S_num_small * self.K

        # Define possible states, but excluding intervals:
        k = [list(range(self.K))]
        b = [list(range(self.B + 1))] * self.N
        p = [list(range(self.Patients))]
        S_small = list(itertools.product(*(b + p)))

        # For visual representation of all states including intervals:
        row_names = ["s" + str(i) for i in range(1, S_num + 1)]
        col_names = ["k"] + ["Day " + str(i) for i in range(1, self.N + 1)] + ["type"]
        all_combos = list(itertools.product(*(k + b + p)))
        S_all = pd.DataFrame(all_combos, index=[_ for _ in row_names], columns=[_ for _ in col_names])

        return S_num_small, S_num, S_small, S_all, all_combos

    def action_space(self):
        # Define number of actions:
        A_num = self.N + 1

        # For visual representation of all actions:
        A_all = ["a" + str(i) for i in range(1, self.N + 1)] + ["Reject"]

        return A_num, A_all

    def cost_matr(self):
        A_num, _ = self.action_space()
        S_num_small, _, S_small, _, _ = self.state_space()
        C_small = np.empty((S_num_small, A_num))

        for i in range(0, S_num_small, self.Patients):
            C_small[i] = np.array([(list([999999]) * (A_num - 1) + [0])])
            C_small[i + 1] = np.array([[0] * (A_num - 1) + [90]])
            C_small[i + 2] = np.array([list(range(0, (A_num - 1) * 4, 4)) + [90]])
            print(S_num_small)
            print(A_num)
            print(C_small.shape)
            if A_num > self.lim_u:
                for j in range(self.lim_u + 1, A_num - 1):
                    C_small[i + 2][j] = C_small[i + 2][j] * (j - self.lim_u) * 4
            if self.Patients == 4:
                C_small[i + 3] = np.array([list(range(A_num - 1)) + [90]])
                for j in range(20, A_num - 1):
                    C_small[i + 3][j] = C_small[i + 3][j] + (j - 19)
        for j, s in enumerate(S_small):
            x = np.array(s)
            y = x.copy()
            if j % self.Patients == 1 or j % self.Patients == 2:
                for a in range(A_num - 1):
                    if y[a] >= self.B:
                        C_small[j][a] = 999999
            elif j % self.Patients == 3:
                for a in range(A_num - 1):
                    x = np.array(s)
                    y = x.copy()
                    if y[a] >= self.B - 1:
                        C_small[j][a] = 999999

        for i in range(self.K):
            if i == 0:
                C_big = C_small
            else:
                C_big = np.vstack((C_big, C_small))

        return C_big

    def trans_small(self):
        A_num, _ = self.action_space()
        _, _, S_small, _, _ = self.state_space()
        trans_small_k = [None] * A_num  # To store transitions if k < k
        trans_small_K = [None] * A_num  # To store transitions if k = K

        '''Define transition dictionary for k < K'''
        # Assigning patients
        for a in range(A_num - 1):
            tr_assign = {}
            for ir, s_sub in enumerate(S_small[::self.Patients]):
                x = np.array(s_sub)
                y = x.copy()
                if y[a] < self.B:
                    y[a] += 1
                    for ic, s_check in enumerate(S_small[::self.Patients]):
                        if np.array_equal(s_check, y):  # Search result of assigning follow-up and urgent patients
                            for i in range(self.Patients):
                                # tr_assign[(ir * Patients, ic * Patients + i)] = p_arr[i]
                                tr_assign[(ir * self.Patients + 1, ic * self.Patients + i)] = self.p_all[i]
                                tr_assign[(ir * self.Patients + 2, ic * self.Patients + i)] = self.p_all[i]
                            break
                    if self.Patients == 4 and y[a] < self.B:
                        y[a] += 1
                        for ic, s_check in enumerate(S_small[::self.Patients]):
                            if np.array_equal(s_check, y):  # Search result of assigning new patients
                                for i in range(self.Patients):
                                    tr_assign[(ir * self.Patients + 3, ic * self.Patients + i)] = self.p_all[i]
                                break
            trans_small_k[a] = tr_assign

        # Rejecting patients
        tr_reject = {}
        for ir, s_sub in enumerate(S_small[::self.Patients]):
            for i in range(self.Patients):
                for j in range(self.Patients):
                    tr_reject[(ir * self.Patients + i, ir * self.Patients + j)] = self.p_all[j]
        trans_small_k[-1] = tr_reject

        '''Define transition dictionary for k = K'''
        # Assigning patients
        for a in range(A_num - 1):
            tr_assign = {}
            for ir, s_sub in enumerate(S_small[::self.Patients]):
                x = np.array(s_sub)
                y = x.copy()
                if y[a] < self.B:
                    y[a] += 1
                    for i in range(A_num - 1):  # Move all values 1 day closer
                        y[i] = y[i + 1]
                    for ic, s_check in enumerate(S_small[::self.Patients]):
                        if np.array_equal(s_check, y):  # Search result of assigning follow-up and urgent patients
                            for i in range(self.Patients):
                                # tr_assign[(ir * Patients, ic * Patients + i)] = p_arr[i]
                                tr_assign[(ir * self.Patients + 1, ic * self.Patients + i)] = self.p_all[i]
                                tr_assign[(ir * self.Patients + 2, ic * self.Patients + i)] = self.p_all[i]
                            break
                y = x.copy()
                if y[a] < self.B - 1 and self.Patients == 4:
                    y[a] += 2
                    for i in range(A_num - 1):  # Move all values 1 day closer
                        y[i] = y[i + 1]
                    for ic, s_check in enumerate(S_small[::self.Patients]):
                        if np.array_equal(s_check, y):  # Search result of assigning new patients
                            for i in range(self.Patients):
                                tr_assign[(ir * self.Patients + 3, ic * self.Patients + i)] = self.p_all[i]
                            break
            trans_small_K[a] = tr_assign

        # Rejecting patients
        tr_reject = {}
        for ir, s_sub in enumerate(S_small[::self.Patients]):
            x = np.array(s_sub)
            y = x.copy()
            for i in range(A_num - 1):  # Move all values 1 day closer
                y[i] = y[i + 1]
            for ic, s_check in enumerate(S_small[::self.Patients]):
                if np.array_equal(s_check, y):
                    for i in range(self.Patients):
                        for j in range(self.Patients):
                            tr_reject[(ir * self.Patients + j, ic * self.Patients + i)] = self.p_all[i]
        trans_small_K[-1] = tr_reject

        return trans_small_k, trans_small_K

    def trans_big(self):
        A_num, _ = self.action_space()
        S_num_small, S_num, _, _, _ = self.state_space()
        trans_small_k, trans_small_K = self.trans_small()
        trans_big = [None] * A_num

        # Assigning patients
        for a in range(A_num - 1):
            # print(len(trans_small_k[a]))
            tr_a = {}
            for k in range(self.K - 1):
                tr_k = trans_small_k[a].copy()
                tr_K = trans_small_K[a].copy()
                for t in trans_small_k[a]:
                    tr_k[t[0] + S_num_small * k, t[1] + S_num_small * (k + 1)] = tr_k.pop(t)
                tr_a.update(tr_k)
            for t in trans_small_K[a]:
                tr_K[t[0] + S_num_small * (self.K - 1), t[1]] = tr_K.pop(t)
            tr_a = {**tr_a, **tr_K}
            trans_big[a] = tr_a

        # Rejecting patients
        tr_r = {}
        for k in range(self.K - 1):
            tr_k = trans_small_k[-1].copy()
            tr_K = trans_small_K[-1].copy()
            for t in trans_small_k[-1]:
                tr_k[t[0] + S_num_small * k, t[1] + S_num_small * (k + 1)] = tr_k.pop(t)
            tr_r.update(tr_k)
        for t in trans_small_K[-1]:
            tr_K[t[0] + S_num_small * (self.K - 1), t[1]] = tr_K.pop(t)
        tr_r.update(tr_K)
        trans_big[-1] = tr_r

        '''From dictionary to sparse matrix'''
        P = [None] * A_num
        for a in range(A_num):
            trans_a = sp.dok_matrix((S_num, S_num))
            for i in trans_big[a]:
                trans_a[i] = trans_big[a].get(i)
            trans_a.tocsr()
            P[a] = trans_a

        return P

    def value_iteration(self, printFull=False, gamma=0.75, eps=0.001, max_it=1000):
        A_num, A_all = self.action_space()
        _, S_num, S_small, _, _ = self.state_space()
        C = self.cost_matr() * gamma
        P = self.trans_big()
        I_p = sp.dok_matrix((S_num, S_num))
        I_p.setdiag(1)
        for a in range(A_num):
            P[a] = gamma * P[a] + (1 - gamma) * I_p

        begin = datetime.datetime.now()
        print("Starting value iteration: " + str(begin))
        n = 0
        span = 2 * eps
        v = np.zeros((1, S_num))
        m = np.ndarray((1, 3), dtype=float, buffer=np.repeat(np.nan, 3))
        while span > eps and n < max_it:
            n = n + 1
            v_all = np.ndarray((S_num, A_num), dtype=float, buffer=np.array(np.repeat(np.nan, S_num * A_num)))
            for a in range(A_num):
                v_all[:, a] = P[a].dot(v[n - 1, :]) + C[:, a]
            x = np.array([])
            for i in range(S_num):
                x = np.append(x, min(v_all[i, :]))
            v = np.vstack((v, x))
            span = max(v[n, :] - v[n - 1, :]) - min(v[n, :] - v[n - 1, :])
            if printFull == 1:
                m = np.vstack((m, (max(v[n, :] - v[n - 1, :]), min(v[n, :] - v[n - 1, :]), span)))
        d = np.array([])
        for i in range(S_num):
            d = np.append(d, A_all[v_all[i, :].argmin()])
        if n == max_it:
            print("Maximum number of iterations exceeded. Value Iteration will be terminated."
                  " Increase the number of iterations or check for periodicity!")
        # else:
            # print("Span = " + str(span))
            # print("The optimal gain = ", str((max(v[n, :] - v[n - 1, :]) + min(v[n, :] - v[n - 1, :])) / 2))
        if printFull:
            v_full = np.hstack((v, m))
            col_names = ["s" + str(i) for i in range(1, S_num + 1)] + ["Max", "Min", "Span"]
            result = pd.DataFrame(v_full, index=[_ for _ in range(0, n + 1)], columns=col_names)
            print(result)
            print("The stationary policy for the states is: " + str(d))

        end = datetime.datetime.now()
        print("Value iteration completed at: " + str(end))
        duration = end - begin
        print("Duration of Value iteration: " + str(duration))

        return d

    def simulation(self, policy, num_states, s_list, prob_list, seed):
        np.random.seed(seed)
        random.seed(seed)
        s_num = num_states
        d = policy
        at_array = np.ndarray((self.Patients - 1, self.N + 1), dtype=int)
        for j in range(self.Patients - 1):
            at_array[j] = np.repeat(0, self.N + 1)
        occ_rate = np.repeat(0., self.total_period)

        #Start of simulation
        interval = 1
        start_s = random.choices(list(range(self.Patients)), weights=[self.p_no, self.p_fup, self.p_urg, self.p_new])[0]
        start_a = d[start_s]
        if start_a == "Reject":
            pos_s = s_list[int(self.N * s_num) + start_s]
            trans_prop = prob_list[int(self.N * s_num) + start_s]
            next_s = random.choices(pos_s, weights=trans_prop)[0]
        else:
            pos_s = s_list[(int(start_a[1]) - 1) * s_num + start_s]
            trans_prop = prob_list[(int(start_a[1]) - 1) * s_num + start_s]
            next_s = random.choices(pos_s, weights=trans_prop)[0]
        interval += 1
        while interval <= self.warm_up * self.K:
            next_a = d[next_s]
            if next_a == "Reject":
                pos_s = s_list[self.N * s_num + next_s]
                trans_prop = prob_list[self.N * s_num + next_s]
                next_s = random.choices(pos_s, weights=trans_prop)[0]
            else:
                pos_s = s_list[(int(next_a[1]) - 1) * s_num + next_s]
                trans_prop = prob_list[(int(next_a[1]) - 1) * s_num + next_s]
                next_s = random.choices(pos_s, weights=trans_prop)[0]
            interval += 1

        interval = 1
        while interval <= self.total_period * self.K:
            next_a = d[next_s]
            if s_num - (s_num / self.K) <= next_s <= s_num:
                block_bin = s_num / (self.K * self.B)
                b = 0
                while s_num - (s_num / self.K) <= next_s - block_bin * b:
                    b += 1
                if next_a == "a1":
                    b += 2 if next_s % self.Patients == 3 else 1
                occ_rate[int(interval / self.K) - 1] = b / self.B
            if next_a == "Reject":
                if next_s % self.Patients != 0:
                    at_array[next_s % self.Patients - 1, -1] += 1
                pos_s = s_list[self.N * s_num + next_s]
                trans_prop = prob_list[self.N * s_num + next_s]
                next_s = random.choices(pos_s, weights=trans_prop)[0]
            else:
                if next_s % self.Patients != 0:
                    at_array[next_s % self.Patients - 1, int(next_a[-1]) - 1] += 1
                pos_s = s_list[(int(next_a[1]) - 1) * s_num + next_s]
                trans_prop = prob_list[(int(next_a[1]) - 1) * s_num + next_s]
                next_s = random.choices(pos_s, weights=trans_prop)[0]
            interval += 1

        return at_array, occ_rate

    def statistics_at(self, at_total, runs):
        at_stats_runs = np.zeros((runs * 3, 5))  # To record AT (pop&mean&var), Rejection rate
        for i in range(runs * 3):
            # Total number of patients per type and run
            at_stats_runs[i, 0] = sum(at_total[i, :])

            # Number of patients per type each run but NOT rejected
            at_stats_runs[i, 1] = sum(at_total[i, :-1])

            # Mean AT
            at_stats_runs[i, 2] = np.average(range(1, self.N + 1), weights=at_total[i, :-1])
            # Var AT
            var = 0
            for j, weight in enumerate(at_total[i, :-1]):
                var += (weight * ((j + 1) - at_stats_runs[i, 2]) ** 2) / (at_stats_runs[i, 1] - 1)
            at_stats_runs[i, 3] = var

            # Rejection rate
            at_stats_runs[i, 4] = at_total[i, -1] / at_stats_runs[i, 0]

        at_stats = np.zeros((3, 3))  # To summarize AT (pop&mean&var), Rejection rate
        for k in range(3):
            at_stats[k, 0] = np.average(at_stats_runs[k::3, 2], weights=at_stats_runs[k::3, 1])
            at_stats[k, 1] = np.average(at_stats_runs[k::3, 3], weights=(at_stats_runs[k::3, 1] /
                                                                         sum(at_stats_runs[k::3, 1])) ** 2)
            at_stats[k, 2] = np.average(at_stats_runs[k::3, 4], weights=at_stats_runs[k::3, 0])
        return at_stats_runs, at_stats

    def statistics_occ(self, occ_total, runs):
        occ_stats_runs = np.zeros((runs, 2))
        for i in range(runs):
            occ_stats_runs[i, 0] = np.average(occ_total[i, :])
            occ_stats_runs[i, 1] = np.var(occ_total[i, :])

        occ_stats = np.array([np.average(occ_stats_runs[:, 0]), np.average(occ_stats_runs[:, 1]) / runs])

        return occ_stats_runs, occ_stats

    def bar(self, pol, at_runs, at, occ):
        at_sum = np.array([sum(at_runs[0::3, :]), sum(at_runs[1::3, :]), sum(at_runs[2::3, :])])
        at_proportions = np.zeros((3, self.N + 2))
        for i in range(3):
            at_proportions[i, 1:] = at_sum[i, :] / sum(at_sum[i, :])
        at_proportions_inverse = 1 - at_proportions
        for j in range(3):
            x = list(at_proportions[j, :])
            y = list(at_proportions_inverse[j, :])
            plt.bar(list(range(self.N + 2)), y, color='white', bottom=x)
            plt.bar(list(range(self.N + 2)), x)
            plt.xticks(range(0, self.N + 2), list(range(0, self.N + 1)) + ["R"])
            plt.xlabel("Days of admission time")
            plt.ylabel("Proportion of arrived patients")
            if j == 0:
                plt.title(pol + ": Follow-up patients"
                          + "\n AT (days): \u03BC = " + str(np.round(at[j, 0], 3))
                          + ", \u03C3\u00b2 = " + str(np.round(at[j, 1], 3))
                          + "\n Occupancy rate: \u03BC = " + str(np.round(occ[0], 3)) +
                          ", \u03C3\u00b2 = " + str(np.round(occ[1], 3)))
            if j == 1:
                plt.title(pol + ": Urgent patients"
                          + "\n AT (days): \u03BC = " + str(np.round(at[j, 0], 3))
                          + ", \u03C3\u00b2 = " + str(np.round(at[j, 1], 3))
                          + "\n Occupancy rate: \u03BC = " + str(np.round(occ[0], 3)) +
                          ", \u03C3\u00b2 = " + str(np.round(occ[1], 3)))
            if j == 2:
                plt.title(pol + ": New patients"
                          + "\n AT (days): \u03BC = " + str(np.round(at[j, 0], 3))
                          + ", \u03C3\u00b2 = " + str(np.round(at[j, 1], 3))
                          + "\n Occupancy rate: \u03BC = " + str(np.round(occ[0], 3)) +
                          ", \u03C3\u00b2 = " + str(np.round(occ[1], 3)))
            plt.show()

def experiment(setup, runs, summary_print=False, runs_print=False, latex_print=False):
    begin_exp = datetime.datetime.now()
    print('Experiments for setup started at: ' + str(begin_exp))
    # Stream of seeds:
    x = [1231, 1322, 2133, 2314, 3125, 3216, 1237, 1328, 2139, 23140]

    print('Creating policy.')
    d = setup.value_iteration()
    print('Policy complete.')

    print('Preparing policy for simulation.')
    s_num = ((setup.B + 1) ** setup.N) * setup.Patients * setup.K
    a_num = setup.N + 1
    P = setup.trans_big()
    s_list = [None] * s_num * a_num
    prob_list = [None] * s_num * a_num
    for m, a in enumerate(P):
        print("Restructuring transition matrix for action " + str(m + 1) + ".")
        for j in range(s_num):
            trans = a[j, ]
            pos_s = list(range(len(trans)))
            trans_prop = list(range(len(trans)))
            for k, l in enumerate(trans.keys()):
                pos_s[k] = l[1]
                trans_prop[k] = trans[0, pos_s[k]]
            s_list[(m * s_num) + j] = pos_s
            prob_list[(m * s_num) + j] = trans_prop
            if j % 2500 == 0:
                print("The first " + str(j) + " states are transformed.")

    # Arrays to record statistics
    at_total = np.zeros((runs * 3, setup.N + 1), dtype=int)
    occ_total = np.zeros((runs, MDP.total_period))
    print('Starting simulation process.')
    for i in range(runs):
        at_total[i * 3: i * 3 + 3, :], occ_total[i, :] = setup.simulation(d, s_num, s_list, prob_list, x[i])
    at_stats_runs, at_stats = setup.statistics_at(at_total, runs)
    occ_stats_runs, occ_stats = setup.statistics_occ(occ_total, runs)

    end_exp = datetime.datetime.now()
    print('Experiments for setup completed at: ' + str(end_exp))
    duration_exp = end_exp - begin_exp
    print('Experiments for setup solving time: ' + str(duration_exp))

    ind = ["Follow-up", "Urgent", "New"]
    col = ["\u03BC AT", "\u03C3\u00b2 AT", "Rej rate"]
    print("Results of the policy: MDP")
    if summary_print:
        at_df = pd.DataFrame(at_stats, index=ind, columns=col).round(3)
        occ_df = pd.DataFrame(occ_stats.reshape(1, 2), index=["Occupancy rate"],
                                     columns=["\u03BC", "\u03C3\u00b2"]).round(3)
        if not latex_print:
            print(at_df)
            print(occ_df)
        else:
            print(at_df.to_latex())
            print(occ_df.to_latex())
    if runs_print:
        ind = ind * runs
        at_df = pd.DataFrame(at_stats_runs[:, 2:], index=ind, columns=col).round(3)
        occ_df = pd.DataFrame(occ_stats_runs, index=list(range(1, runs + 1)),
                                     columns=["\u03BC", "\u03C3\u00b2"]).round(3)
        if not latex_print:
            print(at_df)
            print(occ_df)
        else:
            print(at_df.to_latex())
            print(occ_df.to_latex())

    return at_total, at_stats, occ_stats


def setups(horizon, runs, summary_print=False, runs_print=False, latex_print=False, bar_print=False):

    # setup = MDP(3.26, 0.34, 0.15, horizon)
    # at_total_1, at_stats_1, occ_stats_1 = experiment(setup, runs, summary_print, runs_print, latex_print)

    # setup = MDP(2.93, 0.37, 0.15, horizon)
    # at_total_2, at_stats_2, occ_stats_2 = experiment(setup, runs, summary_print, runs_print, latex_print)

    # setup = MDP(2.61, 0.41, 0.15, horizon)
    # at_total_3, at_stats_3, occ_stats_3 = experiment(setup, runs, summary_print, runs_print, latex_print)

    if bar_print:
        # setup.bar("MDP", at_total_1, at_stats_1, occ_stats_1)
        # setup.bar("MDP", at_total_2, at_stats_2, occ_stats_2)
        # setup.bar("MDP", at_total_3, at_stats_3, occ_stats_3)
        pass

setups(5, 10, True, True, True, True)
