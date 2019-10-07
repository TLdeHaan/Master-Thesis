import numpy as np
from math import *
import itertools
import scipy.sparse as sp
import datetime
import pandas as pd

# class MDP:
#     pass


def state_space(B=3, N=3, Patients=4, K=3):
    # Define number of states:
    S_num_small = ((B + 1) ** N) * Patients     # Excludes intervals
    S_num = S_num_small * K


    # Define possible states, but excluding intervals:
    k = [list(range(K))]
    b = [list(range(B + 1))] * N
    p = [list(range(Patients))]
    S_small = list(itertools.product(*(b + p)))

    # For visual representation of all states including intervals:
    row_names = ["s" + str(i) for i in range(1, S_num + 1)]
    col_names = ["k"] + ["Day " + str(i) for i in range(1, N + 1)] + ["type"]
    all_combos = list(itertools.product(*(k + b + p)))
    S_all = pd.DataFrame(all_combos, index=[_ for _ in row_names], columns=[_ for _ in col_names])

    return S_num_small, S_num, S_small, S_all, all_combos


def action_space(N=3):
    # Define number of actions:
    A_num = N + 1

    # For visual representation of all actions:
    A_all = ["a" + str(i) for i in range(1, N + 1)] + ["Reject"]

    return A_num, A_all


def cost_matr(S_num_small, A_num, S_small, B=3, Patients=4, K=3):
    C_small = np.empty((S_num_small, A_num))

    for i in range(0, S_num_small, Patients):
        C_small[i] = np.array([(list([999999]) * (A_num - 1) + [0])])
        C_small[i + 1] = np.array([[0] * (A_num - 1) + [100]])
        C_small[i + 2] = np.array([list(range(0, (A_num - 1) * 4, 4)) + [1000]])
        if A_num > 8:
            for j in range(7, A_num - 1):
                C_small[i + 2][j] = C_small[i + 2][j] * (j - 6) * 4
        if Patients == 4:
            C_small[i + 3] = np.array([list(range(A_num - 1)) + [1000]])
            for j in range(28, A_num - 1):
                C_small[i + 3][j] = C_small[i + 3][j] + (j - 27)
    for j, s in enumerate(S_small):
        x = np.array(s)
        y = x.copy()
        if j % Patients == 1 or j % Patients == 2:
            for a in range(A_num - 1):
                if y[a] >= B:
                    C_small[j][a] = 999999
        elif j % Patients == 3:
            for a in range(A_num - 1):
                x = np.array(s)
                y = x.copy()
                if y[a] >= B - 1:
                    C_small[j][a] = 999999

    for i in range(K):
        if i == 0:
            C_big = C_small
        else:
            C_big = np.vstack((C_big, C_small))
    return C_big


def trans_small(S_small, A_num, p_arr=[0.6, 0.2, 0.15, 0.05], B=3, Patients=4):
    trans_small_k = [None] * A_num  # To store transitions if k < k
    trans_small_K = [None] * A_num  # To store transitions if k = K

    '''Define transition dictionary for k < K'''
    # Assigning patients
    for a in range(A_num - 1):
        tr_assign = {}
        for ir, s_sub in enumerate(S_small[::Patients]):
            x = np.array(s_sub)
            y = x.copy()
            if y[a] < B:
                y[a] += 1
                for ic, s_check in enumerate(S_small[::Patients]):
                    if np.array_equal(s_check, y):  # Search result of assigning follow-up and urgent patients
                        for i in range(Patients):
                            # tr_assign[(ir * Patients, ic * Patients + i)] = p_arr[i]
                            tr_assign[(ir * Patients + 1, ic * Patients + i)] = p_arr[i]
                            tr_assign[(ir * Patients + 2, ic * Patients + i)] = p_arr[i]
                        break
                if Patients == 4 and y[a] < B:
                    y[a] += 1
                    for ic, s_check in enumerate(S_small[::Patients]):
                        if np.array_equal(s_check, y):  # Search result of assigning new patients
                            for i in range(Patients):
                                tr_assign[(ir * Patients + 3, ic * Patients + i)] = p_arr[i]
                            break
        trans_small_k[a] = tr_assign

    # Rejecting patients
    tr_reject = {}
    for ir, s_sub in enumerate(S_small[::Patients]):
        for i in range(Patients):
            for j in range(Patients):
                tr_reject[(ir * Patients + i, ir * Patients + j)] = p_arr[j]
    trans_small_k[-1] = tr_reject

    '''Define transition dictionary for k = K'''
    # Assigning patients
    for a in range(A_num - 1):
        tr_assign = {}
        for ir, s_sub in enumerate(S_small[::Patients]):
            x = np.array(s_sub)
            y = x.copy()
            if y[a] < B:
                y[a] += 1
                for i in range(A_num - 1):  # Move all values 1 day closer
                    y[i] = y[i + 1]
                for ic, s_check in enumerate(S_small[::Patients]):
                    if np.array_equal(s_check, y):  # Search result of assigning follow-up and urgent patients
                        for i in range(Patients):
                            # tr_assign[(ir * Patients, ic * Patients + i)] = p_arr[i]
                            tr_assign[(ir * Patients + 1, ic * Patients + i)] = p_arr[i]
                            tr_assign[(ir * Patients + 2, ic * Patients + i)] = p_arr[i]
                        break
            y = x.copy()
            if y[a] < B - 1 and Patients == 4:
                y[a] += 2
                for i in range(A_num - 1):  # Move all values 1 day closer
                    y[i] = y[i + 1]
                for ic, s_check in enumerate(S_small[::Patients]):
                    if np.array_equal(s_check, y):  # Search result of assigning new patients
                        for i in range(Patients):
                            tr_assign[(ir * Patients + 3, ic * Patients + i)] = p_arr[i]
                        break
        trans_small_K[a] = tr_assign

    # Rejecting patients
    tr_reject = {}
    for ir, s_sub in enumerate(S_small[::Patients]):
        x = np.array(s_sub)
        y = x.copy()
        for i in range(A_num - 1):  # Move all values 1 day closer
            y[i] = y[i + 1]
        for ic, s_check in enumerate(S_small[::Patients]):
            if np.array_equal(s_check, y):
                for i in range(Patients):
                    for j in range(Patients):
                        tr_reject[(ir * Patients + j, ic * Patients + i)] = p_arr[i]
    trans_small_K[-1] = tr_reject

    return trans_small_k, trans_small_K


def trans_big(S_num_small, S_num, A_num, trans_small_k, trans_small_K, K=3):
    trans_big = [None] * A_num

    # Assigning patients
    for a in range(A_num - 1):
        # print(len(trans_small_k[a]))
        tr_a = {}
        for k in range(K - 1):
            tr_k = trans_small_k[a].copy()
            tr_K = trans_small_K[a].copy()
            for t in trans_small_k[a]:
                tr_k[t[0] + S_num_small * k, t[1] + S_num_small * (k + 1)] = tr_k.pop(t)
            tr_a.update(tr_k)
        for t in trans_small_K[a]:
            tr_K[t[0] + S_num_small * (K - 1), t[1]] = tr_K.pop(t)
        tr_a = {**tr_a, **tr_K}
        trans_big[a] = tr_a

    # Rejecting patients
    tr_r = {}
    for k in range(K - 1):
        tr_k = trans_small_k[-1].copy()
        tr_K = trans_small_K[-1].copy()
        for t in trans_small_k[-1]:
            tr_k[t[0] + S_num_small * k, t[1] + S_num_small * (k + 1)] = tr_k.pop(t)
        tr_r.update(tr_k)
    for t in trans_small_K[-1]:
        tr_K[t[0] + S_num_small * (K - 1), t[1]] = tr_K.pop(t)
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


def value_iteration(S_num, A_num, A_all, C, P, eps=0.001, max_it=100, printFull=False):
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
    else:
        print("Span = " + str(span))
        print("The optimal gain = ", str((max(v[n, :] - v[n - 1, :]) + min(v[n, :] - v[n - 1, :])) / 2))
    if printFull == 1:
        v_full = np.hstack((v, m))
        col_names = ["s" + str(i) for i in range(1, S_num + 1)] + ["Max", "Min", "Span"]
        result = pd.DataFrame(v_full, index=[_ for _ in range(0, n + 1)], columns=col_names)
        print(result)
        print("The stationary policy for the states is: " + str(d))


    end = datetime.datetime.now()
    print("Value iteration completed at: " + str(end))
    duration = end - begin
    print("Length of Value iteration: " + str(duration))



def setup_1():
    # Arrival rates, number of types (incl no patient), number of intervals, slots per day,
    # and probabilities per interval
    labda_fup = 4.2
    labda_urg = 0.55
    labda_new = 0.25
    labda_sum = labda_fup + labda_urg + labda_new
    pat_types = 1 + sum(i > 0 for i in [labda_fup, labda_urg, labda_new])
    K = 1
    while (1 + labda_sum / K) * (exp(-labda_sum / K)) < 0.95:
        K += 1
    B = ceil(labda_sum) + 1  # Define number of slots per day [labda]^+ + 1

    p_no = (((labda_sum / K) ** 0) / factorial(0)) * exp(-labda_sum / K)
    p_arr = 1 - p_no
    p_fup = (p_arr / labda_sum) * labda_fup
    p_urg = (p_arr / labda_sum) * labda_urg
    p_new = (p_arr / labda_sum) * labda_new
    p_all = [p_no, p_fup, p_urg, p_new]

    # Length of schedule, limit on iterations in VIA, convergence measure
    N = 3  # Number of days available in scheduling horizon
    max_iterations = 1000
    epsilon = 0.001
    printFull = True

    S_num_small, S_num, S_small, S_all, all_combos = state_space(B, N, pat_types, K)
    A_num, A_all = action_space(N)
    C = cost_matr(S_num_small, A_num, S_small, B, pat_types, K)
    trans_small_k, trans_small_K = trans_small(S_small, A_num, p_all, B, pat_types)
    P = trans_big(S_num_small, S_num, A_num, trans_small_k, trans_small_K, K)

    # Periodicity fix:
    gamma = 0.75
    C = gamma * C
    I_p = sp.dok_matrix((S_num, S_num))
    I_p.setdiag(1)
    for a in range(A_num):
        P[a] = gamma * P[a] + (1 - gamma) * I_p

    value_iteration(S_num, A_num, A_all, C, P, epsilon, max_iterations, printFull)

setup_1()

# Welke inputs: Arrival rates (3x), Number of days in future (1x)

# Results with normal arrival rates (8.4, 1.1, 0.5)
# Span = 0.0009925203720966731
# The optimal gain =  0.0005168401824335641

# Results with arrival rates (4.2, 0.55, 0.25)
# N = 3, gamma= 0.75, max_it = 1000
# Starting value iteration: 2019-09-22 19:52:09.033776
# Span = 0.000989382822200291
# The optimal gain =  0.005079121969933542
# Value iteration completed at: 2019-09-22 19:55:50.081364
# Length of Value iteration: 0:03:41.047588 <- Did do full print!