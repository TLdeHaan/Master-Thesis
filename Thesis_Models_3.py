import numpy as np
import random
import matplotlib.pyplot as plt
from math import factorial, exp
import pandas as pd
np.set_printoptions(precision=4, linewidth=np.infty)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

class SimModel:

    days_week = 5
    planning_horizon = 4 * days_week
    warm_up = 260
    total_period = 12740
    prob_arr = 0.25

    def __init__(self, fup, urg, new):
        self.labda_fup = fup
        self.labda_urg = urg
        self.labda_new = new
        self.labda_sum = fup + urg + new
        self.phi = self.labda_sum + self.labda_new

    def arrivals(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        patient_types = [1, 2, 3]
        daily_arr = np.random.poisson(self.labda_sum)
        order_of_arr = random.choices(patient_types, weights=[self.labda_fup, self.labda_urg, self.labda_new],
                                      k=daily_arr)
        return order_of_arr

    def cap_no_res(self, phi):
        monthly_blocks = 0
        x = (((phi * self.planning_horizon) ** monthly_blocks) / factorial(monthly_blocks)) * \
            exp(-(phi * self.planning_horizon))
        while 1 - x > self.prob_arr:
            monthly_blocks += 1
            x += (((phi * self.planning_horizon) ** monthly_blocks) / factorial(monthly_blocks)) * \
                exp(-(phi * self.planning_horizon))
        weekly_blocks = int(np.ceil(monthly_blocks / (self.planning_horizon / self.days_week)))
        daily_blocks_low = 0
        # print(monthly_blocks, weekly_blocks)
        while weekly_blocks - daily_blocks_low * self.days_week >= self.days_week:
            daily_blocks_low += 1
        if weekly_blocks % self.days_week != 0:
            daily_blocks_high = daily_blocks_low + 1
        else:
            daily_blocks_high = daily_blocks_low
        days_low = self.days_week - (weekly_blocks - daily_blocks_low * self.days_week)
        #print(days_low)
        schedule = np.repeat(daily_blocks_low, self.planning_horizon)
        for i in range(1, self.planning_horizon + 1):
            if weekly_blocks % self.days_week == 0:
                if i % self.days_week > days_low:
                    schedule[i - 1] += 1
            else:
                if i % self.days_week > days_low or i % self.days_week == 0:
                    schedule[i - 1] += 1

        return daily_blocks_low, daily_blocks_high, days_low, schedule

    def cap_cons_res(self):
        daily_blocks_low, daily_blocks_high, days_low, schedule = self.cap_no_res(self.phi)
        monthly_reserved_blocks = 0
        x = (((self.labda_urg * self.planning_horizon) ** monthly_reserved_blocks) /
             factorial(monthly_reserved_blocks)) * exp(-self.labda_urg * self.planning_horizon)
        while 1 - x > self.prob_arr:
            monthly_reserved_blocks += 1
            x += (((self.labda_urg * self.planning_horizon) ** monthly_reserved_blocks) /
                  factorial(monthly_reserved_blocks)) * exp(-self.labda_urg * self.planning_horizon)
        daily_reserved_blocks = int(np.ceil(monthly_reserved_blocks / self.planning_horizon))
        daily_open_blocks_low = daily_blocks_low - daily_reserved_blocks
        daily_open_blocks_high = daily_blocks_high - daily_reserved_blocks
        res_schedule = np.repeat(daily_reserved_blocks, self.planning_horizon)
        open_schedule = schedule - res_schedule

        return daily_reserved_blocks, daily_open_blocks_low, daily_open_blocks_high, days_low, res_schedule,\
            open_schedule

    def cap_incr_res(self):
        daily_reserved_blocks, daily_open_blocks_low, daily_open_blocks_high, days_low, res_schedule, \
            open_schedule = self.cap_cons_res()
        if daily_reserved_blocks > 1:
            start_res = 1
            for i in range(start_res, daily_reserved_blocks):
                res_schedule[daily_reserved_blocks - i - 1] -= i
                open_schedule[daily_reserved_blocks - i - 1] += i
        else:
            start_res = 1
            while start_res * self.labda_urg < 1:
                start_res += 1
            for i in range(start_res):
                res_schedule[i] -= 1
                open_schedule[i] += 1
        # print(open_schedule)
        # print(res_schedule)

        return daily_reserved_blocks, daily_open_blocks_low, daily_open_blocks_high, days_low, start_res, res_schedule,\
            open_schedule

    def admission_time(self):
        at_array = np.ndarray((3, self.planning_horizon + 1), dtype=int)
        for i in range(3):
            at_array[i] = np.repeat(0, self.planning_horizon + 1)
        return at_array

    def occupancy(self):
        occ_rate = np.repeat(0., self.total_period)
        return occ_rate

    def no_res(self, seed_input):
        bl, bh, dl, s = self.cap_no_res(self.phi)
        # print(s)
        at_array = self.admission_time()
        occ_rate = self.occupancy()
        day = 1
        while day <= self.warm_up:
            a = self.arrivals(int(day * seed_input))
            for pat in a:
                i = 0
                if pat == 1 or pat == 2:
                    while i < self.planning_horizon:
                        if s[i] > 0:
                            s[i] -= 1
                            break
                        else:
                            i += 1
                if pat == 3:
                    while i < self.planning_horizon:
                        if s[i] > 1:
                            s[i] -= 2
                            break
                        else:
                            i += 1
            s = np.delete(s, 0)
            s = np.append(s, bh) if day % self.days_week == 0 or day % self.days_week > dl else np.append(s, bl)
            day += 1
        day = 1
        while day <= self.total_period:
            a = self.arrivals(int((day + self.warm_up) * seed_input))
            for pat in a:
                i = 0
                if pat == 1 or pat == 2:
                    while i < self.planning_horizon:
                        if s[i] > 0:
                            s[i] -= 1
                            at_array[pat - 1, i] += 1
                            break
                        else:
                            i += 1
                            if i == self.planning_horizon:
                                at_array[pat - 1, i] += 1
                if pat == 3:
                    while i < self.planning_horizon:
                        if s[i] > 1:
                            s[i] -= 2
                            at_array[pat - 1, i] += 1
                            break
                        else:
                            i += 1
                            if i == self.planning_horizon:
                                at_array[pat - 1, i] += 1
            if day % self.days_week == 0 or day % self.days_week > dl:
                occ_rate[day - 1] = 1 - s[0] / bh
                s = np.delete(s, 0)
                s = np.append(s, bh)
            else:
                occ_rate[day - 1] = 1 - s[0] / bl
                s = np.delete(s, 0)
                s = np.append(s, bl)
            day += 1
        return at_array, occ_rate

    def cons_res(self, seed_input):
        rb, obl, obh, dl, rs, os = self.cap_cons_res()
        # print(os)
        # print(rs)
        at_array = self.admission_time()
        occ_rate = self.occupancy()
        day = 1
        while day <= self.warm_up:
            a = self.arrivals(int(day * seed_input))
            for pat in a:
                i = 0
                if pat == 1:
                    while i < self.planning_horizon:
                        if os[i] > 0:
                            os[i] -= 1
                            break
                        else:
                            i += 1
                if pat == 2:
                    while i < self.planning_horizon:
                        if rs[i] > 0:
                            rs[i] -= 1
                            break
                        elif os[i] > 0:
                            os[i] -= 1
                            break
                        else:
                            i += 1
                if pat == 3:
                    while i < self.planning_horizon:
                        if os[i] > 1:
                            os[i] -= 2
                            break
                        else:
                            i += 1
            if day % self.days_week == 0 or day % self.days_week > dl:
                os = np.delete(os, 0)
                os = np.append(os, obh)
            else:
                os = np.delete(os, 0)
                os = np.append(os, obl)
            rs = np.delete(rs, 0)
            rs = np.append(rs, rb)
            day += 1
        day = 1
        while day <= self.total_period:
            a = self.arrivals(int((day + self.warm_up) * seed_input))
            for pat in a:
                i = 0
                if pat == 1:
                    while i < self.planning_horizon:
                        if os[i] > 0:
                            os[i] -= 1
                            at_array[pat - 1, i] += 1
                            break
                        else:
                            i += 1
                            if i == self.planning_horizon:
                                at_array[pat - 1, i] += 1
                if pat == 2:
                    while i < self.planning_horizon:
                        if rs[i] > 0:
                            rs[i] -= 1
                            at_array[pat - 1, i] += 1
                            break
                        elif os[i] > 0:
                            os[i] -= 1
                            at_array[pat - 1, i] += 1
                            break
                        else:
                            i += 1
                            if i == self.planning_horizon:
                                at_array[pat - 1, i] += 1
                if pat == 3:
                    while i < self.planning_horizon:
                        if os[i] > 1:
                            os[i] -= 2
                            at_array[pat - 1, i] += 1
                            break
                        else:
                            i += 1
                            if i == self.planning_horizon:
                                at_array[pat - 1, i] += 1
            if day % self.days_week == 0 or day % self.days_week > dl:
                occ_rate[day - 1] = 1 - (rs[0] + os[0]) / (rb + obh)
                os = np.delete(os, 0)
                os = np.append(os, obh)
            else:
                occ_rate[day - 1] = 1 - (rs[0] + os[0]) / (rb + obl)
                os = np.delete(os, 0)
                os = np.append(os, obl)
            rs = np.delete(rs, 0)
            rs = np.append(rs, rb)
            day += 1
        return at_array, occ_rate

    def incr_res(self, seed_input):
        rb, obl, obh, dl, sr, rs, os = self.cap_incr_res()
        # print(os)
        # print(rs)
        at_array = self.admission_time()
        occ_rate = self.occupancy()
        day = 1
        while day <= self.warm_up:
            a = self.arrivals(int(day * seed_input))
            for pat in a:
                i = 0
                if pat == 1:
                    while i < self.planning_horizon:
                        if os[i] > 0:
                            os[i] -= 1
                            break
                        else:
                            i += 1
                if pat == 2:
                    while i < self.planning_horizon:
                        if rs[i] > 0:
                            rs[i] -= 1
                            break
                        elif os[i] > 0:
                            os[i] -= 1
                            break
                        else:
                            i += 1
                if pat == 3:
                    while i < self.planning_horizon:
                        if os[i] > 1:
                            os[i] -= 2
                            break
                        else:
                            i += 1
            rs = np.delete(rs, 0)
            rs = np.append(rs, rb)
            os = np.delete(os, 0)
            os = np.append(os, obh) if day % self.days_week == 0 or day % self.days_week > dl else np.append(os, obl)
            if sr == 1:
                for j in range(rb):
                    if rs[j] > j:
                        rs[j] -= rs[j] - j
                        os[j] += rs[j] - j
            else:
                if rs[sr] > 0:
                    rs[sr] -= 1
                    os[sr] += 1
            day += 1
        day = 1
        while day <= self.total_period:
            a = self.arrivals(int((day + self.warm_up) * seed_input))
            for pat in a:
                i = 0
                if pat == 1:
                    while i < self.planning_horizon:
                        if os[i] > 0:
                            os[i] -= 1
                            at_array[pat - 1, i] += 1
                            break
                        else:
                            i += 1
                            if i == self.planning_horizon:
                                at_array[pat - 1, i] += 1
                if pat == 2:
                    while i < self.planning_horizon:
                        if rs[i] > 0:
                            rs[i] -= 1
                            at_array[pat - 1, i] += 1
                            break
                        elif os[i] > 0:
                            os[i] -= 1
                            at_array[pat - 1, i] += 1
                            break
                        else:
                            i += 1
                            if i == self.planning_horizon:
                                at_array[pat - 1, i] += 1
                if pat == 3:
                    while i < self.planning_horizon:
                        if os[i] > 1:
                            os[i] -= 2
                            at_array[pat - 1, i] += 1
                            break
                        else:
                            i += 1
                            if i == self.planning_horizon:
                                at_array[pat - 1, i] += 1
            if day % self.days_week == 0 or day % self.days_week > dl:
                occ_rate[day - 1] = 1 - os[0] / (obh + rb)
            else:
                occ_rate[day - 1] = 1 - os[0] / (obl + rb)
            rs = np.delete(rs, 0)
            rs = np.append(rs, rb)
            os = np.delete(os, 0)
            os = np.append(os, obh) if day % self.days_week == 0 or day % self.days_week > dl else np.append(os, obl)
            if sr == 1:
                for j in range(rb):
                    if rs[j] > j:
                        rs[j] -= rs[j] - j
                        os[j] += rs[j] - j
            else:
                if rs[sr] > 0:
                    rs[sr] -= 1
                    os[sr] += 1

            day += 1
        return at_array, occ_rate

    def free_sel(self, seed_input):
        bl, bh, dl, s = self.cap_no_res(self.phi)
        #print(s)
        at_array = self.admission_time()
        occ_rate = self.occupancy()
        urg_thres = 3
        new_thres = 2 * self.days_week
        day = 1
        while day <= self.warm_up:
            a = self.arrivals(int(day * seed_input))
            for pat in a:
                i = 0
                if pat == 1:
                    if sum(s) > 0:
                        x = np.random.randint(1, sum(s) + 1)
                        while x > s[i]:
                            x = x - s[i]
                            i += 1
                        s[i] -= 1
                if pat == 2:
                    if sum(s[:urg_thres]) > 0:
                        x = np.random.randint(1, sum(s[:urg_thres]) + 1)
                        while x > s[i]:
                            x = x - s[i]
                            i += 1
                        s[i] -= 1
                    elif sum(s) > 0:
                        i = 3
                        while i < self.planning_horizon:
                            if s[i] > 0:
                                s[i] -= 1
                                break
                            else:
                                i += 1
                if pat == 3:
                    if sum(s[:new_thres][s[:new_thres] > 1]) > 1:
                        x = np.random.randint(1, sum(s[:new_thres][s[:new_thres] > 1]) + 1)
                        while x > s[i]:
                            if s[i] > 1:
                                x = x - s[i]
                            i += 1
                        s[i] -= 2
                    elif sum(s[s > 1]) > 1:
                        i = new_thres
                        while i < self.planning_horizon:
                            if s[i] > 1:
                                s[i] -= 2
                                break
                            else:
                                i += 1
            s = np.delete(s, 0)
            s = np.append(s, bh) if day % self.days_week == 0 or day % self.days_week > dl else np.append(s, bl)
            day += 1
        day = 1
        while day <= self.total_period:
            a = self.arrivals(int((day + self.warm_up) * seed_input))
            for pat in a:
                i = 0
                if pat == 1:
                    if sum(s) > 0:
                        x = np.random.randint(1, sum(s) + 1)
                        while x > s[i]:
                            x = x - s[i]
                            i += 1
                        s[i] -= 1
                        at_array[pat - 1, i] += 1
                    else:
                        at_array[pat - 1, -1] += 1
                if pat == 2:
                    if sum(s[:urg_thres]) > 0:
                        x = np.random.randint(1, sum(s[:urg_thres]) + 1)
                        while x > s[i]:
                            x = x - s[i]
                            i += 1
                        s[i] -= 1
                        at_array[pat - 1, i] += 1
                    elif sum(s) > 0:
                        i = 3
                        while i < self.planning_horizon:
                            if s[i] > 0:
                                s[i] -= 1
                                at_array[pat - 1, i] += 1
                                break
                            i += 1
                    else:
                        at_array[pat - 1, -1] += 1
                if pat == 3:
                    if sum(s[:new_thres][s[:new_thres] > 1]) > 1:
                        x = np.random.randint(1, sum(s[:new_thres][s[:new_thres] > 1]) + 1)
                        while x > s[i]:
                            if s[i] > 1:
                                x = x - s[i]
                            i += 1
                        s[i] -= 2
                        at_array[pat - 1, i] += 1
                    elif sum(s[s > 1]) > 1:
                        i = new_thres
                        while i < self.planning_horizon:
                            if s[i] > 1:
                                s[i] -= 2
                                at_array[pat - 1, i] += 1
                                break
                            else:
                                i += 1
                    else:
                        at_array[pat - 1, -1] += 1
            if day % self.days_week == 0 or day % self.days_week > dl:
                occ_rate[day - 1] = 1 - s[0] / bh
                s = np.delete(s, 0)
                s = np.append(s, bh)
            else:
                occ_rate[day - 1] = 1 - s[0] / bl
                s = np.delete(s, 0)
                s = np.append(s, bl)
            day += 1

        return at_array, occ_rate

    def bar(self, pol, at_runs, at, occ):
        at_sum = np.array([sum(at_runs[0::3, :]), sum(at_runs[1::3, :]), sum(at_runs[2::3, :])])
        at_proportions = np.zeros((3, self.planning_horizon + 2))
        for i in range(3):
            at_proportions[i, 1:] = at_sum[i, :] / sum(at_sum[i, :])
        at_proportions_inverse = 1 - at_proportions
        for j in range(3):
            x = list(at_proportions[j, :])
            y = list(at_proportions_inverse[j, :])
            plt.bar(list(range(self.planning_horizon + 2)), y, color='white', bottom=x)
            plt.bar(list(range(self.planning_horizon + 2)), x)
            ticklist = list([0, "", "", "", "", 5, "", "", "", "", 10, "", "", "", "", 15, "", "", "", "", 20, "R"])
            plt.xticks(range(0, self.planning_horizon + 2), ticklist)
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


def statistics_at(at_total, runs):
    lim_u = 6
    at_stats_runs = np.zeros((runs * 3, 9))     # To record AT (pop&mean&var), AT>limit (pop&mean&var), Rejection rate
    for i in range(runs * 3):
        # Total number of patients per type and run
        at_stats_runs[i, 0] = sum(at_total[i, :])

        # Number of patients per type each run but NOT rejected
        at_stats_runs[i, 1] = sum(at_total[i, :-1])

        # Number of patients(urgent only) each run with AT > limit
        at_stats_runs[i, 2] = sum(at_total[i, lim_u:-1]) if i % 3 == 1 else np.NaN

        # Mean AT
        at_stats_runs[i, 3] = np.average(range(1, SimModel.planning_horizon + 1), weights=at_total[i, :-1])

        # Var AT
        var = 0
        for j, weight in enumerate(at_total[i, :-1]):
            var += (weight * ((j + 1) - at_stats_runs[i, 3]) ** 2) / (at_stats_runs[i, 1] - 1)
        at_stats_runs[i, 4] = var

        # Proportion AT > limit (urgent only), Mean AT > limit & Var AT > limit
        if i % 3 != 1:
            at_stats_runs[i, 5] = np.NaN
            at_stats_runs[i, 6] = np.NaN
            at_stats_runs[i, 7] = np.NaN
        else:
            at_stats_runs[i, 5] = at_stats_runs[i, 2] / at_stats_runs[i, 0]
            if at_stats_runs[i, 2] > 0:
                at_stats_runs[i, 6] = np.average(range(lim_u + 1, SimModel.planning_horizon + 1),
                                                 weights=at_total[i, lim_u:-1])
                var_lim = 0
                if at_stats_runs[i, 2] > 1:
                    for j, weight in enumerate(at_total[i, lim_u:-1]):
                        var_lim += (weight * ((j + lim_u + 1) - at_stats_runs[i, 6]) ** 2) / (at_stats_runs[i, 2] - 1)
                at_stats_runs[i, 7] = var_lim
            else:
                at_stats_runs[i, 6] = np.NaN
                at_stats_runs[i, 7] = np.NaN

        # Rejection rate
        at_stats_runs[i, 8] = at_total[i, -1] / at_stats_runs[i, 0]

    at_stats = np.zeros((3, 6))     # To summarize AT (pop&mean&var), AT>limit (pop&mean&var), Rejection rate
    for k in range(3):
        at_stats[k, 0] = np.average(at_stats_runs[k::3, 3], weights=at_stats_runs[k::3, 1])
        at_stats[k, 1] = np.average(at_stats_runs[k::3, 4], weights=(at_stats_runs[k::3, 1] /
                                    sum(at_stats_runs[k::3, 1])) ** 2)
        at_stats[k, 2] = np.average(at_stats_runs[k::3, 5], weights=at_stats_runs[k::3, 0])
        if sum(at_stats_runs[k::3, 2]) > 0:
            at_stats[k, 3] = np.average(at_stats_runs[k::3, 6], weights=at_stats_runs[k::3, 2])
            at_stats[k, 4] = np.average(at_stats_runs[k::3, 7], weights=(at_stats_runs[k::3, 2] /
                                        sum(at_stats_runs[k::3, 2])) ** 2)
        else:
            at_stats[k, 3] = np.NaN
            at_stats[k, 4] = np.NaN
        at_stats[k, 5] = np.average(at_stats_runs[k::3, 8], weights=at_stats_runs[k::3, 0])

    return at_stats_runs, at_stats


def statistics_occ(occ_total, runs):
    occ_stats_runs = np.zeros((runs, 2))
    for i in range(runs):
        occ_stats_runs[i, 0] = np.average(occ_total[i, :])
        occ_stats_runs[i, 1] = np.var(occ_total[i, :])

    occ_stats = np.array([np.average(occ_stats_runs[:, 0]), np.average(occ_stats_runs[:, 1]) / runs])

    return occ_stats_runs, occ_stats


def experiments_2(setup, policy, runs, summary_print=False, runs_print=False, bar_print=False, latex_print=False):
    # Stream of seeds:
    x = [1231, 1322, 2133, 2314, 3125, 3216, 1237, 1328, 2139, 23140]

    # No reservation policy
    if policy == 1 or policy == 5:
        no_res_at_total = np.zeros((runs * 3, SimModel.planning_horizon + 1), dtype=int)
        no_res_occ_total = np.zeros((runs, SimModel.total_period))
        for i in range(runs):
            no_res_at_total[i * 3: i * 3 + 3, :], no_res_occ_total[i, :] = setup.no_res(x[i])
        no_res_at_stats_runs, no_res_at_stats = statistics_at(no_res_at_total, runs)
        no_res_occ_stats_runs, no_res_occ_stats = statistics_occ(no_res_occ_total, runs)
        ind = ["Follow-up", "Urgent", "New"]
        col = ["\u03BC AT", "$\mu$ \u00b2 AT", "Prop AT>Lim", "$\mu$ AT>Lim", "\u03C3\u00b2 AT>Limit", "Rej rate"]
        print("Results of the policy: No reservation")
        if summary_print:
            no_res_at_df = pd.DataFrame(no_res_at_stats, index=ind, columns=col).round(3)
            no_res_occ_df = pd.DataFrame(no_res_occ_stats.reshape(1, 2), index=["Occupancy rate"],
                                         columns=["\u03BC", "\u03C3\u00b2"]).round(3)
            if not latex_print:
                print(no_res_at_df)
                print(no_res_occ_df)
            else:
                print(no_res_at_df.to_latex())
                print(no_res_occ_df.to_latex())
        if runs_print:
            ind = ind * runs
            no_res_at_df = pd.DataFrame(no_res_at_stats_runs[:, 3:], index=ind, columns=col).round(3)
            no_res_occ_df = pd.DataFrame(no_res_occ_stats_runs, index=list(range(1, runs + 1)),
                                         columns=["\u03BC", "\u03C3\u00b2"]).round(3)
            if not latex_print:
                print(no_res_at_df)
                print(no_res_occ_df)
            else:
                print(no_res_at_df.to_latex())
                print(no_res_occ_df.to_latex())
        if bar_print:
            setup.bar("No reservation", no_res_at_total, no_res_at_stats, no_res_occ_stats)

    # Constant reservation policy
    if policy == 2 or policy == 5:
        cons_res_at_total = np.zeros((runs * 3, SimModel.planning_horizon + 1), dtype=int)
        cons_res_occ_total = np.zeros((runs, SimModel.total_period))
        for i in range(runs):
            cons_res_at_total[i * 3: i * 3 + 3, :], cons_res_occ_total[i, :] = setup.cons_res(x[i])
        cons_res_at_stats_runs, cons_res_at_stats = statistics_at(cons_res_at_total, runs)
        cons_res_occ_stats_runs, cons_res_occ_stats = statistics_occ(cons_res_occ_total, runs)
        ind = ["Follow-up", "Urgent", "New"]
        col = ["\u03BC AT", "\u03C3\u00b2 AT", "Prop AT>Lim", "\u03BC AT>Lim", "\u03C3\u00b2 AT>Limit", "Rej rate"]
        print("Results of the policy: Constant reservation")
        if summary_print:
            cons_res_at_df = pd.DataFrame(cons_res_at_stats, index=ind, columns=col).round(3)
            cons_res_occ_df = pd.DataFrame(cons_res_occ_stats.reshape(1, 2), index=["Occupancy rate"],
                                           columns=["\u03BC", "\u03C3\u00b2"]).round(3)
            if not latex_print:
                print(cons_res_at_df)
                print(cons_res_occ_df)
            else:
                print(cons_res_at_df.to_latex())
                print(cons_res_occ_df.to_latex())
        if runs_print:
            ind = ind * runs
            cons_res_at_df = pd.DataFrame(cons_res_at_stats_runs[:, 3:], index=ind, columns=col).round(3)
            cons_res_occ_df = pd.DataFrame(cons_res_occ_stats_runs, index=list(range(1, runs + 1)),
                                           columns=["\u03BC", "\u03C3\u00b2"]).round(3)
            if not latex_print:
                print(cons_res_at_df)
                print(cons_res_occ_df)
            else:
                print(cons_res_at_df.to_latex())
                print(cons_res_occ_df.to_latex())
        if bar_print:
            setup.bar("Constant reservation", cons_res_at_total, cons_res_at_stats, cons_res_occ_stats)

    # Increasing reservation policy
    if policy == 3 or policy == 5:
        incr_res_at_total = np.zeros((runs * 3, SimModel.planning_horizon + 1), dtype=int)
        incr_res_occ_total = np.zeros((runs, SimModel.total_period))
        for i in range(runs):
            incr_res_at_total[i * 3: i * 3 + 3, :], incr_res_occ_total[i, :] = setup.incr_res(x[i])
        incr_res_at_stats_runs, incr_res_at_stats = statistics_at(incr_res_at_total, runs)
        incr_res_occ_stats_runs, incr_res_occ_stats = statistics_occ(incr_res_occ_total, runs)
        ind = ["Follow-up", "Urgent", "New"]
        col = ["\u03BC AT", "\u03C3\u00b2 AT", "Prop AT>Lim", "\u03BC AT>Lim", "\u03C3\u00b2 AT>Limit", "Rej rate"]
        print("Results of the policy: Increasing reservation")
        if summary_print:
            incr_res_at_df = pd.DataFrame(incr_res_at_stats, index=ind, columns=col).round(3)
            incr_res_occ_df = pd.DataFrame(incr_res_occ_stats.reshape(1, 2), index=["Occupancy rate"],
                                           columns=["\u03BC", "\u03C3\u00b2"]).round(3)
            if not latex_print:
                print(incr_res_at_df)
                print(incr_res_occ_df)
            else:
                print(incr_res_at_df.to_latex())
                print(incr_res_occ_df.to_latex())
        if runs_print:
            ind = ind * runs
            incr_res_at_df = pd.DataFrame(incr_res_at_stats_runs[:, 3:], index=ind, columns=col).round(3)
            incr_res_occ_df = pd.DataFrame(incr_res_occ_stats_runs, index=list(range(1, runs + 1)),
                                           columns=["\u03BC", "\u03C3\u00b2"]).round(3)
            if not latex_print:
                print(incr_res_at_df)
                print(incr_res_occ_df)
            else:
                print(incr_res_at_df.to_latex())
                print(incr_res_occ_df.to_latex())
        if bar_print:
            setup.bar("Increasing reservation", incr_res_at_total, incr_res_at_stats, incr_res_occ_stats)

    # Free selection policy
    if policy == 4 or policy == 5:
        free_sel_at_total = np.zeros((runs * 3, SimModel.planning_horizon + 1), dtype=int)
        free_sel_occ_total = np.zeros((runs, SimModel.total_period))
        for i in range(runs):
            free_sel_at_total[i * 3: i * 3 + 3, :], free_sel_occ_total[i, :] = setup.free_sel(x[i])
        free_sel_at_stats_runs, free_sel_at_stats = statistics_at(free_sel_at_total, runs)
        free_sel_occ_stats_runs, free_sel_occ_stats = statistics_occ(free_sel_occ_total, runs)
        ind = ["Follow-up", "Urgent", "New"]
        col = ["\u03BC AT", "\u03C3\u00b2 AT", "Prop AT>Lim", "\u03BC AT>Lim", "\u03C3\u00b2 AT>Limit", "Rej rate"]
        print("Results of the policy: Free selection")
        if summary_print:
            free_sel_at_df = pd.DataFrame(free_sel_at_stats, index=ind, columns=col).round(3)
            free_sel_occ_df = pd.DataFrame(free_sel_occ_stats.reshape(1, 2), index=["Occupancy rate"],
                                           columns=["\u03BC", "\u03C3\u00b2"]).round(3)
            if not latex_print:
                print(free_sel_at_df)
                print(free_sel_occ_df)
            else:
                print(free_sel_at_df.to_latex())
                print(free_sel_occ_df.to_latex())
        if runs_print:
            ind = ind * runs
            free_sel_at_df = pd.DataFrame(free_sel_at_stats_runs[:, 3:], index=ind, columns=col).round(3)
            free_sel_occ_df = pd.DataFrame(free_sel_occ_stats_runs, index=list(range(1, runs + 1)),
                                           columns=["\u03BC", "\u03C3\u00b2"]).round(3)
            if not latex_print:
                print(free_sel_at_df)
                print(free_sel_occ_df)
            else:
                print(free_sel_at_df.to_latex())
                print(free_sel_occ_df.to_latex())
        if bar_print:
            setup.bar("Free selection", free_sel_at_total, free_sel_at_stats, free_sel_occ_stats)


# setup_1 = SimModel(3.26, 0.34, 0.15)
# setup_2 = SimModel(2.93, 0.37, 0.15)
# setup_3 = SimModel(2.61, 0.41, 0.15)
# experiments_2(setup_1, 1, 10, True, True, False, True)    # Policy 1=no_res, 2=cons_res, 3=incr_res, 4=free_sel, 5=all
# experiments_2(setup_2, 4, 10, True, True, True, True)    # Policy 1=no_res, 2=cons_res, 3=incr_res, 4=free_sel, 5=all
# experiments_2(setup_3, 1, 10, True, True, False, True)   # Policy 1=no_res, 2=cons_res, 3=incr_res, 4=free_sel, 5=all
