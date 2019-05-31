import numpy as np
import scipy
from scipy.stats import poisson
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)            # not working????
scipy.random.seed(3)                        # To "lock" outcome

'''Model that assumes urgent and follow-up types with priority for urgent and reserved time slots'''
def two_types_prio_res(labda_u, labda_f, mu, Days, urg_u, rule, start_res):
    a_u = poisson(labda_u).rvs(Days)            # Generate number of urgent arrivals each day
    a_f = poisson(labda_f).rvs(Days)            # Generate number of follow-up arrivals each day
    s = np.ones_like(a_u) * mu                  # Generate equal sized array with time slots
    s = np.append(s, np.ones_like(a_u) * mu)    # Add some additional days at the end of time horizon
    AT_u = np.array([0])                        # Create an array that records admission times of urgent patients
    AT_f = np.array([0])                        # Create an array that records admission times of follow-up patients
    for i in range(0, len(a_u)):                # Cycle through all days of time horizon
        j = i + 1                               # Patients cannot be served at day of arrival
        while a_u[i] > 0:                       # Assign urgent patients that arrived to a time slot
            if s[j] > 0:                        # Check if time slots are available at day j
                a_u[i] -= 1                     # To indicate that a patient is assigned
                s[j] -= 1                       # To indicate that a time slot is filled at day j
                AT_u = np.append(AT_u, j - i)   # Record admission time of the assigned urgent patient
            else:
                j += 1                          # Check for free time slots next day
        while a_f[i] > 0:                       # Assign follow-up patients that arrived to a time slot
            if rule == "staircase":             # Rule applied to reserving time slots
                if s[j] > max(0, min(mu - 1, mu - 1 - (urg_u - (j - i)))):  # Check if time slots are available at day j
                    a_f[i] -= 1                     # To indicate that a patient is assigned
                    s[j] -= 1                       # To indicate that a time slot is filled at day j
                    AT_f = np.append(AT_f, j - i)   # Record admission time of the assigned urgent patient
                else:
                    j += 1                                  # Check for free time slots next day
            elif rule != "staircase":                       # Rule applied to reserving time slots
                if s[j] > 0 and j - i <= start_res - 1:     # Check if time slots are available at day j
                    a_f[i] -= 1                             # To indicate that a patient is assigned
                    s[j] -= 1                               # To indicate that a time slot is filled at day j
                    AT_f = np.append(AT_f, j - i)           # Record admission time of the assigned urgent patient
                elif s[j] > rule:                           # Check if time slots are available at day j
                    a_f[i] -= 1                             # To indicate that a patient is assigned
                    s[j] -= 1                               # To indicate that a time slot is filled at day j
                    AT_f = np.append(AT_f, j - i)           # Record admission time of the assigned urgent patient
                else:
                    j += 1                          # Check for free time slots next day

    AT_u = np.delete(AT_u, 0)                       # Delete first element (was only used to define AT_u)
    AT_f = np.delete(AT_f, 0)                       # Delete first element (was only used to define AT_f)

    unique_u, count_u = np.unique(np.sort(AT_u), return_counts=True)  # Record frequency of each admission time observed
    count_norm_u = count_u / sum(count_u)                             # Normalize frequencies
    count_norm_inv_u = 1 - count_norm_u                               # Inverse values for white space above bars

    unique_f, count_f = np.unique(np.sort(AT_f), return_counts=True)  # Record frequency of each admission time observed
    count_norm_f = count_f / sum(count_f)                             # Normalize frequencies
    count_norm_inv_f = 1 - count_norm_f                               # Inverse values for white space above bars

    occ = 1 - np.sum(s[1:Days]) / (Days * mu)                         # Compute occupancy rate of servers in time horizon

    #print("a_u = " + str(a_u))
    #print("a_f = " + str(a_f))
    #print("s = " + str(s))
    #print("AT_u = " + str(AT_u))
    #print("AT_f = " + str(AT_f))
    print("-----------------------------------------------------------------------------------------------------------")
    print("Results two types model with urgent arrival rate " + str(labda_u) + ", follow-up arrival rate " + str(labda_f)
          + ", service rate " + str(mu) + ", and time horizon " + str(Days) + " days, reserved slots " + str(rule) + ", and start reserving slots at " + str(start_res) + ":")
    print("Mean Admission Times: Urgent = "  + str(AT_u.mean()) + "; Follow-up = " + str(AT_f.mean()))
    print("Standard Deviation Admission Time: Urgent = " + str(np.std(AT_u)) + "; Follow-up = " + str(np.std(AT_f)))
    print("Maximum Admission Times: Urgent =  " + str(max(AT_u)) + "; Follow-up = " + str(max(AT_f)))
    print("Occupancy rate = " + str(occ))

    plt.bar(unique_u, count_norm_inv_u, color='white', bottom=count_norm_u)
    plt.bar(unique_u, count_norm_u)
    plt.title("Admission Times of urgent patients \n"
              "labda_u=" + str(labda_u) + ", labda_f=" + str(labda_f) + ", mu=" + str(mu) + ",res slots=" + str(rule) + ", start res=" + str(start_res))
    plt.xticks(range(1, max(AT_u) + 1))
    plt.xlabel("Admission Time (in days)")
    plt.ylabel("Proportion of occurrence")
    plt.show()

    plt.bar(unique_f, count_norm_inv_f, color='white', bottom=count_norm_f)
    plt.bar(unique_f, count_norm_f)
    plt.title("Admission Times of follow-up patients \n"
              "labda_u=" + str(labda_u) + ", labda_f=" + str(labda_f) + ", mu=" + str(mu) + ",res slots=" + str(rule) + ", start res=" + str(start_res))
    plt.xticks(range(1, max(AT_f) + 1))
    plt.xlabel("Admission Time (in days)")
    plt.ylabel("Proportion of occurrence")
    plt.show()



'''Model that assumes urgent, new, and follow-up types with priority urgent > new > follow-up and reserved time slots'''
def three_types_prio_res(labda_u, labda_n, labda_f, mu, Days, urg_u, rule, start_res):
    a_u = poisson(labda_u).rvs(Days)            # Generate number of urgent arrivals each day
    a_n = poisson(labda_n).rvs(Days)            # Generate number of new arrivals each day
    a_f = poisson(labda_f).rvs(Days)            # Generate number of follow-up arrivals each day
    s = np.ones_like(a_u) * mu                  # Generate equal sized array with time slots
    s = np.append(s, np.ones_like(a_u) * mu)    # Add some additional days at the end of time horizon
    AT_u = np.array([0])                        # Create an array that records admission times of urgent patients
    AT_n = np.array([0])                        # Create an array that records admission times of new patients
    AT_f = np.array([0])                        # Create an array that records admission times of follow-up patients
    for i in range(0, len(a_u)):                # Cycle through all days of time horizon
        j = i + 1                               # Patients cannot be served at day of arrival
        while a_u[i] > 0:                       # Assign urgent patients that arrived to a time slot
            if s[j] > 0:                        # Check if time slots are available at day j
                a_u[i] -= 1                     # To indicate that a patient is assigned
                s[j] -= 1                       # To indicate that a time slot is filled at day j
                AT_u = np.append(AT_u, j - i)   # Record admission time of the assigned urgent patient
            else:
                j += 1                          # Check for free time slots next day
        while a_n[i] > 0:                       # Assign follow-up patients that arrived to a time slot
            if rule == "staircase":             # Rule applied to reserving time slots
                if s[j] > max(1, min(mu - 1, mu - (urg_u - (j - i)))):
                    a_n[i] -= 1                     # To indicate that a patient is assigned
                    s[j] -= 2                       # To indicate that two time slots are filled at day j
                    AT_n = np.append(AT_n, j - i)   # Record admission time of the assigned urgent patient
                else:
                    j += 1                          # Check for free time slots next day
            elif rule != "staircase":               # Rule applied to reserving time slots
                if s[j] > 1 and j - i <= start_res - 1:     # Check if time slots are available at day j
                    a_n[i] -= 1                     # To indicate that a patient is assigned
                    s[j] -= 2                       # To indicate that two time slots are filled at day j
                    AT_n = np.append(AT_n, j - i)   # Record admission time of the assigned urgent patient
                elif s[j] > rule + 1:               # Check if time slots are available at day j
                    a_n[i] -= 1                     # To indicate that a patient is assigned
                    s[j] -= 2                       # To indicate that two time slots are filled at day j
                    AT_n = np.append(AT_n, j - i)   # Record admission time of the assigned urgent patient
                else:
                    j += 1                          # Check for free time slots next day
        j = i + 1                                   # Reset j to check for single time slots left open by new patients
        while a_f[i] > 0:                           # Assign follow-up patients that arrived to a time slot
            if rule == "staircase":                 # Rule applied to reserving time slots
                if s[j] > max(0, min(mu - 1, mu - 1 - (urg_u - (j - i)))):
                    a_f[i] -= 1                     # To indicate that a patient is assigned
                    s[j] -= 1                       # To indicate that a time slot is filled at day j
                    AT_f = np.append(AT_f, j - i)   # Record admission time of the assigned urgent patient
                else:
                    j += 1                          # Check for free time slots next day
            elif rule != "staircase":               # Rule applied to reserving time slots
                if s[j] > 0 and j - i <= start_res - 1:     # Check if time slots are available at day j
                    a_f[i] -= 1                     # To indicate that a patient is assigned
                    s[j] -= 1                       # To indicate that a time slot is filled at day j
                    AT_f = np.append(AT_f, j - i)   # Record admission time of the assigned urgent patient
                elif s[j] > rule:                   # Check if time slots are available at day j
                    a_f[i] -= 1                     # To indicate that a patient is assigned
                    s[j] -= 1                       # To indicate that a time slot is filled at day j
                    AT_f = np.append(AT_f, j - i)   # Record admission time of the assigned urgent patient
                else:
                    j += 1                          # Check for free time slots next day

    AT_u = np.delete(AT_u, 0)                       # Delete first element (was only used to define AT_u)
    AT_n = np.delete(AT_n, 0)                       # Delete first element (was only used to define AT_n)
    AT_f = np.delete(AT_f, 0)                       # Delete first element (was only used to define AT_f)

    unique_u, count_u = np.unique(np.sort(AT_u), return_counts=True)  # Record frequency of each admission time observed
    count_norm_u = count_u / sum(count_u)                             # Normalize frequencies
    count_norm_inv_u = 1 - count_norm_u                               # Inverse values for white space above bars

    unique_n, count_n = np.unique(np.sort(AT_n), return_counts=True)  # Record frequency of each admission time observed
    count_norm_n = count_n / sum(count_n)                             # Normalize frequencies
    count_norm_inv_n = 1 - count_norm_n                               # Inverse values for white space above bars

    unique_f, count_f = np.unique(np.sort(AT_f), return_counts=True)  # Record frequency of each admission time observed
    count_norm_f = count_f / sum(count_f)                             # Normalize frequencies
    count_norm_inv_f = 1 - count_norm_f                               # Inverse values for white space above bars

    occ = 1 - np.sum(s[1:Days - 6]) / ((Days - 7) * mu)                 # Compute occupancy rate of servers in time horizon

    #print("a_u = " + str(a_u))
    #print("a_n = " + str(a_n))
    #print("a_f = " + str(a_f))
    #print("s = " + str(s))
    #print("AT_u = " + str(AT_u))
    #print("AT_n = " + str(AT_n))
    #print("AT_f = " + str(AT_f))
    print("-----------------------------------------------------------------------------------------------------------")
    print("Results two types model with urgent arrival rate " + str(labda_u) + ", follow-up arrival rate " + str(labda_f)
          + ", service rate " + str(mu) + ", and time horizon " + str(Days) + " days, reserved slots " + str(rule) + ", and start reserving slots at " + str(start_res) + ":")
    print("Mean Admission Times: Urgent = " + str(AT_u.mean()) + "; New = " + str(AT_n.mean()) + "; Follow-up = " + str(AT_f.mean()))
    print("Standard Deviation Admission Time: Urgent = " + str(np.std(AT_u)) + "; New = " + str(np.std(AT_n)) + "; Follow-up = " + str(np.std(AT_f)))
    print("Maximum Admission Times: Urgent =  " + str(max(AT_u)) + "; New = " + str(max(AT_n)) + "; Follow-up = " + str(max(AT_f)))
    print("Occupancy rate = " + str(occ))

    plt.bar(unique_u, count_norm_inv_u, color='white', bottom=count_norm_u)
    plt.bar(unique_u, count_norm_u)
    plt.title("Admission Times of urgent patients \n"
              "labda_u=" + str(labda_u) + ", labda_n=" + str(labda_n) + ", labda_f=" + str(labda_f) + ", mu=" + str(mu) + ",res slots=" + str(rule) + ", start res=" + str(start_res))
    plt.xticks(range(1, max(AT_u) + 1))
    plt.xlabel("Admission Time (in days)")
    plt.ylabel("Proportion of occurrence")
    plt.show()

    plt.bar(unique_n, count_norm_inv_n, color='white', bottom=count_norm_n)
    plt.bar(unique_n, count_norm_n)
    plt.title("Admission Times of new patients \n"
              "labda_u=" + str(labda_u) + ", labda_n=" + str(labda_n) + ", labda_f=" + str(labda_f) + ", mu=" + str(mu) + ",res slots=" + str(rule) + ", start res=" + str(start_res))
    plt.xticks(range(1, max(AT_n) + 1))
    plt.xlabel("Admission Time (in days)")
    plt.ylabel("Proportion of occurrence")
    plt.show()

    plt.bar(unique_f, count_norm_inv_f, color='white', bottom=count_norm_f)
    plt.bar(unique_f, count_norm_f)
    plt.title("Admission Times of follow-up patients \n"
              "labda_u=" + str(labda_u) + ", labda_n=" + str(labda_n) + ", labda_f=" + str(labda_f) + ", mu=" + str(mu) + ",res slots=" + str(rule) + ", start res=" + str(start_res))
    plt.xticks(range(1, max(AT_f) + 1))
    plt.xlabel("Admission Time (in days)")
    plt.ylabel("Proportion of occurrence")
    plt.show()

'''Test different amount of time slots available for service for two types with priority and reserved slots'''
def two_types_prio_res_mu():
    labda_u = 1             # Arrival rate of urgent patients per day
    labda_f = 4             # Arrival rate of follow-up patients per day
    time_horizon = 365      # Length of period considered
    urg_u = 8               # Maximum admission time allowed for urgent patients
    rule = "staircase"      # Reserve increasing number slots as urg_u is approached
    for i in range(int(np.floor(labda_u + labda_f)), int(np.ceil((labda_u + labda_f) * 1.2) + 2)):
        mu = i              # Evaluate service rates ranging from approximately 100%-120% of sum or arrival rates
        start_res = urg_u - mu + 2  # Number of days before urg_u the reservation rule should start
        two_types_prio_res(labda_u, labda_f, mu, time_horizon, urg_u, rule, start_res)

'''Test different amount of time slots reserved and from which day onward to reserve'''
def two_types_prio_res_rule():
    labda_u = 1             # Arrival rate of urgent patients per day
    labda_f = 4             # Arrival rate of follow-up patients per day
    mu = 5                  # Number of time slots available per day
    time_horizon = 21       # Length of period considered
    urg_u = 8               # Maximum admission time allowed for urgent patients
    for i in range(1, labda_u + 1):
        res_slots = i       # Reserve i number of time slots
        for j in range(1, urg_u + 1):
            start_res = j
            two_types_prio_res(labda_u, labda_f, mu, time_horizon, urg_u, res_slots, start_res)

'''Test different amount of time slots available for service for three types with priority and reserved slots'''
def three_types_prio_res_mu():
    labda_u = 1             # Arrival rate of urgent patients per day
    labda_n = 1             # Arrival rate of new patients per day
    labda_f = 4             # Arrival rate of follow-up patients per day
    time_horizon = 365      # Length of period considered
    urg_u = 8               # Maximum admission time allowed for urgent patients
    rule = "staircase"      # Reserve increasing number slots as urg_u is approached
    for i in range(int(np.floor(labda_u + labda_n + labda_f)), int(np.ceil((labda_u + labda_n + labda_f) * 1.2) + 2)):
        mu = i              # Evaluate service rates ranging from approximately 100%-120% of sum or arrival rates
        start_res = urg_u - mu + 2  # Number of days before urg_u the reservation rule should start
        three_types_prio_res(labda_u, labda_n, labda_f, mu, time_horizon, urg_u, rule, start_res)

'''Test different amount of time slots reserved and from which day onward to reserve'''
def three_types_prio_res_rule():
    labda_u = 1             # Arrival rate of urgent patients per day
    labda_n = 1             # Arrival rate of new patients per day
    labda_f = 4             # Arrival rate of follow-up patients per day
    mu = 5                  # Number of time slots available per day
    time_horizon = 21       # Length of period considered
    urg_u = 8               # Maximum admission time allowed for urgent patients
    for i in range(1, labda_u + 1):
        res_slots = i       # Reserve i number of time slots
        for j in range(1, urg_u + 1):
            start_res = j   # Start reserving j days in the future
            three_types_prio_res(labda_u, labda_n, labda_f, mu, time_horizon, urg_u, res_slots, start_res)



#two_types_prio_res_mu()
#two_types_prio_res_rule()
#three_types_prio_res_mu()
#three_types_prio_res_rule()