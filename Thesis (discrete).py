import numpy as np
import scipy
from scipy.stats import poisson
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)            # not working????
scipy.random.seed(3)                        # To "lock" outcome

'''Simple model with no distinction between types'''
def simple_model(labda, mu, Days):
    a = poisson(labda).rvs(Days)            # Generate array with number of arrivals each day
    s = np.ones_like(a) * mu                # Generate equal sizes array with time slots
    s = np.append(s, np.ones_like(a) * mu)  # Add some additional days at the end of time horizon
    AT = np.array([0])                      # Create an array that records admission times of patients
    for i in range(0, len(a)):              # Cycle through all days of time horizon
        j = i + 1                           # Patients cannot be served at day of arrival
        while a[i] > 0:                     # Assign all patients that arrived to a time slot
            if s[j] > 0:                    # Check if time slots are available at day j
                a[i] -= 1                   # To indicate that a patient is assigned
                s[j] -= 1                   # To indicate that a time slot is filled at day j
                AT = np.append(AT, j - i)   # Record admission time of the assigned patient
            else:
                j += 1                      # Check for free time slots next day

    AT = np.delete(AT, 0)                                       # Delete first element (was only used to define AT)

    unique, count = np.unique(np.sort(AT), return_counts=True)  # Record frequency of each admission time observed
    count_norm = count / sum(count)                             # Normalize frequencies
    count_norm_inv = 1 - count_norm                             # Inverse values for white space above bars

    occ = 1 - np.sum(s[1:Days]) / (Days * mu)                   # Compute occupancy rate of servers in time horizon

    #print("a = " + str(a))
    #print("s = " + str(s))
    #print("AT = " + str(AT))
    print("-----------------------------------------------------------------------------------------------------------")
    print("Results simple model with arrival rate " + str(labda) + ", service rate " + str(mu) + ", and time horizon " + str(Days) + " days:")
    print("Mean Admission Time = " + str(AT.mean()))
    print("Standard deviation Admission Time = " + str(np.std(AT)))
    print("Maximum Admission time = " + str(max(AT)))
    print("Occupancy rate = " + str(occ))

    plt.bar(unique, count_norm_inv, color='white', bottom=count_norm)
    plt.bar(unique, count_norm)
    plt.title("Admission Times of patients \n"
              "labda=" + str(labda) + ", mu=" + str(mu))
    plt.xticks(range(1, max(AT) + 1))
    plt.xlabel("Admission Time (in days)")
    plt.ylabel("Proportion of occurrence")
    plt.show()

'''Model that assumes urgent and follow-up types with priority urgent > follow-up'''
def two_types_prio(labda_u, labda_f, mu, Days):
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
            if s[j] > 0:                        # Check if time slots are available at day j
                a_f[i] -= 1                     # To indicate that a patient is assigned
                s[j] -= 1                       # To indicate that a time slot is filled at day j
                AT_f = np.append(AT_f, j - i)   # Record admission time of the assigned urgent patient
            else:
                j += 1                          # Check for free time slots next day

    AT_u = np.delete(AT_u, 0)                                   # Delete first element (was only used to define AT_u)
    AT_f = np.delete(AT_f, 0)                                   # Delete first element (was only used to define AT_f)

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
          + ", service rate " + str(mu) + ", and time horizon " + str(Days) + " days:")
    print("Mean Admission Times: Urgent = "  + str(AT_u.mean()) + "; Follow-up = " + str(AT_f.mean()))
    print("Standard Deviation Admission Time: Urgent = " + str(np.std(AT_u)) + "; Follow-up = " + str(np.std(AT_f)))
    print("Maximum Admission Times: Urgent =  " + str(max(AT_u)) + "; Follow-up = " + str(max(AT_f)))
    print("Occupancy rate = " + str(occ))

    plt.bar(unique_u, count_norm_inv_u, color='white', bottom=count_norm_u)
    plt.bar(unique_u, count_norm_u)
    plt.title("Admission Times of urgent patients \n"
              "labda_u=" + str(labda_u) + ", labda_f=" + str(labda_f) + ", mu=" + str(mu))
    plt.xticks(range(1, max(AT_u) + 1))
    plt.xlabel("Admission Time (in days)")
    plt.ylabel("Proportion of occurrence")
    plt.show()

    plt.bar(unique_f, count_norm_inv_f, color='white', bottom=count_norm_f)
    plt.bar(unique_f, count_norm_f)
    plt.title("Admission Times of follow-up patients \n"
              "labda_u=" + str(labda_u) + ", labda_f=" + str(labda_f) + ", mu=" + str(mu))
    plt.xticks(range(1, max(AT_f) + 1))
    plt.xlabel("Admission Time (in days)")
    plt.ylabel("Proportion of occurrence")
    plt.show()

'''Model that assumes urgent, follow-up, and new types with priority urgent > new > follow-uped'''
def three_types_prio(labda_u, labda_n, labda_f, mu, Days):
    a_u = poisson(labda_u).rvs(Days)            # Generate number of urgent arrivals each day
    a_n = poisson(labda_n).rvs(Days)
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
        while a_n[i] > 0:                       # Assign new patients that arrived to a time slot
            if s[j] > 1:                        # Check if time slots are available at day j
                a_n[i] -= 1                     # To indicate that a patient is assigned
                s[j] -= 2                       # To indicate that two time slot are filled at day j
                AT_n = np.append(AT_n, j - i)   # Record admission time of the assigned urgent patient
            else:
                j += 1                          # Check for free time slots next day
        j = i + 1                               # Reset j to check for single time slots left open by new patients
        while a_f[i] > 0:                       # Assign follow-up patients that arrived to a time slot
            if s[j] > 0:                        # Check if time slots are available at day j
                a_f[i] -= 1                     # To indicate that a patient is assigned
                s[j] -= 1                       # To indicate that a time slot is filled at day j
                AT_f = np.append(AT_f, j - i)   # Record admission time of the assigned urgent patient
            else:
                j += 1                          # Check for free time slots next day

    AT_u = np.delete(AT_u, 0)                                   # Delete first element (was only used to define AT_u)
    AT_n = np.delete(AT_n, 0)                                   # Delete first element (was only used to define AT_n)
    AT_f = np.delete(AT_f, 0)                                   # Delete first element (was only used to define AT_f)

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
    print("Results two types model with urgent arrival rate " + str(labda_u) + ", new arrival rate " + str(labda_n) + ", follow-up arrival rate " + str(labda_f)
          + ", service rate " + str(mu) + ", and time horizon " + str(Days) + " days:")
    print("Mean Admission Times: Urgent = "  + str(AT_u.mean()) + "; New = " + str(AT_n.mean()) + "; Follow-up = " + str(AT_f.mean()))
    print("Standard Deviation Admission Time: Urgent = " + str(np.std(AT_u)) + "; New = " + str(np.std(AT_n)) + "; Follow-up = " + str(np.std(AT_f)))
    print("Maximum Admission Times: Urgent =  " + str(max(AT_u)) + "; New = " + str(max(AT_n)) + "; Follow-up = " + str(max(AT_f)))
    print("Occupancy rate = " + str(occ))

    plt.bar(unique_u, count_norm_inv_u, color='white', bottom=count_norm_u)
    plt.bar(unique_u, count_norm_u)
    plt.title("Admission Times of urgent patients \n"
              "labda_u=" + str(labda_u) + ", labda_n=" + str(labda_n) + ", labda_f=" + str(labda_f) + ", mu=" + str(mu))
    plt.xticks(range(1, max(AT_u) + 1))
    plt.xlabel("Admission Time (in days)")
    plt.ylabel("Proportion of occurrence")
    plt.show()

    plt.bar(unique_n, count_norm_inv_n, color='white', bottom=count_norm_n)
    plt.bar(unique_n, count_norm_n)
    plt.title("Admission Times of new patients \n"
              "labda_u=" + str(labda_u) + ", labda_n=" + str(labda_n) + ", labda_f=" + str(labda_f) + ", mu=" + str(mu))
    plt.xticks(range(1, max(AT_n) + 1))
    plt.xlabel("Admission Time (in days)")
    plt.ylabel("Proportion of occurrence")
    plt.show()

    plt.bar(unique_f, count_norm_inv_f, color='white', bottom=count_norm_f)
    plt.bar(unique_f, count_norm_f)
    plt.title("Admission Times of follow-up patients \n"
              "labda_u=" + str(labda_u) + ", labda_n=" + str(labda_n) + ", labda_f=" + str(labda_f) + ", mu=" + str(mu))
    plt.xticks(range(1, max(AT_f) + 1))
    plt.xlabel("Admission Time (in days)")
    plt.ylabel("Proportion of occurrence")
    plt.show()

'''Test different amount of time slots available for service for one type with priority'''
def simple_mu():
    labda = 5           # Arrival rate of patients per day
    time_horizon = 365   # Length of period considerd
    for i in range(int(np.floor(labda)), int(np.ceil(labda * 1.2) + 2)):
        mu = i          # Evaluate service rates ranging from approximately 100%-120% of arrival rates
        simple_model(labda, mu, time_horizon)

'''Test different amount of time slots available for service for two types with priority'''
def two_types_prio_mu():
    labda_u = 1         # Arrival rate of urgent patients per day
    labda_f = 4         # Arrival rate of follow-up patients per day
    time_horizon = 14   # Length of period considered
    for i in range(int(np.floor(labda_u + labda_f)), int(np.ceil((labda_u + labda_f) * 1.2) + 2)):
        mu = i          # Evaluate service rates ranging from approximately 100%-120% of sum or arrival rates
        two_types_prio(labda_u, labda_f, mu, time_horizon)

'''Test different amount of time slots available for service for three types with priority'''
def three_types_prio_mu():
    labda_u = 1         # Arrival rate of urgent patients per day
    labda_n = 1         # Arrival rate of new patients per day
    labda_f = 4         # Arrival rate of follow-up patients per day
    time_horizon = 365   # Length of period considered
    for i in range(int(np.floor(labda_u + labda_n + labda_f)), int(np.ceil((labda_u + labda_n + labda_f) * 1.2) + 2)):
        mu = i          # Evaluate service rates ranging from approximately 100%-120% of sum or arrival rates
        three_types_prio(labda_u, labda_n, labda_f, mu, time_horizon)



#simple_mu()
#two_types_prio_mu()
#three_types_prio_mu()
