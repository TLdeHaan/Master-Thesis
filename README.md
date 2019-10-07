# Master-Thesis

This repository contains Python code used for my master thesis.

Thesis (discrete).py
Discrete simulation of a scheduling problem for patients. One, two, and three types of patients considered. The code gives priority to more urgent patients.

Thesis (discrete, with res).py
Discrete simulation of a scheduling problem for patients. Two and three types of patients considered. The code gives priority to more urgent patients.

Thesis_Models_3.py
This model is eventually used to obtain the results of the simple policies in my thesis. There are three types of patients considered; follow-up, urgent, and new. For the assignment of appointments four policies are applied that differ in the number of serive blocks reserved and the freedom of selection of a service block by the patient.

MDP_3.py
This model is used to try and solve the MDP proposed in my thesis. However, due to memory error, the planning horizon is limit to 5 days such that the MDP cannot produce valuable information as a part of the MDP does not come into affect with this length (minimum would be 8 days).
