# Global variables for the simulations
import numpy as np
# import os
# base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])

def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-b * (x - a)))

USE_GPU = False

dt = 1 # ms
unit_gran = 4

# Place field parameters
infield_rate = 20.0
background_rate = 0.1
phi_PF_rad = 1 # [unit]

# tuning coefficient of the spike rate if the position is on the boundary of the place field.
# For example, the firing rate of position A when the animal is at position B,
# and if the distance b/w A and B is phi_PF_rad/2, then the firing rate of place cell B is infield_rate * boundary_rate.
boundary_rate = 0.16
tuning_curve_std = np.sqrt(-(phi_PF_rad**2)/(8*np.log(boundary_rate))) # [unit]
place_thr_FR = 15
theta_mod_factor = 0.3

# Network parameters
cell_per_unit = 400
num_IN_neurons = 150
CA3_CA3_connection_prob = 0.1
CA3_CA1_connection_prob = 0.1
IN_connection_prob = 0.25
place_cell_ratio = 1.0

wmax = 10#1e-8
w_init = 1e-2*wmax  # S

tpre = 1000.0
tpost = 500.0

# Firing rate parameters
max_input_FR = infield_rate # [Hz]

rate_shift_CA3 = 4
rate_slope_CA3 = 3

rate_shift_CA1 = 4
rate_slope_CA1 = 2

target_FR_CA3 = 0.35# ((infield_rate/1.5)*0.5 + background_rate*(num_state_total-1))/num_state_total
target_FR_CA1 = 2*target_FR_CA3

# BTSP parameters
ET_amp = 0.4
plateau_amp = 1.6
act_thr = 0.9*ET_amp

## Plateau
base_prob_CA3 = 5e-2
base_prob_CA1 = 5e-2
min_prob_CA1 = 1e-5

firing_prob_slope_CA3 = 5
firing_prob_slope_CA1 = 10

plateau_refrac_duration = 1000 # [ms]
BTSP_scaling_CA3 = 1.#1.5
BTSP_scaling_CA1 = 1.#1.5

# Simulation parameters
sec = 1000 # [ms]
step_time_length = 1*sec # [ms]
softmax_scaling = 0.7
softmax_shift = 2
cue_weight_LR = 1e-2*(1/max_input_FR)*(1/3200)

v_mice = 1.0/sec # [unit/ms]

f_theta = 7.0
dA_granularity = 100
steps_per_unit = step_time_length//dA_granularity

## BTSP kernel
a_pos = 0.8; b_pos = 6
a_neg = 0.05; b_neg = 44.44
pos_base = sigmoid(0, a_pos, b_pos)
pos_denominator = (sigmoid(1, a_pos, b_pos) - sigmoid(0, a_pos, b_pos))
neg_base = sigmoid(0, a_neg, b_neg)
neg_denominator = (sigmoid(1, a_neg, b_neg) - sigmoid(0, a_neg, b_neg))
LR_scaling_factor = 5e-2
k_pos = LR_scaling_factor*1.7; k_neg = LR_scaling_factor*0.204


