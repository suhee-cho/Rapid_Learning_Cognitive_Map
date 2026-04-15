"""
global_variables.py
===================
Project-wide hyperparameters shared by all task environments and both the
online and offline simulation pipelines.  Every module imports this file with
``from global_variables import *``.

All constants here are NEW (specific to this project) unless noted.
"""

import numpy as np
# import os
# base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])

def sigmoid(x, a, b):
    """Logistic sigmoid: 1 / (1 + exp(-b*(x-a))).  Used as a transfer function throughout."""
    return 1 / (1 + np.exp(-b * (x - a)))

USE_GPU = False  # GPU acceleration toggle (currently unused; kept for future use)

dt = 1          # Simulation time step [ms]
unit_gran = 4   # Spatial grid resolution: number of sub-samples per maze unit

# ---------------------------------------------------------------------------
# Place field parameters
# ---------------------------------------------------------------------------
infield_rate = 20.0    # Peak firing rate of a CA3 place cell at its field centre [Hz]
background_rate = 0.1  # Baseline (out-of-field) firing rate [Hz]
phi_PF_rad = 1         # Place field radius [maze units]

# Gaussian tuning curve width derived from the boundary-rate constraint:
# the firing rate at the field edge (distance = phi_PF_rad/2) equals
# infield_rate * boundary_rate.
boundary_rate = 0.16
tuning_curve_std = np.sqrt(-(phi_PF_rad**2)/(8*np.log(boundary_rate))) # [maze units]
place_thr_FR = 15      # Minimum peak rate to classify a CA1 neuron as a place cell [Hz]
theta_mod_factor = 0.3 # Theta-modulation attenuation factor applied to CA3 input to CA1

# ---------------------------------------------------------------------------
# Network structure
# ---------------------------------------------------------------------------
cell_per_unit = 400          # Number of CA3/CA1 neurons assigned to each maze state
num_IN_neurons = 150         # Number of fast-spiking interneurons per region (CA3/CA1)
CA3_CA3_connection_prob = 0.1  # Probability of a recurrent CA3→CA3 synapse
CA3_CA1_connection_prob = 0.1  # Probability of a feedforward CA3→CA1 synapse
IN_connection_prob = 0.25    # Probability of PC→IN and IN→PC / IN→IN synapses
place_cell_ratio = 1.0       # Fraction of CA3 neurons with a place field (1.0 = all)

wmax = 10        # Maximum synaptic weight (hard clip in BTSP_update and update_STDP)
w_init = 1e-2*wmax  # Initial weight (mean of the Gaussian initialisation)

# Eligibility / plateau trace decay time constants (online simulation):
tpre  = 1000.0   # Pre-synaptic (ET) decay time constant [ms]
tpost = 500.0    # Post-synaptic (PT) decay time constant [ms]

# ---------------------------------------------------------------------------
# Firing rate transfer function (sigmoid from CA3 input to CA1 rate)
# ---------------------------------------------------------------------------
max_input_FR = infield_rate  # Maximum achievable firing rate [Hz] (sigmoid ceiling)

# Sigmoid threshold (shift) and slope for CA3 and CA1 rate-to-input conversion:
rate_shift_CA3 = 4;  rate_slope_CA3 = 3
rate_shift_CA1 = 4;  rate_slope_CA1 = 2

# Homeostatic target firing rates used in plateau_probability_calc:
target_FR_CA3 = 0.35   # Target average CA3 population rate [Hz]
target_FR_CA1 = 2*target_FR_CA3  # CA1 target is set higher to allow richer activity

# ---------------------------------------------------------------------------
# BTSP parameters
# ---------------------------------------------------------------------------
ET_amp = 0.4       # Eligibility trace increment per spike
plateau_amp = 1.6  # Plateau trace increment when a plateau is initiated
act_thr = 0.9*ET_amp  # ET value below which the trace is reset to zero (avoids hyper-depression)

# Plateau probability parameters (see plateau_probability_calc):
base_prob_CA3 = 5e-2   # Baseline plateau probability for CA3 neurons
base_prob_CA1 = 5e-2   # Baseline plateau probability for CA1 neurons
min_prob_CA1  = 1e-5   # Minimum probability floor for CA1 (prevents full suppression)

firing_prob_slope_CA3 = 5   # Sigmoid slope of the inhibitory term in plateau_probability_calc
firing_prob_slope_CA1 = 10

plateau_refrac_duration = 1000  # Post-plateau refractory period [ms]
BTSP_scaling_CA3 = 1.  # Global multiplier on ET⊗PT product before the BTSP sigmoid
BTSP_scaling_CA1 = 1.

# ---------------------------------------------------------------------------
# Simulation timing
# ---------------------------------------------------------------------------
sec = 1000              # Milliseconds per second (unit conversion constant)
step_time_length = 1*sec  # Nominal duration for traversing one maze unit [ms]
softmax_scaling = 0.7   # Softmax temperature scaling (used in compute_transition_matrix)
softmax_shift = 2
cue_weight_LR = 1e-2*(1/max_input_FR)*(1/3200)  # Learning rate for CA1→feature delta rule

v_mice = 1.0/sec  # Baseline animal running speed [maze units / ms]

f_theta = 7.0            # Theta oscillation frequency [Hz] (phase precession modulation)
dA_granularity = 100     # Spike regeneration interval [ms]; spike trains are redrawn every this many ms
steps_per_unit = step_time_length//dA_granularity  # Number of dA windows per maze unit

# ---------------------------------------------------------------------------
# BTSP kernel shape parameters (sigmoidal LTP / LTD functions)
# The weight update amplitude for a given ET⊗PT coincidence is:
#   LTP: k_pos * (sigmoid(ET⊗PT, a_pos, b_pos) - pos_base) / pos_denominator
#   LTD: k_neg * (sigmoid(ET⊗PT, a_neg, b_neg) - neg_base) / neg_denominator
# Normalising by pos_denominator / neg_denominator maps the sigmoid output
# to [0, 1] over the range [0, 1] of normalised ET⊗PT values.
# ---------------------------------------------------------------------------
a_pos = 0.8;  b_pos = 6      # LTP sigmoid: threshold and slope
a_neg = 0.05; b_neg = 44.44  # LTD sigmoid: threshold and slope (steep near 0)
pos_base        = sigmoid(0, a_pos, b_pos)
pos_denominator = (sigmoid(1, a_pos, b_pos) - sigmoid(0, a_pos, b_pos))
neg_base        = sigmoid(0, a_neg, b_neg)
neg_denominator = (sigmoid(1, a_neg, b_neg) - sigmoid(0, a_neg, b_neg))
LR_scaling_factor = 5e-2   # Global learning rate scale
k_pos = LR_scaling_factor*1.7;   k_neg = LR_scaling_factor*0.204  # LTP / LTD amplitudes


