"""
linear_reward_variables.py  [NEW]
==================================
Task configuration for the Linear Reward environment.

Layout: a straight 1-D track with 8 discrete states (columns 0–7).
The animal always runs left-to-right (action 1 = [0,+1]).

Feature structure (num_features = 8 + 2 = 10):
    features[0] — reward cue (positive valence; present at state 3 in early laps,
                               at state 7 in later laps, depending on cue_lap)
    features[1] — irrelevant/null cue (always absent; MI = 0 makes it harmless)
    features[2..9] — location identity cues (one per state; always present at
                      the animal's current state)

Reward schedule:
    Laps 6–10  (cue_lap[0]): reward cue appears at feature_unit_ID[0][0] = state 3
    Laps 11–15 (cue_lap[1]): reward cue appears at feature_unit_ID[1][0] = state 7

MI_vector (Motivational Importance):
    [3, 10, 1, 1, 1, 1, 1, 1, 1, 1]
    High MI for the reward cue (index 1, MI=10) ensures large perceived salience.

All variables here are [NEW] (specific to this task).
"""

import numpy as np
from common_functions import *
from global_variables import cell_per_unit

# Track parameters
num_state_row = 0    # No vertical segment; pure horizontal track
num_state_col = 8    # 8 discrete spatial states along the track
num_state_total = num_state_row+num_state_col

# 2-D coordinates of each state's centre (row=0 for horizontal track)
state_position = np.array([[0, 0.5], [0, 1.5], [0, 2.5], [0, 3.5], [0, 4.5], [0, 5.5], [0, 6.5], [0, 7.5]])
actions = np.array([[0, -1], [0, 1]])  # Discrete actions: move left / right
num_actions = len(actions)

# Each maze state is represented by `cell_per_unit` CA3 and CA1 neurons.
num_CA3_neurons = num_state_total*cell_per_unit
num_CA1_neurons = num_CA3_neurons

# ---------------------------------------------------------------------------
# Online learning schedule
# ---------------------------------------------------------------------------
exploration_num = 5       # Number of pure exploration laps (no cue) before any reward
init_cue_exp_num = 5      # Number of laps in first reward phase (cue at state 3)

# Each lap traverses the track once left-to-right; action 1 = [0,+1] at every step.
single_lap = np.ones(num_state_total, dtype=int)
loop_num = 15             # Total number of laps
tot_lap = loop_num

# feature_unit_ID[phase][feature_idx] = array of state IDs where that feature is active.
# Phase 0 (early laps): reward cue at state 3; null cue never active (state -1).
# Phase 1 (late laps):  reward cue moves to state 7 (the end of the track).
feature_unit_ID = [[np.array([3]), np.array([-1])] + [np.array([ii]) for ii in range(num_state_total)],
                   [np.array([num_state_total-1]), np.array([-1])] + [np.array([ii]) for ii in range(num_state_total)]]

# cue_lap[0]: laps during which the first reward schedule is active.
# cue_lap[1]: laps during which the second (shifted) reward schedule is active.
cue_lap = [np.arange(exploration_num+1, exploration_num+1+init_cue_exp_num),
           np.arange(exploration_num+1+init_cue_exp_num, tot_lap+1)]
exploration_actions = np.tile(single_lap,(loop_num,1))  # shape: (tot_lap, num_state_total)

start = np.array([0, 0])  # Animal starts at position (row=0, col=0)

# ---------------------------------------------------------------------------
# Feature parameters
# ---------------------------------------------------------------------------
# feature_speed: effective movement speed multiplier when the animal is at a
# location where each feature is active.  0.5 = animal slows at the reward cue.
feature_speed = np.concatenate((np.array([0.5,0.0]),np.ones((num_state_total))),axis=0)

# MI_vector (Motivational Importance): scales PS_update; higher MI → faster
# CA1 learning when the animal encounters a surprising version of that feature.
MI_vector = np.concatenate((np.array([3,10]),np.ones((num_state_total))),axis=0)

# value_vector: signed valence of each feature (used for downstream analysis only).
value_vector = np.concatenate((np.array([1,-1]),np.zeros((num_state_total))),axis=0)
num_features = len(MI_vector)  # Total number of feature channels

# ---------------------------------------------------------------------------
# Offline learning variables
# ---------------------------------------------------------------------------
rest_time = 3000  # Duration of the offline (SWR replay) simulation [ms]

