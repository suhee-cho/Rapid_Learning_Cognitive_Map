import numpy as np
from common_functions import *
from global_variables import cell_per_unit

# Track parameters
num_state_row = 0
num_state_col = 8
num_state_total = num_state_row+num_state_col

state_position = np.array([[0, 0.5], [0, 1.5], [0, 2.5], [0, 3.5], [0, 4.5], [0, 5.5], [0, 6.5], [0, 7.5]])
state_position = state_position[:num_state_total]
actions = np.array([[0, -1], [0, 1]])
num_actions = len(actions)

num_CA3_neurons = num_state_total*cell_per_unit  # Number of neurons
num_CA1_neurons = num_CA3_neurons  # Number of neurons

# Online learning variables
exploration_num = 3
init_cue_exp_num = 1

single_lap = np.ones(num_state_total, dtype=int)
loop_num = 4
tot_lap = loop_num

feature_unit_ID = [np.array([-1]), np.array([num_state_col-1])] + [np.array([ii]) for ii in range(num_state_total)]

cue_lap = np.arange(exploration_num+1, exploration_num+1+init_cue_exp_num)
exploration_actions = np.tile(single_lap,(loop_num,1))

start = np.array([0, 0])

# Feature parameters
feature_speed = np.concatenate((np.array([0.5,2.0]),np.ones((num_state_total))),axis=0) # [unit/ms]

MI_vector = np.concatenate((np.array([0,10]),np.ones((num_state_total))),axis=0) # MI for the cues.
value_vector = np.concatenate((np.array([10,-50]),np.zeros((num_state_total))),axis=0) # Valence for the cues: [neutral, positive, negative]
num_features = len(MI_vector) # Number of cues

# Offline learning variables
rest_time = 10000 # msec

