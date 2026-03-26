import numpy as np
from common_functions import *
from global_variables import cell_per_unit

# Track parameters
num_state_row = 3
num_state_col = 7
num_state_total = num_state_row+num_state_col
circular = False

state_position = np.array([[3.5, 3.5], [2.5, 3.5], [1.5, 3.5], \
                            [0.5, 0.5], [0.5, 1.5], [0.5, 2.5], [0.5, 3.5], [0.5, 4.5], [0.5, 5.5], [0.5, 6.5]])
actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [-0.5, 0], [0.5, 0], [0, -0.5], [0, 0.5]])
num_actions = len(actions)

num_CA3_neurons = num_state_total*cell_per_unit  # Number of neurons
num_CA1_neurons = num_CA3_neurons  # Number of neurons

single_lap = np.array([[0,0,0,4,6,2,2,2],[0,0,0,4,7,3,3,3]])
loop_num = 5
tot_lap = single_lap.shape[0]*loop_num

feature_unit_ID = [np.array([num_state_row]), np.array([num_state_total-1])] + [np.array([ii]) for ii in range(num_state_total)]

exploration_actions = np.tile(single_lap,(loop_num,1))

start_ID = 0
start = state_position[start_ID] + np.array([0.5, 0])

# Feature parameters
feature_speed = np.concatenate((np.array([0.5,0.5]),np.ones((num_state_total))),axis=0) # [unit/ms]

MI_vector = np.concatenate((np.array([3,3]),np.ones((num_state_total))),axis=0) # MI for the cues.
value_vector = np.concatenate((np.array([1,1]),np.zeros((num_state_total))),axis=0) # Valence for the cues: [neutral, positive, negative]
num_features = len(MI_vector) # Number of cues

# Offline learning variables
rest_time = 5000 # msec
replay_trajectory = [[3,4,5,6,7,8,9],[0,1,2,6,5,4,3],[0,1,2,6,7,8,9]]