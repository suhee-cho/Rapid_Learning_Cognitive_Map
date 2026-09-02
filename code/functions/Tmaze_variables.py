"""
Tmaze_variables.py  [NEW]
=========================
Task configuration for the T-maze environment.

Layout: an L-shaped / T-shaped maze with 10 discrete states.
  States 0–2  : vertical stem (rows 3.5 → 1.5, column 3.5)
  States 3–9  : horizontal arm (row 0.5, columns 0.5 → 6.5)
  The junction is at state 3 (row 0.5, col 3.5).

Two outcome locations:
  State  3 (num_state_row = 3): left-arm end → outcome A  (MI = 3)
  State  9 (num_state_total-1): right-arm end → outcome B (MI = 3)

Exploration sequences (two alternating trajectories per loop):
  Lap type 0: stem → left junction → left arm end  (action sequence [0,0,0,4,6,2,2,2])
  Lap type 1: stem → right junction → right arm end (action sequence [0,0,0,4,7,3,3,3])
  Each `loop_num` repetitions results in tot_lap = 2 * loop_num laps.

replay_trajectory: three canonical CA3 replay paths used for replay-type
  classification in detect_replay_type (run_offline.py):
    Type 0: right arm only          [3,4,5,6,7,8,9]
    Type 1: left arm (reversed)     [0,1,2,6,5,4,3]
    Type 2: right arm from stem     [0,1,2,6,7,8,9]

All variables here are [NEW].
"""

import numpy as np
from common_functions import *
from global_variables import cell_per_unit

# Track parameters
num_state_row = 3   # Vertical stem: 3 states
num_state_col = 7   # Horizontal arm: 7 states
num_state_total = num_state_row+num_state_col  # Total: 10 discrete states
circular = False  # Non-circular topology (dead-ends at both arm tips)

# 2-D (row, col) centre coordinate of each state
state_position = np.array([
    [3.5, 3.5],   # state 0  (1-indexed: 1)  — stem top
    [2.5, 3.5],   # state 1  (1-indexed: 2)  — stem mid
    [1.5, 3.5],   # state 2  (1-indexed: 3)  — stem bottom
    [0.5, 0.5],   # state 3  (1-indexed: 4)  — left-arm end
    [0.5, 1.5],   # state 4  (1-indexed: 5)
    [0.5, 2.5],   # state 5  (1-indexed: 6)
    [0.5, 3.5],   # state 6  (1-indexed: 7)  — horizontal junction
    [0.5, 4.5],   # state 7  (1-indexed: 8)  — right-arm start  [NOT potentiated]
    [0.5, 5.5],   # state 8  (1-indexed: 9)                      [NOT potentiated]
    [0.5, 6.5],   # state 9  (1-indexed: 10) — right-arm end     [NOT potentiated]
])
# 8 discrete actions: cardinal unit steps + half-steps for turning at the junction
actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1],      # full steps: up, down, left, right
                    [-0.5, 0], [0.5, 0], [0, -0.5], [0, 0.5]])  # half steps (used at junction)
num_actions = len(actions)

num_CA3_neurons = num_state_total*cell_per_unit
num_CA1_neurons = num_CA3_neurons

# Two alternating lap types per loop iteration:
#   Row 0: turn left at junction (go to state 3 side)
#   Row 1: turn right at junction (go to state 9 side)
single_lap = np.array([[0,0,0,4,6,2,2,2],[0,0,0,4,7,3,3,3]])
loop_num = 5
tot_lap = single_lap.shape[0]*loop_num  # 10 laps total
detailed_laps = []

# feature_unit_ID[feat]: state IDs where feature feat is physically present.
# Feat 0 = left-arm outcome (state num_state_row = 3)
# Feat 1 = right-arm outcome (state num_state_total-1 = 9)
# Feat 2..11 = location identity (one per state)
feature_unit_ID = [np.array([num_state_row]), np.array([num_state_total-1])] + [np.array([ii]) for ii in range(num_state_total)]

exploration_actions = np.tile(single_lap,(loop_num,1))  # shape: (10, 8)

start_ID = 0
start = state_position[start_ID] + np.array([0.5, 0])  # Start just above the top of the stem

# feature_speed: 0.5 at both outcome locations (animal slows to experience the outcome)
feature_speed = np.concatenate((np.array([0.5,0.5]),np.ones((num_state_total))),axis=0)

# Both outcomes have equal motivational importance (symmetric T-maze)
MI_vector = np.concatenate((np.array([3,3]),np.ones((num_state_total))),axis=0)
value_vector = np.concatenate((np.array([1,1]),np.zeros((num_state_total))),axis=0)
num_features = len(MI_vector)

# Offline learning variables
rest_time = 5000  # [ms] — longer rest to allow complete replay of the longer T-maze trajectory

# Canonical replay trajectories for replay-type classification (used in run_offline.detect_replay_type)
maze_component = [[0,1,2,6],[3,4,5,6],[6,7,8,9]]  # stem, left arm, right arm
replay_trajectory = [maze_component[1][:-1] + maze_component[2],  # shortcut
                     maze_component[0][:-1] + maze_component[1][::-1],  # left arm from stem
                     maze_component[0][:-1] + maze_component[2]]  # right arm from stem
