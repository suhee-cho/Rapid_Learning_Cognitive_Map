"""
linear_shock_functions.py
=========================
Task-specific functions for the Linear Shock environment.

This module is structurally identical to linear_reward_functions.py.
The only behavioural difference is the presence_update logic, which marks
the aversive cue (shock at state 7) as active when the lap is in `cue_lap`.

Attribution guide (same convention as linear_reward_functions.py):
  All functions in this file follow the same attribution as their counterparts
  in linear_reward_functions.py (see that file's module docstring for details).
  Key differences from the reward version:
    - cue_lap is a 1-D array (single phase) rather than a list of two phases.
    - presence_update checks a single cue_lap (not list of two) and sets the
      shock feature (index 1) active rather than the reward feature (index 0).
    - calc_distance does NOT use wrap-around (same as linear track, no circular).
"""

from global_variables import *
from linear_shock_variables import *
import numpy as np
from common_functions import *
from tqdm import tqdm

base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
data_path = os.path.join(base_path,"results/linear_shock")
pklf_name = os.path.join(data_path, "PF_peak_data.pkl")

def analyse_replay_type(spk_time, spk_neurons, rate, target_trajectory=[[3,4,5,6],[3,2,1,0]],
                        coverage_thr=0.75, save_path=False, verbose=True):
    """
    Classify detected replay events into canonical T-maze trajectory types.

    For each trajectory in `target_trajectory`, isolates the CA3 neurons whose
    place fields lie along that path (neurons may overlap across trajectory types),
    recomputes the population rate from those neurons, runs `analyse_replay`, and
    accepts events where ≥ coverage_thr of decoded positions fall within the
    trajectory's spatial range.

    Parameters
    ----------
    spk_time : np.ndarray
        Spike times [ms] from the offline CA3 simulation.
    spk_neurons : np.ndarray of int
        Neuron index for each spike.
    rate : np.ndarray
        Full CA3 population rate time series; used only for its length and time span
        to set the binning resolution for the per-trajectory rate.
    target_trajectory : list of lists  (default: replay_trajectory)
        Each entry is a list of state IDs defining one full canonical replay path.
        Defaults to the three types in Tmaze_variables.replay_trajectory:
          [0] forward        [3,4,5,6]
          [1] backward  [0,1,2,3]
    coverage_thr : float
        Minimum fraction of decoded positions that must fall within the
        trajectory's spatial units for an event to be accepted (default 0.75).
    save_path : str or None
        Directory; if given, pickles detected events as replay_type_N.pkl per type.
    verbose : bool
        Print event counts per trajectory type.

    Returns
    -------
    detected_per_type : list of dicts, length = len(target_trajectory)
        Each dict maps (lb, ub) → fitted_path (np.ndarray).
    """
    import pickle

    CA3_PF = load_PF_starts()
    CA3_PC_ID_list = generate_place_cell_ID_list(
        np.array(list(CA3_PF.keys()), dtype=int),
        np.array(list(CA3_PF.values()))
    )
    
    # Rate binning parameters — match the original rate array's time resolution
    len_rate  = len(rate)
    bin_dur_s = (rest_time / len_rate) / 1000.0          # duration of each bin [s]
    rate_bins = np.linspace(0, rest_time, len_rate + 1)  # bin edges [ms]

    detected_per_type = []
    for rep_type, targ_traj in enumerate(target_trajectory):
        detected = {}

        # Isolate neurons whose place fields lie along this trajectory
        target_idx = reorder_neuron_idx(CA3_PC_ID_list, CA3_PF, targ_traj)
        mask = np.isin(spk_neurons, target_idx)

        n_target = int(mask.sum())
        if n_target == 0:
            detected_per_type.append(detected)
            if verbose:
                print(f"Trajectory type {rep_type}: 0 spikes from trajectory neurons")
            continue

        # Recompute population rate from target neurons only [Hz]
        counts, _ = np.histogram(spk_time[mask], bins=rate_bins)
        rate_target = counts / (n_target * bin_dur_s)
        # print(len(rate))
        # print(len(rate_target))
        # print(rate_target.max())
        result = analyse_replay(spk_time[mask], spk_neurons[mask], rate_target*20, verbose=False)
        if not isinstance(result[1], dict):
            detected_per_type.append(detected)
            continue
        _, replay_results = result[0], result[1]

        # Spatial units covered by this trajectory
        target_units = np.array(
            [u for ss in targ_traj for u in range(ss * unit_gran, (ss + 1) * unit_gran)]
        )

        for tt, res in replay_results.items():
            if res['significance'] != 1:
                continue
            path = res['fitted_path'].copy()
            # Mark positions outside this trajectory's spatial range as invalid
            invalid = ~np.isin(np.round(path).astype(int), target_units)
            path[invalid] = -10
            if (path >= 0).sum() / len(path) >= coverage_thr:
                detected[tt] = path

        if verbose:
            print(f"Trajectory type {rep_type}: {len(detected)} replay events detected")

        if save_path:
            with open(os.path.join(save_path, f"replay_type_{rep_type}.pkl"), 'wb') as fp:
                pickle.dump(detected, fp)

        detected_per_type.append(detected)

    return detected_per_type

def compute_transition_matrix(num_states, value_states, possible_actions, end_state=None, softmax_coeff=1):
    """
    Build a softmax-weighted Markov transition matrix from state value estimates.

    For each non-terminal state, the probability of transitioning to each
    reachable next state is proportional to softmax(softmax_coeff * value).
    Terminal states (in end_state) self-loop with probability 1.

    Parameters
    ----------
    num_states : int
        Total number of discrete maze states.
    value_states : np.ndarray, shape (num_states,)
        Estimated value of each state (e.g., from the predictive map).
    possible_actions : np.ndarray
        Array of action vectors; each row is a (row, col) displacement.
    end_state : list of int or None
        State indices treated as absorbing terminal states.
    softmax_coeff : float
        Temperature coefficient for the softmax (higher = greedier policy).

    Returns
    -------
    transition_matrix : np.ndarray, shape (num_states, num_states)
        Row-stochastic transition probability matrix.
    value_matrix : np.ndarray, shape (num_states, num_states)
        State values of reachable next states (0 for unreachable transitions).
    """
    transition_matrix = np.zeros((num_states, num_states))
    value_matrix = np.zeros((num_states, num_states))

    for state_ID in range(num_states):
            next_state_ID = list()
            movement_taken = list()
            if state_ID in end_state:
                transition_matrix[state_ID,state_ID] = 1
                value_matrix[state_ID,state_ID] = np.inf
                continue

            for action_ID, action in enumerate(possible_actions):
                try:
                    next_state_ID.append(retreive_ID_from_position(state_position[state_ID]+action)[0])
                    movement_taken.append(action_ID)
                except: pass
            values_action = value_states[next_state_ID]
            probabilities = softmax(softmax_coeff*values_action)
            transition_matrix[state_ID,next_state_ID] = probabilities
            value_matrix[state_ID,next_state_ID] = values_action

    return transition_matrix, value_matrix

def sample_spatial_points(unit_gran):
    """
    Generate a uniform grid of 2-D spatial sample points along the linear track.

    For the linear reward track (row = 0), points are sampled at `unit_gran`
    sub-unit intervals along the column axis.

    Parameters
    ----------
    unit_gran : int
        Number of sample points per maze unit.

    Returns
    -------
    spatial_points : np.ndarray, shape (unit_gran * num_state_total, 2)
        Array of (row, col) coordinates covering the entire track.
    """
    col = np.linspace(0,num_state_total-1.0/unit_gran,unit_gran*num_state_total).reshape(-1, 1)
    row = np.zeros((len(col),1))
    spatial_points = np.hstack([row, col])
    return spatial_points

def load_tuning_curves(spatial_points):
    """
    Loads in tau_i(x) tuning curves (used for generating 'teaching' spike train, see `poisson_proc.py`)
    (Can handle multiple place fields in different environments)
    :param pklf_name: see `load_PF_starts`
    :param spatial_points: spatial coordinates to evaluate the tuning curves
    :return: tuning_curves: dict of tuning curves {neuronID: tuning curve}
    """

    place_fields = load_PF_starts()
    tuning_curves = {}
    for neuron_id, phi_start in place_fields.items():
        if type(phi_start) != list:
            tuning_curves[neuron_id] = get_tuning_curve(spatial_points, phi_start)
        else:  # multiple envs.
            tuning_curves_ = np.zeros((len(phi_start), len(spatial_points)))
            for i, phi_start_ in enumerate(phi_start):
                tuning_curves_[i, :] = get_tuning_curve(spatial_points, phi_start_)
            tuning_curve = np.sum(tuning_curves_, axis=0)
            tuning_curve[np.where(tuning_curve > 1.)] = 1.
            tuning_curves[neuron_id] = tuning_curve

    return tuning_curves

from tqdm import tqdm
def analyse_replay(spike_times, spiking_neurons, rate, len_sim=rest_time, spatial_points=sample_spatial_points(4), delta_t=10, N=100, t_incr=10, verbose=True):

    if len(spike_times) > 0:  # check if there is any activity
        slice_idx = slice_high_activity(rate, th=1.25, min_len=150, len_sim=len_sim)
        
        if slice_idx:
            tuning_curves = load_tuning_curves(spatial_points)
            sign_replays, replay_results = [], {}

            for bounds in tqdm(slice_idx, desc="Detecting replay"):  # iterate through sustained high activity periods
                lb, ub = bounds[0], bounds[1]
                idx = np.where((lb <= spike_times) & (spike_times < ub))
                bin_spike_counts = extract_binspikecount(lb, ub, delta_t, t_incr, spike_times[idx], spiking_neurons[idx],
                                                        tuning_curves)
                # decode place of the animal and try to fit path
                X_posterior = calc_posterior(bin_spike_counts, tuning_curves, delta_t)
                R, fitted_path, _ = fit_trajectory(X_posterior)
                sign, shuffled_Rs = test_significance(bin_spike_counts, tuning_curves, delta_t, R, N)
                sign_replays.append(sign)
                replay_results[bounds] = {"X_posterior": X_posterior, "fitted_path": fitted_path,
                                "R": R, "shuffled_Rs": shuffled_Rs, "significance": sign}
            significance = 1 if not np.isnan(sign_replays).all() else np.nan
        else:
            significance, replay_results = np.nan, {}

        if verbose:
            if not np.isnan(significance):
                print("Replay detected!")
            else:
                print("No replay detected...")

        return [significance, replay_results]
    
    else:
        if verbose:
            print("No activity!")
        return [np.nan for _ in range(20)]

def sample_spatial_points(unit_gran):
    """
    Generate a uniform grid of 2-D spatial sample points along the linear shock track.

    Parameters
    ----------
    unit_gran : int
        Number of sample points per maze unit.

    Returns
    -------
    spatial_points : np.ndarray, shape (unit_gran * num_state_total, 2)
        Array of (row=0, col) coordinates covering the entire track.
    """
    col = np.linspace(0,num_state_total-1.0/unit_gran,unit_gran*num_state_total).reshape(-1, 1)
    row = np.zeros((len(col),1))
    spatial_points = np.hstack([row, col])
    return spatial_points

# Adopted and revised from Ecker et al., 2022, ELife (https://github.com/KaliLab/ca3net).
# Original: assigned 1-D place field centres on a linear track.
# Revised: extended to 2-D track geometry; neurons are partitioned between row/column segments.
def sample_place_cells(n_neurons, place_cell_ratio, seed=11111):
    """
    Randomly assign place field centres to CA3 neurons on the linear shock track.

    Parameters
    ----------
    n_neurons : int
        Total number of CA3 neurons (must be >= 1000).
    place_cell_ratio : float
        Fraction of neurons assigned a place field.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    place_fields : dict {neuron_id: np.ndarray([row, col])}
    place_cells : np.ndarray of int
    phi_mid : np.ndarray, shape (n_place_cells, 2)
    """
    assert n_neurons >= 1000, "The assumptions made during the setup hold only for a reasonably big group of neurons"

    print("Generating place fields for %d neurons..."%n_neurons)
    neuronIDs = np.arange(0, n_neurons)
    # generate random neuronIDs being place cells and starting points for place fields
    
    np.random.seed(seed)
    p = np.ones(n_neurons)*1./n_neurons
    place_cells = np.sort(np.random.choice(neuronIDs, int(n_neurons*place_cell_ratio), p=p, replace=False), kind="mergsort")
    
    # Vertical part of the maze
    n_neurons_row = int(n_neurons*place_cell_ratio*(num_state_row/num_state_total))  # number of neurons in the vertical part of the maze
    phi_mid_row = np.sort(np.random.rand(n_neurons_row), kind="mergesort")[::-1] # sort in descending order
    phi_mid_row *= num_state_row  # [unitless]
    phi_mid_row += 0.5  # [unitless]

    # Convert phi_mid_row to a row of 2D array, set the column to the horizontal middle of the track
    phi_mid_row = phi_mid_row.reshape(-1, 1)
    phi_mid_row = np.hstack((phi_mid_row,np.random.uniform((num_state_col-1)/2-0.5,(num_state_col-1)/2+0.5,size=n_neurons_row).reshape(-1, 1)))

    # Horizontal part of the maze
    n_neurons_col = int(n_neurons*place_cell_ratio) - n_neurons_row   # number of neurons in the horizontal part of the maze
    phi_mid_col = np.sort(np.random.rand(n_neurons_col), kind="mergesort")
    phi_mid_col *= num_state_col  # [unitless]

    # Convert phi_mid_col to a column of 2D array, set the row to 0-1
    phi_mid_col = phi_mid_col.reshape(1, -1)
    phi_mid_col = np.vstack((np.random.uniform(-0.5,0.5,size=n_neurons_col), phi_mid_col)).T
    phi_mid = np.vstack((phi_mid_row, phi_mid_col))

    place_fields = {neuron_id:phi_mid[i] for i, neuron_id in enumerate(place_cells)}
    save_place_fields(place_fields,pklf_name)

    return place_fields, place_cells, phi_mid

def generate_place_field(initial_seed, num_neurons):
    """
    Wrapper: generate and save CA3 place fields for the linear shock task.

    Parameters
    ----------
    initial_seed : int
        Seed for reproducibility.
    num_neurons : int
        Total number of CA3 neurons.

    Returns
    -------
    place_fields : dict, place_cell_ID : np.ndarray, place_cell_ID_list : list
    """
    np.random.seed(initial_seed)

    place_fields, place_cell_ID, phi_mid_array = sample_place_cells(num_neurons,place_cell_ratio,initial_seed)
    place_cell_ID_list = generate_place_cell_ID_list(place_cell_ID, phi_mid_array)

    return place_fields, place_cell_ID, place_cell_ID_list

def generate_place_cell_ID_list(place_cell_ID, phi_mid_array):
    """
    Group CA3 place cell IDs by which maze state their field centre falls in.

    Parameters
    ----------
    place_cell_ID : np.ndarray of int
    phi_mid_array : np.ndarray, shape (n_place_cells, 2)

    Returns
    -------
    place_cell_ID_list : list of np.ndarray
        place_cell_ID_list[state] = neuron IDs whose field is within ±0.5 units
        of state_position[state].
    """
    place_cell_ID_list = []
    for ID in range(state_position.shape[0]):
        indices = np.where((phi_mid_array[:,0] >= state_position[ID,0]-0.5) & (phi_mid_array[:,0] < state_position[ID,0]+0.5)\
                            & (phi_mid_array[:,1] >= state_position[ID,1]-0.5) & (phi_mid_array[:,1] < state_position[ID,1]+0.5))[0]
        place_cell_ID_list.append(place_cell_ID[indices])
        del indices
    return place_cell_ID_list

def presence_update(current_unit_ID, lap, verbose=False):
    """
    Build the feature presence vector f_presence for the linear shock task.

    The shock cue (feature index 1) becomes active when the animal is at the
    shock location (state 7) during laps in cue_lap.

    Parameters
    ----------
    current_unit_ID : int
        Current maze state index.
    lap : int
        Current lap number.
    verbose : bool
        Print active features if True.

    Returns
    -------
    f_presence : np.ndarray, shape (num_features,)
    """
    f_presence = np.zeros((num_features), dtype=float)
    f_presence[2+current_unit_ID] = 1

    feature_case = -1
    for cc in range(len(cue_lap)):
        if lap in cue_lap[cc]: feature_case = cc; break

    if feature_case>=0:
        if verbose: print("Currently in cue position case #%d"%feature_case)
        for f_idx in range(num_features):
            if current_unit_ID in feature_unit_ID[feature_case][f_idx]:
                f_presence[f_idx] = 1
                if verbose: print("Feature %d present at position %d"%(f_idx,current_unit_ID))
    else:
        f_presence[:2] = 0
        if verbose: print("No affective feats present at position %d"%(current_unit_ID))

    return f_presence

def reorder_neuron_idx(place_cell_ID_list, place_fields, reordered_unit_ID_list, align_dir=None, include_cue=False):
    """
    Sort CA3 neuron indices by place field position along the traversal direction.

    See linear_reward_functions.reorder_neuron_idx for full documentation.
    Identical logic; present here so the module is self-contained.
    """
    reordered_idx = list()
    for unit_idx in range(len(reordered_unit_ID_list)):
        unit = reordered_unit_ID_list[unit_idx]
        if (unit==0)|(unit==num_state_col)|(unit==num_state_row+int(num_state_col/2))|(unit_idx == len(reordered_unit_ID_list)-1):
            if align_dir is None: pass
            align_dir = state_position[unit]-state_position[reordered_unit_ID_list[unit_idx-1]]
        else: align_dir = state_position[reordered_unit_ID_list[unit_idx+1]]-state_position[unit]

        PF_arr = np.array([place_fields.get(ID) for ID in place_cell_ID_list[unit]])
        if len(PF_arr)==0: temp_arr = []
        else:
            nz_dir = np.nonzero(align_dir)[0][0]

            sorted_idx = np.argsort(PF_arr[:,nz_dir])
            sorted_idx = sorted_idx[::-1] if (align_dir[nz_dir] < 0) else sorted_idx

            temp_arr = place_cell_ID_list[unit][sorted_idx]
        reordered_idx = reordered_idx + list(temp_arr)

    if include_cue:
        cue_cell_ID = list(np.setdiff1d(np.arange(num_CA3_neurons),list(place_fields.keys())))
        
        temp_arr = np.zeros(num_CA3_neurons)
        
        indices = np.random.permutation(num_CA3_neurons)
        temp_arr[np.sort(indices[:len(reordered_idx)])] = reordered_idx
        if len(cue_cell_ID) > 0: temp_arr[indices[-len(cue_cell_ID):]] = cue_cell_ID

        reordered_idx = temp_arr

    return np.array(reordered_idx,dtype=int)

# Adopted and revised from Ecker et al., 2022, ELife (https://github.com/KaliLab/ca3net).
# Revised: extended to 2-D start/stop positions; calls the 2-D inhom_poisson below.
def generate_spike_byPlace(neuron_ids, place_fields, start_position, stop_position, t_max, mice_speed=v_mice, seed=11111):
    """
    Generate CA3 spike trains driven purely by place field tuning.

    See linear_reward_functions.generate_spike_byPlace for full documentation.
    """
    # generate spike trains
    spike_trains = []

    for neuron_id in neuron_ids:
        if neuron_id in place_fields:
            spike_train = inhom_poisson(infield_rate, start_position, stop_position, t_max, place_fields[neuron_id], seed, mice_speed)
        else:
            spike_train = hom_poisson(background_rate, int(500*t_max), t_max, seed)
        spike_trains.append(spike_train)
        seed += 1
    # if start_position > stop_position: spike_trains = list(reversed(spike_trains))
    spike_trains = refractoriness(spike_trains)

    return spike_trains

def generate_spike_byPlaceAndInput(neuron_ids, place_fields, start_position, stop_position, t_max, w, upstream_activity, mice_speed=v_mice, seed=11111):
    """
    Generate CA3 spike trains combining place field drive and recurrent synaptic input.

    See linear_reward_functions.generate_spike_byPlaceAndInput for full documentation.
    """
    # generate spike trains
    spike_trains = []
    for neuron_id in neuron_ids:
        rate_modulation = input_driven_rate(neuron_id, upstream_activity, w, rate_shift=rate_shift_CA3, rate_slope=rate_slope_CA3)
        if neuron_id in place_fields:
            spike_train_recurrent = hom_poisson(0.05*rate_modulation, t_max, seed)
            spike_train_place = inhom_poisson(infield_rate, start_position, stop_position, t_max, place_fields[neuron_id], seed, mice_speed)
            spike_train = np.sort(np.concatenate((spike_train_recurrent, spike_train_place), axis=0))
        else:
            spike_train = hom_poisson(background_rate, t_max, seed)
        spike_trains.append(spike_train)
        seed += 1

    spike_trains = refractoriness(spike_trains)

    return spike_trains

def retreive_ID_from_position(position):
    """
    Map a continuous 2-D position to the discrete maze state it falls in.

    Parameters
    ----------
    position : array-like, shape (2,)
        (row, col) coordinates of the animal.
    circular : bool
        Unused; kept for interface consistency with other task modules.

    Returns
    -------
    state_id : int, position : np.ndarray
    """
    match_x = np.where((position[0] >= state_position[:,0]-0.5) & (position[0] < state_position[:,0]+0.5))[0]
    match_y = np.where((position[1] >= state_position[:,1]-0.5) & (position[1] < state_position[:,1]+0.5))[0]
    if match_x.size == 0 or match_y.size == 0:
        raise ValueError("No match found for position: {}".format(position))
    else: return np.intersect1d(match_x, match_y)[0], position

# Adopted and revised from Ecker et al., 2022, ELife (https://github.com/KaliLab/ca3net).
# Original: computed 1-D Euclidean distance. Revised: 2-D; no circular wrap-around
# (the shock track is non-circular — the animal hits the wall at state 7 and turns back).
def calc_distance(position, target, axis=1):
    """
    Compute Euclidean distance between position(s) and a target.

    No circular wrap-around (unlike linear_reward_functions.calc_distance) since
    the shock track is not circular.

    Parameters
    ----------
    position : array-like, shape (2,) or (N, 2)
        Query position(s).
    target : array-like, shape (2,)
        Reference position (place field centre).
    axis : int
        Axis along which the absolute difference is computed (default 1).

    Returns
    -------
    np.ndarray, shape (N,)
    """
    position = np.atleast_2d(position)
    target = np.asarray(target)

    diffs = position - target
    diffs[:, axis] = np.abs(diffs[:, axis])

    # Euclidean norm of adjusted diffs
    return np.linalg.norm(diffs, axis=1)

# Adopted and revised from Ecker et al., 2022, ELife (https://github.com/KaliLab/ca3net).
# Revised: calls the 2-D calc_distance defined in this module.
def evaluate_theta_modulation(t, start_position, phi_mid, f_theta, phase_init):
    """
    Compute theta-band modulation for spike thinning.

    See linear_reward_functions.evaluate_theta_modulation for full documentation.
    """
    try: distance = calc_distance(start_position, phi_mid) #[unit]
    except: print(start_position, phi_mid)
    phase = 2*np.pi*(f_theta*t + phase_init)
    phase_shift = -2*np.pi*distance # [unit]
    return np.cos(phase - phase_shift)

def get_tuning_curve(spatial_points, phi_mid):
    """
    Calculates (not estimates) tuning curve (Gaussian function)
    :param spatial_points: spatial points along the track
    :param phi_mid: peak location of the place field
    :return: tau: tuning curve of the place cell
    """

    distance = calc_distance(spatial_points, phi_mid, axis=1) #[unit]
    tau = np.exp(-np.power(distance, 2)/(2*tuning_curve_std**2))

    return tau

# Adopted and revised from Ecker et al., 2022, ELife (https://github.com/KaliLab/ca3net).
# Revised: supports 2-D positions and direction vectors.
def evaluate_lambda_t(t, start_position, direction, phi_mid, mice_speed=v_mice, phase_init=0.0, theta_modulation=True):
    """
    Evaluate the time-varying Poisson rate lambda(t) for spike thinning.

    See linear_reward_functions.evaluate_lambda_t for full documentation.
    """
    x = [start_position + mice_speed*direction*ss for ss in t] # [unit]
    if len(x) == 0: return x

    tau_x = get_tuning_curve(x, phi_mid) # kernel-filtered x
    if theta_modulation: theta_mod = evaluate_theta_modulation(t, start_position, phi_mid, f_theta, phase_init)
    else: theta_mod = 1

    lambda_t = tau_x * theta_mod
    lambda_t[np.where(lambda_t < 0.0)] = 0.0

    return lambda_t

# Adopted and revised from Ecker et al., 2022, ELife (https://github.com/KaliLab/ca3net).
# Identical I/O logic; path updated to this task's directory.
def load_PF_starts(pklf_name=pklf_name):
    """
    Load pre-generated CA3 place field peak positions from disk.

    Parameters
    ----------
    pklf_name : str
        Path to the pickle file containing the place field dict.

    Returns
    -------
    place_fields : dict {neuron_id: np.ndarray([row, col])}
    """
    with open(pklf_name, "rb") as f:
        place_fields = pickle.load(f, encoding="latin1")

    return place_fields

def inhom_poisson(lambda_, start_position, stop_position, t_max, phi_mid, seed, mice_speed=v_mice):
    """
    Generate homogeneous spikes (batched) and thin them using the inhomogeneous rate.
    All heavy elementwise ops are done on GPU if available.
    """
    poisson_proc = hom_poisson(lambda_, t_max, seed)  # returns CPU NumPy

    if poisson_proc.size == 0:
        return poisson_proc

    # Your evaluate_lambda_t uses NumPy internally; compute then move to backend
    lambda_t_cpu = evaluate_lambda_t(poisson_proc, start_position,
                                     stop_position - start_position, phi_mid, mice_speed)

    if seed is not None:
        np.random.seed(seed)
    keep = lambda_t_cpu >= np.random.rand(poisson_proc.shape[0])

    kept = poisson_proc[keep]
    return kept

def _avg_rate(rate, bin_, len_sim=rest_time):
    """
    Averages rate (used also for bar plots)
    :param rate: np.array representing firing rates
    :param bin_: bin size
    :param zoomed: bool for zoomed in plots
    """

    t = np.linspace(0, len_sim, len(rate))
    t0 = 0
    t1 = np.arange(t0, len_sim, bin_)
    t2 = t1 + bin_
    avg_rate = np.zeros_like(t1, dtype=float)
    for i, (t1_, t2_) in enumerate(zip(t1, t2)):
        avg_ = np.mean(rate[np.where((t1_ <= t) & (t < t2_))])
        if avg_ != 0.:
            avg_rate[i] = avg_

    return avg_rate


def load_spike_trains(npzf_name):
    """
    Loads in spike trains and converts it to 2 np.arrays for Brian2's SpikeGeneratorGroup
    :param npzf_name: file name of saved spike trains
    :return spiking_neurons, spike_times: same spike trains converted into SpikeGeneratorGroup format
    """

    npz_f = np.load(npzf_name, allow_pickle=True)
    spike_trains = [npz_f[i] for i in npz_f]

    spiking_neurons = 0 * np.ones_like(spike_trains[0])
    spike_times = np.asarray(spike_trains[0])
    for neuron_id in range(1, num_CA3_neurons):
        tmp = neuron_id * np.ones_like(spike_trains[neuron_id])
        spiking_neurons = np.concatenate((spiking_neurons, tmp), axis=0)
        spike_times = np.concatenate((spike_times, np.asarray(spike_trains[neuron_id])), axis=0)

    return spiking_neurons, spike_times

