"""
Tmaze_functions.py
==================
Task-specific functions for the T-maze environment.

This module provides the same interface as linear_reward_functions.py and
linear_shock_functions.py so that run_online.py and run_offline.py can import
any of the three interchangeably based on the `mode` argument.

Additional T-maze–specific functions not present in the other task modules:
  analyse_replay_type   — classifies detected SWR events by which canonical
                          trajectory (left arm, right arm, full loop) they match.
  plot_Tmaze_heat       — visualises a scalar per-state value on the T-maze grid.
  compute_transition_matrix — builds a softmax-weighted Markov transition
                          matrix from state values (used for behavioural analysis).
  pred_norm             — column-wise min-max normalisation of a prediction array.
  build_potentiated_weight_matrix — selectively scales the recurrent CA3–CA3
                          synapses whose pre- and post-synaptic neurons both have
                          place fields in a chosen set of states.
  simulate_choices      — rolls out Markov-chain trajectories per trial and
                          tallies which outcome state was reached.

Attribution guide (same convention as linear_reward_functions.py):
  sample_spatial_points       — [NEW] generates a 2-D grid covering the T-maze shape.
  sample_place_cells          — [REVISED] same logic as linear_reward version but
                                 adapted for the T-maze row/column geometry.
  generate_place_field        — [NEW]
  generate_place_cell_ID_list — [NEW]
  presence_update             — [NEW]
  reorder_neuron_idx          — [NEW]
  generate_spike_byPlace      — [REVISED]
  generate_spike_byPlaceAndInput — [NEW]
  retreive_ID_from_position   — [NEW]
  calc_distance               — [REVISED] no wrap-around (non-circular T-maze).
  analyse_replay              — [REVISED] supports ordered_neuron_idx remapping
                                 for sorting CA1 neurons by place field position.
  analyse_replay_type         — [NEW]
  compute_transition_matrix   — [NEW]
  pred_norm                   — [NEW]
  plot_Tmaze_heat             — [NEW]
  build_potentiated_weight_matrix — [NEW]
  load_PF_starts / load_tuning_curves — [REVISED]
  evaluate_theta_modulation / get_tuning_curve / evaluate_lambda_t / inhom_poisson
                              — [REVISED]
  _avg_rate / load_spike_trains — [COPIED from Ecker et al.]

Analysis helper sections (numbering continues from common_functions.py; all [NEW],
moved here out of FigS6_Carey_replay.ipynb):
  15. Per-region replay burst detection
        state_color                — colour-codes a state by arm membership.
        get_region_neuron_ids      — CA3 neuron IDs with place fields in given states.
        region_population_rate     — binned population rate from a region's neurons only.
        smooth_rate                — Gaussian smoothing of a rate trace.
        detect_burst_intervals     — contiguous above-threshold burst intervals.
        classify_arm_direction     — sequential-structure test (spike time vs. PF
                                      position) marking a burst as a valid replay.
        count_region_replay_events — per-region burst detection plus validity check.
        count_replay_by_region     — runs the above for every region in region_defs.
        merge_close_events         — merges intervals separated by < merge_gap_ms.
        count_total_replay_events  — merges valid events across regions so a replay
                                      sweeping several regions is counted once.
  16. Cached network-file lookup
        lap_file                   — path to a trial's online-learning lap_N.npz.
        replay_activity_file       — path to a trial's offline replay activity file.
        has_online_network         — True if every trial has its online lap file.
        has_offline_network        — True if every trial has its offline replay file.
  17. Behavioural choice simulation
        simulate_choices           — Markov-chain rollouts per trial; returns the
                                      outcome fractions and termination steps.

Note: inhom_poisson in this file uses to_xp / to_cpu helpers for optional GPU
acceleration (via CuPy), unlike the CPU-only version in linear_reward_functions.py.
"""

from global_variables import *
from Tmaze_variables import *
import numpy as np
from common_functions import *
from tqdm import tqdm

base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
data_path = os.path.join(base_path,"results/Tmaze")
pklf_name = os.path.join(data_path, "PF_peak_data.pkl")

# [NEW] — builds a softmax-weighted Markov transition matrix from state value estimates.
# Used in downstream behavioural analysis notebooks; not part of the simulation itself.
def compute_transition_matrix(num_states, value_states, possible_actions, end_state=None, softmax_coeff=1):
    """
    Build a softmax-weighted Markov transition matrix from state value estimates.

    For each non-terminal state, transition probabilities to reachable next states
    are computed as softmax(softmax_coeff * value).  Used for downstream
    behavioural analysis (e.g., computing expected arm-choice probabilities).

    Parameters
    ----------
    num_states : int
        Total number of discrete T-maze states.
    value_states : np.ndarray, shape (num_states,)
        Estimated value (e.g., prediction signal) for each state.
    possible_actions : np.ndarray
        Array of action displacement vectors.
    end_state : list of int or None
        Absorbing terminal states (self-loop, value = inf).
    softmax_coeff : float
        Temperature for softmax (higher = more greedy policy).

    Returns
    -------
    transition_matrix : np.ndarray, shape (num_states, num_states)
    value_matrix : np.ndarray, shape (num_states, num_states)
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

def pred_norm(pred):
    """
    Min-max normalise a prediction array column-wise to [0, 1].

    Parameters
    ----------
    pred : np.ndarray, shape (N, num_features)
        Raw prediction values.

    Returns
    -------
    norm_pred : np.ndarray, shape (N, num_features)
        Column-normalised values in [0, 1].
    """
    norm_pred = np.zeros_like(pred)
    pred_size = pred.max(0)-pred.min(0)
    norm_pred = (pred - pred.min(0)) / pred_size
    return norm_pred

def sample_spatial_points(unit_gran=4):
    """
    Generate a 2-D grid of spatial sample points covering the T-maze geometry.

    The grid covers the vertical stem (num_state_row states) and the horizontal
    arm (num_state_col states) separately, with `unit_gran` sub-samples per state.

    Parameters
    ----------
    unit_gran : int
        Number of sample points per maze state (default 4).

    Returns
    -------
    spatial_points : np.ndarray, shape ((num_state_row + num_state_col) * unit_gran, 2)
        Array of (row, col) coordinates in T-maze space.
    """
    col = (0.5+int((num_state_col)/2) * np.ones(unit_gran*(num_state_row))).reshape(-1, 1)
    col = np.vstack([col, np.linspace(1/unit_gran,num_state_col,unit_gran*int(num_state_col)).reshape(-1, 1)])

    row = np.linspace(num_state_row+1-+1/unit_gran,1,unit_gran*num_state_row).reshape(-1, 1)
    row = np.vstack([row, 0.5*np.ones(unit_gran*(num_state_col)).reshape(-1, 1)])

    spatial_points = np.hstack([row, col])
    return spatial_points

# Adopted and revised from Ecker et al., 2022, ELife (https://github.com/KaliLab/ca3net).
# Original: detected replay on a linear track using 1-D tuning curves.
# Revised: supports 2-D T-maze tuning curves; added optional ordered_neuron_idx
# remapping for sorting CA1 neurons by place field position along the path.
def analyse_replay(spike_times, spiking_neurons, rate, len_sim=rest_time, ordered_neuron_idx=None, spatial_points=sample_spatial_points(4), delta_t=10, N=100, t_incr=10, verbose=True):
    """
    Detect and decode significant replay events in offline spike data.

    Identifies periods of high population activity (> 1.25x baseline),
    applies Bayesian position decoding within each window, fits a linear
    trajectory to the posterior, and tests significance against shuffled controls.

    Parameters
    ----------
    spike_times : np.ndarray
        Spike event times [ms].
    spiking_neurons : np.ndarray of int
        Neuron index for each spike.
    rate : np.ndarray
        Population firing rate time series.
    len_sim : float
        Total simulation duration [ms].
    ordered_neuron_idx : np.ndarray or None
        If provided, remaps neuron indices so that place-field-sorted neurons
        appear first (enables ordered raster plots and cleaner decoding).
    spatial_points : np.ndarray, shape (N_pos, 2)
        Spatial positions at which tuning curves are evaluated.
    delta_t : int
        Bin size for spike count windowing [ms].
    N : int
        Number of shuffles for significance testing.
    t_incr : int
        Step size for sliding the decoding window [ms].
    verbose : bool
        Print detection results.

    Returns
    -------
    list : [significance (1 or nan), replay_results dict]
    """
    if len(spike_times) > 0:  # check if there is any activity

        slice_idx = slice_high_activity(rate, th=1.25, min_len=180, len_sim=len_sim)
        print(f"Detected high activity slices: {len(slice_idx)}")
        if slice_idx:
            tuning_curves = load_tuning_curves(spatial_points)

            if ordered_neuron_idx is not None:
                scrambled = np.random.permutation(np.setdiff1d(np.arange(num_CA1_neurons), ordered_neuron_idx))
                neuron_idx_concat = np.concatenate([ordered_neuron_idx,scrambled])
                tuning_curves = {ii: tuning_curves[key] for ii,key in enumerate(neuron_idx_concat)}
                spiking_neurons = np.array([np.where(neuron_idx_concat==neuron)[0][0] for neuron in spiking_neurons])

            sign_replays, replay_results = [], {}
            for bounds in slice_idx:  # iterate through sustained high activity periods
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
            significance = np.nan; replay_results = {}

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

# [NEW] — T-maze-specific function.
# Classifies each detected replay event into one of the canonical trajectory types
# defined in replay_trajectory (Tmaze_variables.py).
def analyse_replay_type(spk_time, spk_neurons, rate, target_trajectory=replay_trajectory,
                        coverage_thr=0.75, save_path=None, verbose=True):
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
          [0] shortcut        [3,4,5,6,7,8,9]
          [1] left from stem  [0,1,2,6,5,4,3]
          [2] right from stem [0,1,2,6,7,8,9]
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

        if save_path is not None:
            with open(os.path.join(save_path, f"replay_type_{rep_type}.pkl"), 'wb') as fp:
                pickle.dump(detected, fp)

        detected_per_type.append(detected)

    return detected_per_type


# def replay_Tmaze(spike_times, spiking_neurons, slice_idx, spatial_points=sample_spatial_points(4), ordered_neuron_idx, activity_arr, pklf_name, N, delta_t=10, t_incr=10):

#     if slice_idx:

#         tuning_curves = load_tuning_curves(pklf_name, spatial_points)

#         if ordered_neuron_idx is not None:
#             neuron_idx_concat = np.concatenate([ordered_neuron_idx,np.setdiff1d(np.arange(num_CA1_neurons), ordered_neuron_idx)])
#             tuning_curves = {ii: tuning_curves[key] for ii,key in enumerate(neuron_idx_concat)}
#             spiking_neurons = np.array([np.where(neuron_idx_concat==neuron)[0][0] for neuron in spiking_neurons])

#         sign_replays, replay_results = [], {}
#         for bounds in tqdm(slice_idx, desc="Detecting replay"):  # iterate through sustained high activity periods
#             lb, ub = bounds[0], bounds[1]
#             idx = np.where((lb <= spike_times) & (spike_times < ub))
#             bin_spike_counts = extract_binspikecount(lb, ub, delta_t, t_incr, spike_times[idx], spiking_neurons[idx],
#                                                      tuning_curves)
#             # decode place of the animal and try to fit path
#             X_posterior = calc_posterior(bin_spike_counts, tuning_curves, delta_t)
#             R, fitted_path, _ = fit_trajectory(X_posterior)
#             sign, shuffled_Rs = test_significance(bin_spike_counts, tuning_curves, delta_t, R, N)
#             sign_replays.append(sign)
#             replay_results[bounds] = {"X_posterior": X_posterior, "fitted_path": fitted_path,
#                                "R": R, "shuffled_Rs": shuffled_Rs, "significance": sign}
#         significance = 1 if not np.isnan(sign_replays).all() else np.nan
#         return significance, replay_results
#     else:
#         return np.nan, {}
    

# Adopted and revised from Ecker et al., 2022, ELife (https://github.com/KaliLab/ca3net).
# Identical I/O logic; path updated to this task's directory.
def load_PF_starts(pklf_name=pklf_name):
    """
    Load pre-generated CA3 place field peak positions from disk.

    Parameters
    ----------
    pklf_name : str
        Path to the pickle file.

    Returns
    -------
    place_fields : dict {neuron_id: np.ndarray([row, col])}
    """
    with open(pklf_name, "rb") as f:
        place_fields = pickle.load(f, encoding="latin1")

    return place_fields

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

# Adopted and revised from Ecker et al., 2022, ELife (https://github.com/KaliLab/ca3net).
# Original: assigned 1-D place field centres on a linear track.
# Revised: partitions neurons between the vertical stem and horizontal arm of
# the T-maze, placing each neuron's field centre within its segment.
def sample_place_cells(n_neurons, place_cell_ratio, seed=11111):
    """
    Randomly assign place field centres to CA3 neurons on the T-maze.

    Neurons are partitioned between the vertical stem (num_state_row states) and
    the horizontal arm (num_state_col states) in proportion to segment length.

    Parameters
    ----------
    n_neurons : int
        Total number of CA3 neurons (must be >= 1000).
    place_cell_ratio : float
        Fraction of neurons with a place field.
    seed : int
        Random seed.

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
    phi_mid_row += 1.0  # [unitless]

    # Convert phi_mid_row to a row of 2D array, set the column to the horizontal middle of the track
    phi_mid_row = phi_mid_row.reshape(-1, 1)
    phi_mid_row = np.hstack((phi_mid_row,np.random.uniform((num_state_col-1)/2,(num_state_col-1)/2+1.0,size=n_neurons_row).reshape(-1, 1)))

    # Horizontal part of the maze
    n_neurons_col = int(n_neurons*place_cell_ratio) - n_neurons_row   # number of neurons in the horizontal part of the maze
    phi_mid_col = np.sort(np.random.rand(n_neurons_col), kind="mergesort")
    phi_mid_col *= num_state_col  # [unitless]

    # Convert phi_mid_col to a column of 2D array, set the row to 0-1
    phi_mid_col = phi_mid_col.reshape(1, -1)
    phi_mid_col = np.vstack((np.random.uniform(0,1.0,size=n_neurons_col), phi_mid_col)).T
    
    phi_mid = np.vstack((phi_mid_row, phi_mid_col))

    place_fields = {neuron_id:phi_mid[i] for i, neuron_id in enumerate(place_cells)}
    save_place_fields(place_fields,pklf_name)

    return place_fields, place_cells, phi_mid

def generate_place_field(initial_seed, num_neurons):
    """
    Wrapper: generate and save CA3 place fields for the T-maze task.

    Parameters
    ----------
    initial_seed : int
    num_neurons : int

    Returns
    -------
    place_fields : dict, place_cell_ID : np.ndarray, place_cell_ID_list : list
    """
    np.random.seed(initial_seed)

    place_fields, place_cell_ID, phi_mid_array = sample_place_cells(num_neurons,place_cell_ratio,initial_seed)
    place_cell_ID_list = generate_place_cell_ID_list(place_cell_ID, phi_mid_array)

    return place_fields, place_cell_ID, place_cell_ID_list

# [NEW] — visualises per-state scalar data on the T-maze grid using imshow.
def plot_Tmaze_heat(data,ax,colormap='RdBu_r',vmax=1.5):
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    Tmaze_grid = np.array([state_position[ss]-[0.5,0.5] for ss in range(num_state_total)],dtype=int)
    data_arr = np.zeros((num_state_row+1,num_state_col))
    for ss in range(num_state_total):
        data_arr[Tmaze_grid[ss,0],Tmaze_grid[ss,1]] = data[ss]
    im = ax.imshow(data_arr, cmap=colormap, norm=norm)  # shading='auto' avoids shape mismatch

    for ss in range(num_state_total):
        x_mid = Tmaze_grid[ss,1]; y_mid = Tmaze_grid[ss,0]
        ax.hlines(y=[y_mid-0.5, y_mid+0.5], xmin=x_mid-0.5, xmax=x_mid+0.5, color='k', linewidth=1.5)
        ax.vlines(x=[x_mid-0.5, x_mid+0.5], ymin=y_mid-0.5, ymax=y_mid+0.5, color='k', linewidth=1.5)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)

    return im

def generate_place_cell_ID_list(place_cell_ID, phi_mid_array):
    """
    Group CA3 place cell IDs by which T-maze state their field centre falls in.

    Parameters
    ----------
    place_cell_ID : np.ndarray of int
    phi_mid_array : np.ndarray, shape (n_place_cells, 2)

    Returns
    -------
    place_cell_ID_list : list of np.ndarray, length num_state_total
    """
    place_cell_ID_list = []
    for ID in range(state_position.shape[0]):
        indices = np.where((phi_mid_array[:,0] >= state_position[ID,0]-0.5) & (phi_mid_array[:,0] < state_position[ID,0]+0.5)\
                            & (phi_mid_array[:,1] >= state_position[ID,1]-0.5) & (phi_mid_array[:,1] < state_position[ID,1]+0.5))[0]
        place_cell_ID_list.append(place_cell_ID[indices])
        del indices
    return place_cell_ID_list

# [NEW] — selectively potentiates recurrent CA3-CA3 synapses within a set of
# states (see run_offline.simulate_potentiated_replay / Supple_replay_Carey.ipynb).
def build_potentiated_weight_matrix(W_raw, PF_dict, potentiated_states, factor):
    """
    Scale recurrent CA3-CA3 synapses whose pre- AND post-synaptic neurons both
    have place fields in `potentiated_states` by `factor`; all other (non-zero)
    synapses are left unchanged.

    Parameters
    ----------
    W_raw : np.ndarray, shape (num_CA3, num_CA3)
        Learned recurrent weight matrix.
    PF_dict : dict {neuron_id: [row, col]}
        CA3 place field peak positions.
    potentiated_states : list of int
        0-indexed T-maze states whose mutual recurrent synapses are potentiated.
    factor : float
        Multiplicative scale applied to the selected synapses (1.15 = +15 %).

    Returns
    -------
    W_pot : np.ndarray
        Copy of `W_raw` with the selected synapses scaled.
    potentiated_mask : np.ndarray of bool, shape (num_CA3, num_CA3)
        True where a weight was scaled.
    n_potentiated : int
        Number of potentiated synapses.
    """
    place_cell_ID_list = generate_place_cell_ID_list(np.array(list(PF_dict.keys()), dtype=int),
                                                       np.array(list(PF_dict.values())))
    in_pot = np.zeros(W_raw.shape[0], dtype=bool)
    for s in potentiated_states:
        in_pot[place_cell_ID_list[s]] = True

    potentiated_mask = np.outer(in_pot, in_pot) & (W_raw > 0)
    W_pot = W_raw.copy()
    W_pot[potentiated_mask] *= factor
    return W_pot, potentiated_mask, int(potentiated_mask.sum())

def presence_update(current_unit_ID, lap, verbose=False):
    """
    Build the feature presence vector f_presence for the T-maze task.

    The two outcome features (left-arm and right-arm) become active when the
    animal reaches the corresponding arm endpoint.

    Parameters
    ----------
    current_unit_ID : int
        Current maze state index.
    lap : int
        Current lap number (determines which arm is visited from single_lap).
    verbose : bool
        Print active features if True.

    Returns
    -------
    f_presence : np.ndarray, shape (num_features,)
    """
    f_presence = np.zeros((num_features), dtype=float)
    f_presence[2+current_unit_ID] = 1

    for cc in range(len(cue_lap)):
        feature_case = -1
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

def reorder_neuron_idx(place_cell_ID_list, place_fields, reordered_unit_list, include_cue=False):
    """
    Sort CA3 neuron indices by place field position along a T-maze traversal path.

    For each state in `reordered_unit_list`, neurons are sorted along the local
    movement direction so that earlier-firing neurons appear first.

    Parameters
    ----------
    place_cell_ID_list : list of np.ndarray
        Neuron IDs per maze state (from generate_place_cell_ID_list).
    place_fields : dict {neuron_id: np.ndarray([row, col])}
    reordered_unit_list : list of int
        Ordered sequence of maze state IDs along the desired trajectory.
    include_cue : bool
        If True, append non-place-cell (cue) neurons interleaved at the end.

    Returns
    -------
    np.ndarray of int
        Neuron indices sorted by place field position along the trajectory.
    """
    reordered_idx = list()
    for unit_idx, unit in enumerate(reordered_unit_list):
        if unit_idx == len(reordered_unit_list)-1: align_dir = state_position[unit] - state_position[reordered_unit_list[unit_idx-1]]
        else: align_dir = state_position[reordered_unit_list[unit_idx+1]]-state_position[unit]

        PF_arr = np.array([place_fields.get(ID) for ID in place_cell_ID_list[unit]])
        if len(PF_arr)==0: temp_arr = []
        else:
            nz_dir = np.nonzero(align_dir)[0][0]

            sorted_idx = np.argsort(PF_arr[:,nz_dir])
            sorted_idx = sorted_idx[::-1] if (align_dir[nz_dir] < 0) else sorted_idx

            temp_arr = place_cell_ID_list[unit][sorted_idx]
        reordered_idx = reordered_idx + list(temp_arr)

    if include_cue:
        cell_portion = num_state_total/len(reordered_unit_list)
        cue_cell_ID = list(np.setdiff1d(np.arange(num_CA3_neurons),list(place_fields.keys())))
        
        temp_arr = np.zeros(int(num_CA3_neurons*cell_portion))
        
        indices = np.random.permutation(int(num_CA3_neurons*cell_portion))
        temp_arr[np.sort(indices[:len(reordered_idx)])] = reordered_idx
        if len(cue_cell_ID) > 0: temp_arr[indices[len(reordered_idx):]] = cue_cell_ID[:len(reordered_idx)+1]

        reordered_idx = temp_arr

    return np.array(reordered_idx,dtype=int)

# Adopted and revised from Ecker et al., 2022, ELife (https://github.com/KaliLab/ca3net).
# Revised: extended to 2-D T-maze positions; calls the 2-D inhom_poisson below.
def generate_spike_byPlace(neuron_ids, place_fields, start_position, stop_position, t_max, mice_speed=v_mice, seed=11111):
    """
    Generate CA3 spike trains driven purely by place field tuning (T-maze version).

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
    Generate CA3 spike trains combining place field drive and recurrent input (T-maze version).

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
    Map a continuous 2-D position to the discrete T-maze state it falls in.

    Parameters
    ----------
    position : array-like, shape (2,)
        (row, col) coordinates of the animal.

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
# (the T-maze has dead-end arms, not a circular topology).
def calc_distance(position, target, axis=1):
    """
    Compute Euclidean distance between position(s) and a target on the T-maze.

    No circular wrap-around (T-maze arms are dead-ends).

    Parameters
    ----------
    position : array-like, shape (2,) or (N, 2)
    target : array-like, shape (2,)
    axis : int
        Axis for absolute difference (default 1).

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
    Compute theta-band modulation for spike thinning (T-maze version).

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
# Revised: supports 2-D positions and direction vectors for T-maze navigation.
def evaluate_lambda_t(t, start_position, direction, phi_mid, mice_speed=v_mice, phase_init=0.0, theta_modulation=True):
    """
    Evaluate the time-varying Poisson rate lambda(t) for spike thinning (T-maze version).

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
# Original: used a fixed-size buffer and 1-D position. Revised: batched hom_poisson
# call; supports 2-D positions; optionally accelerated via CuPy (to_xp / to_cpu).
def inhom_poisson(lambda_, start_position, stop_position, t_max, phi_mid, seed, mice_speed=v_mice):
    """
    Generate an inhomogeneous Poisson spike train using thinning.

    Draws candidate times from a homogeneous process at rate `lambda_`, then
    retains each candidate with probability proportional to the place-field
    tuning curve (and optional theta modulation) evaluated at the animal's
    position at that moment.

    Parameters
    ----------
    lambda_ : float
        Upper bound on the firing rate [Hz] (peak in-field rate).
    start_position, stop_position : np.ndarray, shape (2,)
        Start and end 2-D positions for this time window [maze units].
    t_max : float
        Duration [s].
    phi_mid : np.ndarray, shape (2,)
        Place field centre [maze units].
    seed : int
        Random seed for thinning.
    mice_speed : float
        Running speed [maze units / ms].

    Returns
    -------
    np.ndarray
        Accepted spike times [s].
    """
    poisson_proc = hom_poisson(lambda_, t_max, seed)  # returns CPU NumPy

    if poisson_proc.size == 0:
        return poisson_proc

    # Your evaluate_lambda_t uses NumPy internally; compute then move to backend
    lambda_t_cpu = evaluate_lambda_t(poisson_proc, start_position,
                                     stop_position - start_position, phi_mid, mice_speed)
    lam_xp = to_xp(lambda_t_cpu)
    t_xp   = to_xp(poisson_proc)

    if seed is not None:
        xp.random.seed(seed)
    keep = lam_xp >= xp.random.rand(t_xp.shape[0])

    kept = t_xp[keep]
    return to_cpu(kept)

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

# =============================================================================
# 15. Per-region replay burst detection (Carey T-maze replay analysis)  [NEW]
# =============================================================================
# Moved from FigS6_Carey_replay.ipynb: detects replay-like population bursts
# separately within named T-maze regions (e.g. stem / left arm / right arm)
# and merges cross-region detections of the same physical event.

def state_color(state, left_states, right_states, left_color="steelblue", right_color="tomato", default_color="lightgrey"):
    """
    Color-code a T-maze state by arm membership (for replay/behaviour plots).

    Parameters
    ----------
    state : int
    left_states, right_states : list of int
    left_color, right_color, default_color : str

    Returns
    -------
    str
    """
    if state in left_states: return left_color
    if state in right_states: return right_color
    return default_color

def get_region_neuron_ids(states):
    """
    Union of CA3 neuron IDs whose place field lies in any of `states`.

    Parameters
    ----------
    PF_dict : dict {neuron_id: [row, col]}
    states : list of int

    Returns
    -------
    np.ndarray of int
    """
    PF_dict = load_PF_starts()
    place_cell_ID_list = generate_place_cell_ID_list(
        np.array(list(PF_dict.keys()), dtype=int), np.array(list(PF_dict.values())))
    ids = np.concatenate([place_cell_ID_list[s] for s in states])
    return np.unique(ids)

def region_population_rate(spike_t_ms, spike_nids, region_nids, duration_ms, bin_ms=1.0):
    """Binned, per-neuron population rate (Hz), built only from `region_nids` spikes."""
    bins         = np.arange(0, duration_ms + bin_ms, bin_ms)
    in_region    = np.isin(spike_nids, region_nids)
    spk_count, _ = np.histogram(spike_t_ms[in_region], bins=bins)
    rate_hz      = spk_count / (max(len(region_nids), 1) * (bin_ms / 1000))
    times_ms     = bins[:-1] + bin_ms / 2
    return rate_hz, times_ms

def smooth_rate(rate_arr, dt_ms, sigma_ms=20.0):
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(rate_arr, sigma=sigma_ms / dt_ms)

def detect_burst_intervals(rate_smooth, times_ms, threshold, min_dur_ms=70, min_gap_ms=50):
    """Return list of (t_start_ms, t_end_ms) for contiguous bursts above threshold."""
    above = rate_smooth > threshold
    if not above.any():
        return []
    changes = np.diff(above.astype(int))
    on_idx  = np.where(changes ==  1)[0] + 1
    off_idx = np.where(changes == -1)[0] + 1
    if above[0]:  on_idx  = np.concatenate([[0],            on_idx])
    if above[-1]: off_idx = np.concatenate([off_idx, [len(times_ms) - 1]])

    intervals = []
    for si, ei in zip(on_idx, off_idx):
        t0, t1 = times_ms[si], times_ms[ei]
        if t1 - t0 < min_dur_ms:
            continue
        if intervals and t0 - intervals[-1][1] < min_gap_ms:
            intervals[-1] = (intervals[-1][0], t1)   # merge
        else:
            intervals.append((t0, t1))
    return intervals

def classify_arm_direction(burst_spike_t, burst_pos, r_thresh=0.15, min_spikes=5):
    """
    Pearson r between spike time and PF position along the region's own axis
    (row for the stem, column for the arms).  A burst is a *valid replay
    event* for that region when |r| exceeds `r_thresh`, i.e. spikes progress
    systematically along the region rather than firing without sequential
    structure.
    """
    valid = burst_pos >= 0
    if valid.sum() < min_spikes:
        return False
    r = np.corrcoef(burst_spike_t[valid], burst_pos[valid])[0, 1]
    return (not np.isnan(r)) and (abs(r) > r_thresh)

def count_region_replay_events(spike_t_ms, spike_nids, states, axis,
                               duration_ms, bin_ms=1.0, sigma_ms=20,
                               thresh_factor=0.5, min_dur_ms=50, min_gap_ms=50,
                               r_thresh=0.1, min_spikes=5):
    """
    Isolate spikes from neurons whose place field lies in `states`, detect
    burst intervals in that subpopulation's own rate, and keep the bursts
    that pass the sequential-correlation check (valid replay events).

    Returns
    -------
    n_valid   : int, number of valid replay events
    n_bursts  : int, number of detected burst intervals (before validity check)
    intervals : list of (t0, t1) for the valid replay events
    """
    region_nids        = get_region_neuron_ids(states)
    rate_arr, times_ms = region_population_rate(spike_t_ms, spike_nids,
                                                 region_nids, duration_ms, bin_ms)
    # rate_sm   = smooth_rate(rate_arr, bin_ms, sigma_ms)
    # threshold = rate_sm.mean() + thresh_factor * rate_sm.std()
    # bursts    = detect_burst_intervals(rate_sm, times_ms, threshold, min_dur_ms, min_gap_ms)
    PF_dict = load_PF_starts()
    from common_functions import slice_high_activity
    threshold = rate_arr.mean() + thresh_factor * rate_arr.std()
    bursts = slice_high_activity(rate=rate_arr, th=threshold,
                                    min_len=min_dur_ms, len_sim=duration_ms)

    in_region  = np.isin(spike_nids, region_nids)
    region_t   = spike_t_ms[in_region]
    region_pos = np.array([PF_dict.get(int(n), [-1, -1])[axis] for n in spike_nids[in_region]])

    valid_intervals = []
    for t0, t1 in bursts:
        mask = (region_t >= t0) & (region_t <= t1)
        if mask.sum() < min_spikes:
            continue
        if classify_arm_direction(region_t[mask], region_pos[mask], r_thresh, min_spikes):
            valid_intervals.append((t0, t1))
    return len(valid_intervals), len(bursts), valid_intervals

def count_replay_by_region(spike_t_ms, spike_nids, duration_ms, region_defs, **kwargs):
    """
    Run count_region_replay_events separately for each region in `region_defs`.

    Parameters
    ----------
    region_defs : dict {region_name: {"states": [...], "axis": int}}
    """
    counts, n_bursts, intervals = {}, {}, {}
    for region, spec in region_defs.items():
        n_valid, nb, ivals = count_region_replay_events(
            spike_t_ms, spike_nids, spec["states"], spec["axis"],
            duration_ms, **kwargs
        )
        counts[region]    = n_valid
        n_bursts[region]  = nb
        intervals[region] = ivals
    return counts, n_bursts, intervals

def merge_close_events(intervals, merge_gap_ms=30):
    """
    Merge a list of (t0, t1) intervals so that any two whose gap (start of
    the later one minus end of the earlier one) is < merge_gap_ms collapse
    into a single event.
    """
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged  = [ordered[0]]
    for t0, t1 in ordered[1:]:
        last_t0, last_t1 = merged[-1]
        if t0 - last_t1 < merge_gap_ms:
            merged[-1] = (last_t0, max(last_t1, t1))
        else:
            merged.append((t0, t1))
    return merged

def count_total_replay_events(region_intervals, merge_gap_ms=30):
    """
    Combine the valid replay-event intervals detected separately across regions
    (e.g. stem, left arm, right arm).  A single physical replay that sweeps
    across regions (e.g. stem -> left arm) gets detected once per region, so
    events starting within `merge_gap_ms` of another region's event are merged
    and counted as ONE big replay event.

    Parameters
    ----------
    region_intervals : dict of {region: [(t0, t1), ...]}, e.g. the
                       `intervals` dict returned by count_replay_by_region.

    Returns
    -------
    n_total : int, number of merged replay events
    merged  : list of (t0, t1) for the merged events
    """
    all_intervals = []
    for ivals in region_intervals.values():
        all_intervals.extend(ivals)
    merged = merge_close_events(all_intervals, merge_gap_ms)
    return len(merged), merged

# =============================================================================
# 16. Cached network-file lookup (online vs. offline-consolidated)  [NEW]
# =============================================================================
# Moved from FigS6_Carey_replay.ipynb: locates per-trial online-learning
# ("lap_N.npz") and offline-consolidated ("CA1_activity_lap_N_replay_M.npz")
# output files under a given scenario data directory.

def lap_file(data_dir, folder_root, tr, target_lap):
    return os.path.join(data_dir, folder_root+str(tr), "lap_%d.npz" % target_lap)

def replay_activity_file(data_dir, folder_root, tr, target_lap, rest_time_ms):
    return os.path.join(data_dir, folder_root+str(tr), "activity",
                         "CA1_activity_lap_%d_replay_%d.npz" % (target_lap, rest_time_ms))

def has_online_network(data_dir, folder_root, trial_number, target_lap):
    return all(os.path.exists(lap_file(data_dir, folder_root, tr, target_lap)) for tr in range(trial_number))

def has_offline_network(data_dir, folder_root, trial_number, target_lap, rest_time_ms):
    return all(os.path.exists(replay_activity_file(data_dir, folder_root, tr, target_lap, rest_time_ms))
               for tr in range(trial_number))

# =============================================================================
# 17. Behavioral choice simulation across trials  [NEW]
# =============================================================================
# Moved from FigS6_Carey_replay.ipynb: rolls out `behavior_markov` many times
# per trial's transition matrix and tallies which outcome state was reached.

def simulate_choices(tm, trial_number, num_behav_trial, total_time, start_ID, end_state):
    """
    Roll out `num_behav_trial` Markov-chain trajectories per trial.

    Parameters
    ----------
    tm : np.ndarray, shape (trial_number, num_states, num_states)
        Per-trial transition matrices (see compute_transition_matrix).
    trial_number, num_behav_trial, total_time : int
    start_ID : int
    end_state : list of int, length 2
        [outcome_A_state, outcome_B_state].

    Returns
    -------
    goal_frac : np.ndarray, shape (trial_number, 3)
        Fraction of runs ending at [outcome_A, outcome_B, neither].
    goal_time : np.ndarray, shape (trial_number, num_behav_trial)
        Step at which each run terminated (nan if it timed out).
    """
    outcome_A, outcome_B = end_state
    goal_frac = np.zeros((trial_number, 3))
    goal_time = np.full((trial_number, num_behav_trial), np.nan)
    for tr in range(trial_number):
        counts = np.zeros(3)
        for bb in range(num_behav_trial):
            _, gt, gs = behavior_markov(tm[tr], total_time=total_time, start_state=start_ID, end_state=end_state)
            goal_time[tr, bb] = gt
            if gs == outcome_A: counts[0] += 1
            elif gs == outcome_B: counts[1] += 1
            else: counts[2] += 1
        goal_frac[tr] = counts / num_behav_trial
    return goal_frac, goal_time
