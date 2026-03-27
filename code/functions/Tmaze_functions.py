from global_variables import *
from Tmaze_variables import *
import numpy as np
from common_functions import *
from tqdm import tqdm

base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
data_path = os.path.join(base_path,"results/Tmaze")
pklf_name = os.path.join(data_path, "PF_peak_data.pkl")

def compute_transition_matrix(num_states, value_states, possible_actions, end_state=None, softmax_coeff=1):
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
    norm_pred = np.zeros_like(pred)
    pred_size = pred.max(0)-pred.min(0)
    norm_pred = (pred - pred.min(0)) / pred_size
    return norm_pred

def sample_spatial_points(unit_gran=4):
    col = (0.5+int((num_state_col)/2) * np.ones(unit_gran*(num_state_row))).reshape(-1, 1)
    col = np.vstack([col, np.linspace(1/unit_gran,num_state_col,unit_gran*int(num_state_col)).reshape(-1, 1)])

    row = np.linspace(num_state_row+1-+1/unit_gran,1,unit_gran*num_state_row).reshape(-1, 1)
    row = np.vstack([row, 0.5*np.ones(unit_gran*(num_state_col)).reshape(-1, 1)])

    spatial_points = np.hstack([row, col])
    return spatial_points

def analyse_replay(spike_times, spiking_neurons, rate, len_sim=rest_time, ordered_neuron_idx=None, spatial_points=sample_spatial_points(4), delta_t=10, N=100, t_incr=10, verbose=True):

    if len(spike_times) > 0:  # check if there is any activity

        slice_idx = slice_high_activity(rate, th=1.25, min_len=130, len_sim=len_sim)

        if slice_idx:
            tuning_curves = load_tuning_curves(spatial_points)

            if ordered_neuron_idx is not None:
                scrambled = np.random.permutation(np.setdiff1d(np.arange(num_CA1_neurons), ordered_neuron_idx))
                neuron_idx_concat = np.concatenate([ordered_neuron_idx,scrambled])
                tuning_curves = {ii: tuning_curves[key] for ii,key in enumerate(neuron_idx_concat)}
                spiking_neurons = np.array([np.where(neuron_idx_concat==neuron)[0][0] for neuron in spiking_neurons])
                
                # tuning_curves = {ii: tuning_curves[key] for ii,key in enumerate(ordered_neuron_idx)}
                # spiking_neurons = np.array([np.where(ordered_neuron_idx==neuron)[0][0] for neuron in spiking_neurons])

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

def analyse_replay_type(spk_time, spk_neurons, rate, replay_trajectory=replay_trajectory, save_path=None, verbose=True):
    CA3_PF = load_PF_starts()
    CA3_PC_ID_list = generate_place_cell_ID_list(np.array(list(CA3_PF.keys()),dtype=int), np.array(list(CA3_PF.values())))

    for rep_type, rep_traj in enumerate(replay_trajectory):
        detected_replay = {}
        target_idx = reorder_neuron_idx(CA3_PC_ID_list,CA3_PF,rep_traj)
        idx = np.where(np.isin(spk_neurons, target_idx))[0]

        _, replay_results = analyse_replay(spk_time[idx], spk_neurons[idx], rate, verbose=False)
        for tt in replay_results.keys():
            if replay_results[tt]['significance'] == 1:
                
                path = replay_results[(tt[0], tt[1])]['fitted_path']
                path[path<0] = unit_gran*num_state_total+path[path<0]
                target_path_units = []
                for ss in rep_traj: target_path_units += list(np.arange(ss*4,(ss+1)*4))
                target_path_units = np.array(target_path_units)
                path[np.where(~np.isin(np.round(path),target_path_units))[0]]=-10
                if len(np.where(path>=0)[0])/len(path) > 0.75:
                    detected_replay[tt] = path
                else: continue
        if verbose: print("Type %d replay: %d events detected!"%(rep_type,len(list(detected_replay.keys()))))
        if save_path is not None:
            import pickle
            with open(os.path.join(save_path,"replay_type_%d.pkl"%rep_type), 'wb') as fp:
                pickle.dump(detected_replay, fp)


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
    

def load_PF_starts(pklf_name=pklf_name):

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

def sample_place_cells(n_neurons, place_cell_ratio, seed=11111):
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
    
    np.random.seed(initial_seed)

    place_fields, place_cell_ID, phi_mid_array = sample_place_cells(num_neurons,place_cell_ratio,initial_seed)
    place_cell_ID_list = generate_place_cell_ID_list(place_cell_ID, phi_mid_array)

    return place_fields, place_cell_ID, place_cell_ID_list

def plot_Tmaze_heat(data,ax,colormap='RdBu_r',vmax=1.5):
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
    place_cell_ID_list = []
    for ID in range(state_position.shape[0]):
        indices = np.where((phi_mid_array[:,0] >= state_position[ID,0]-0.5) & (phi_mid_array[:,0] < state_position[ID,0]+0.5)\
                            & (phi_mid_array[:,1] >= state_position[ID,1]-0.5) & (phi_mid_array[:,1] < state_position[ID,1]+0.5))[0]
        place_cell_ID_list.append(place_cell_ID[indices])
        del indices
    return place_cell_ID_list

def presence_update(current_unit_ID, lap, verbose=False):
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

def generate_spike_byPlace(neuron_ids, place_fields, start_position, stop_position, t_max, mice_speed=v_mice, seed=11111):

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
    match_x = np.where((position[0] >= state_position[:,0]-0.5) & (position[0] < state_position[:,0]+0.5))[0]
    match_y = np.where((position[1] >= state_position[:,1]-0.5) & (position[1] < state_position[:,1]+0.5))[0]
    if match_x.size == 0 or match_y.size == 0:
        raise ValueError("No match found for position: {}".format(position))
    else: return np.intersect1d(match_x, match_y)[0], position

def calc_distance(position, target, axis=1):
    
    position = np.atleast_2d(position)
    target = np.asarray(target)

    diffs = position - target
    diffs[:, axis] = np.abs(diffs[:, axis])
    
    # Euclidean norm of adjusted diffs
    return np.linalg.norm(diffs, axis=1)

def evaluate_theta_modulation(t, start_position, phi_mid, f_theta, phase_init):
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

def evaluate_lambda_t(t, start_position, direction, phi_mid, mice_speed=v_mice, phase_init=0.0, theta_modulation=True):
    x = [start_position + mice_speed*direction*ss for ss in t] # [unit]
    if len(x) == 0: return x
    
    tau_x = get_tuning_curve(x, phi_mid) # kernel-filtered x
    if theta_modulation: theta_mod = evaluate_theta_modulation(t, start_position, phi_mid, f_theta, phase_init)
    else: theta_mod = 1
   
    lambda_t = tau_x * theta_mod
    lambda_t[np.where(lambda_t < 0.0)] = 0.0

    return lambda_t

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
