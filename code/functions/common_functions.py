import os, pickle
from global_variables import *
from copy import deepcopy

from scipy import signal
from scipy.special import softmax, comb, factorial
from scipy.optimize import curve_fit
from scipy.stats import t as student_t
from scipy.signal import convolve2d

import numpy as np

def init_weights(num_CA3_neurons, num_CA1_neurons, num_features):
    w_CA3_CA3 = np.random.normal(w_init,w_init*1e-2,size=(num_CA3_neurons, num_CA3_neurons))
    w_CA3_CA1 = np.random.normal(w_init,w_init*1e-2,size=(num_CA3_neurons, num_CA1_neurons))
    w_CA1_feat = np.zeros((num_CA1_neurons, num_features), dtype=float)

    connectivity_CA3_CA3 = (np.random.rand(num_CA3_neurons, num_CA3_neurons) < CA3_CA3_connection_prob) & (np.eye(num_CA3_neurons) == 0)
    connectivity_CA3_CA1 = (np.random.rand(num_CA3_neurons, num_CA1_neurons) < CA3_CA1_connection_prob)
    
    # Disconnect unconnected neurons
    w_CA3_CA3[connectivity_CA3_CA3==False] = 0
    w_CA3_CA1[connectivity_CA3_CA1==False] = 0
    return w_CA3_CA3, w_CA3_CA1, w_CA1_feat, connectivity_CA3_CA3, connectivity_CA3_CA1

def init_layervars(num_neurons):
    ET = np.zeros((num_neurons), dtype=float); PT = np.zeros((num_neurons), dtype=float)
    plateau_flag = np.zeros((num_neurons), dtype=bool); plateau_refractory = np.zeros((num_neurons), dtype=int)
    FR = np.zeros(num_neurons)
    return ET, PT, plateau_flag, plateau_refractory, FR

def input_driven_rate(neuron_ids, CA3_activity, w, rate_shift=rate_shift_CA1, rate_slope=rate_slope_CA1):
    # Ensure backend arrays                          # shape: (N_pre, N_post)

    scale = 0.02  # Avg input through a single synapse
    if isinstance(neuron_ids, list) or np.ndim(neuron_ids) == 1:
        # Vectorized gather of columns
        idx = np.asarray(neuron_ids)
        x = CA3_activity @ w[:, idx] * scale  # shape: (len(neuron_ids),) or (K, len(neuron_ids))
        act = max_input_FR * sigmoid(x, rate_shift, rate_slope)
        return act

    # Single neuron id
    nid = int(neuron_ids)
    x = CA3_activity @ w[:, nid] * scale
    act = max_input_FR * sigmoid(x, rate_shift, rate_slope)
    return act

def generate_spike_byInput(neuron_ids, t_max, w, upstream_activity, seed=11111):
    # generate spike trains
    spike_trains = []
    for neuron_id in neuron_ids:
        rate_modulation = input_driven_rate(neuron_id, upstream_activity, w)
        indiv_rate = np.maximum(rate_modulation,background_rate)
        spike_train = hom_poisson(indiv_rate, t_max, seed)
        spike_trains.append(spike_train)
        seed += 1

    spike_trains = refractoriness(spike_trains)

    return spike_trains

def add_spike_train(place_cell_ID, spike_trains, place_spikes):
    ii=0
    for IDs in place_cell_ID:
        spike_trains[IDs] = np.append(spike_trains[IDs],place_spikes[ii])
        spike_trains = refractoriness(spike_trains)
        ii+=1

    return spike_trains

def concat_spike_trains(spike_trains, num_neuron):
    spiking_neurons = 0 * np.ones_like(spike_trains[0])
    spike_times = np.asarray(spike_trains[0])
    for neuron_id in range(1, num_neuron):
        tmp = neuron_id * np.ones_like(spike_trains[neuron_id])
        spiking_neurons = np.concatenate((spiking_neurons, tmp), axis=0)
        spike_times = np.concatenate((spike_times, np.asarray(spike_trains[neuron_id])), axis=0)
    
    return spiking_neurons, spike_times


def find_PF_peak(act_array):
    # peak_idx = np.argmax(act_array,axis=0)
    # place_cell_idx = np.where((np.max(act_array,axis=0)>=place_thr_FR))[0]
    rev = act_array[::-1,:]
    last_occurrence_reversed_index = np.argmax(rev,axis=0)
    reordered_peak_index = rev.shape[0] - 1 - last_occurrence_reversed_index

    return reordered_peak_index

def plateau_probability_calc(input_FR, target_FR, base_prob, p_slope, min_prob=0):
    firing_prob = np.zeros(len(input_FR))
    for cell_id in range(len(input_FR)):
        layerwise_FR = np.average(input_FR)

        indiv_potentiation = np.minimum(1,(input_FR[cell_id]/infield_rate))
        # if indiv_potentiation > 1: indiv_potentiation = 1
        layewise_inhibition = (min_prob+((2*base_prob)*sigmoid((target_FR-layerwise_FR),0,p_slope)-base_prob)) if layerwise_FR < target_FR else min_prob

        firing_prob[cell_id] = indiv_potentiation*layewise_inhibition

    return firing_prob

def ET_update(t, spike_times, spiking_neurons, activity, ET=ET_amp):

    if spike_times.size != 0:
        spiking_neuron = spiking_neurons[spike_times == t]
        activity[spiking_neuron] += ET

    # We stop updating the activity if it is below the threshold, so that we can avoid hyperdepression
    activity[activity<act_thr] = 0
    # ET_flag = activity >= act_thr # AP_flag is True if the activity is above the threshold
    
    return activity

def plateau_update(firing_rate, activity, target_FR, plateau_flag, plateau_refractory, base_prob=base_prob_CA1, p_slope=firing_prob_slope_CA1, min_prob=0, PS=-1, seed=11111, PT=plateau_amp, verbose=False):
    
    no_prev_plateau_flag = (plateau_flag == 0)
    no_plateau_refractory = (plateau_refractory == 0)

    if PS == -1:
        prob_scale, targetFR_scale, plateau_scale = 1, 1, 1
    else:
        prob_scale = 1+5*PS
        targetFR_scale = 1+2*PS
        plateau_scale = 0.25+0.2*PS

    firing_probabilities = plateau_probability_calc(firing_rate, targetFR_scale*target_FR, prob_scale*base_prob, p_slope, min_prob=min_prob)

    # For neurons that are already in plateau state or in refractory peirod,
    # Set the plateau onset  probability to 0
    firing_probabilities[(~no_prev_plateau_flag)|(~no_plateau_refractory)] = 0

    np.random.seed(seed)
    rand_num = np.random.rand(len(activity))
    
    plateau_onset = (firing_probabilities > rand_num) # newly initiated plateau
    plateau_flag[plateau_onset] = True
    # print("Num of plateaus:", np.count_nonzero(plateau_flag==True))
    # plateau_duration[plateau_flag] += 1

    plateau_end = (~no_prev_plateau_flag & (activity < act_thr) & no_plateau_refractory)#|(~no_prev_plateau_flag & (plateau_duration > 2500) & no_plateau_refractory)
    plateau_flag[plateau_end] = False
    # plateau_duration[plateau_end] = 0

    plateau_refractory[plateau_end] += plateau_refrac_duration
    plateau_refractory[~no_plateau_refractory] -= 1
    plateau_refractory[plateau_refractory<0] = 0 # Just in case

    activity[plateau_onset] += plateau_scale*PT
    activity[plateau_end] = 0
    #activity[activity > 7*plateau_amp] = 7*plateau_amp

    if verbose:
        print("%d plateaus are onsetted at this timepoint."%len(np.where(plateau_onset==True)[0]))

    return activity, plateau_flag, plateau_refractory

def BTSP_update(A_presyn, A_postsyn, plateau_flag, w, connectivity, BTSP_scaling=1.0, verbose=False):

    A_presyn = A_presyn.reshape((-1, 1))

    # Select postsynaptic indices in plateau
    post_mask = plateau_flag == True
    if post_mask.sum() == 0:
        # Nothing to update
        updated_w = w.copy()
        updated_w[connectivity == False] = 0
        return updated_w

    A_postsyn_sel = A_postsyn[post_mask].reshape((1, int(post_mask.sum())))   # (1, M)
    multiplied = (A_presyn @ A_postsyn_sel)                                # (N_pre, M)
    multiplied *= BTSP_scaling

    pos_term = (sigmoid(multiplied, a_pos, b_pos) - pos_base) / pos_denominator
    neg_term = (sigmoid(multiplied, a_neg, b_neg) - neg_base) / neg_denominator

    # Only update the selected post columns
    w_sel = w[:, post_mask]
    update_w = ((wmax - w_sel) * k_pos * pos_term - w_sel * k_neg * neg_term)
    updated_w = w.copy()
    updated_w[:, post_mask] = w_sel + update_w

    if verbose:
        max_mult = float(multiplied.max())
        max_upd = float(update_w.max())
        print(max_mult, max_upd)

    # Enforce connectivity mask
    updated_w[connectivity == False] = 0
    return updated_w


def feat_weight_update(w_CA1_feat, CA1_FR, presence_vector):
    input_to_feat = np.matmul(CA1_FR, w_CA1_feat) # Has the size of 1*num_cues
    # print(input_to_cue)
    error = presence_vector-input_to_feat
    updated_w_CA1_cue = w_CA1_feat + cue_weight_LR*np.matmul(CA1_FR.reshape(-1,1),error.reshape(1,-1)) # Has the size of num_CA1_neurons*num_cues
    return updated_w_CA1_cue, error

def PS_update(presence, MI, novelty):
    return np.sum((presence+1)*MI*novelty)





def _generate_exp_rand_numbers(lambda_, n_rnds, seed):
    """
    MATLAB's random exponential number
    :param lambda_: normalization (will be the rate of Poisson proc - see `hom_poisson()`)
    :param n_rnds: number of random numbers to gerenerate
    :param seed: seed for random number generation
    :return: exponential random numbers
    """

    np.random.seed(seed)
    return -1.0 / lambda_ * np.log(np.random.rand(n_rnds))

def hom_poisson(lambda_, t_max, seed=None):
    """
    Vectorized homogeneous Poisson spike generator.
    Returns a CPU NumPy array for compatibility with downstream code.
    """
    if lambda_ <= 0 or t_max <= 0:
        return np.array([])

    if seed is not None:
        np.random.seed(seed)

    # Expected spikes ≈ Poisson(lambda * t_max); draw a safety margin per batch
    expected = int(max(256, lambda_ * t_max * 2.0))

    batches = []
    t_cursor = 0.0
    while t_cursor < t_max:
        isis = np.random.exponential(1.0 / lambda_, size=expected)
        cs = np.cumsum(isis)
        candidate_times = cs + t_cursor  # shift by current time
        mask = candidate_times <= t_max
        if mask.any():
            batches.append(candidate_times[mask])
        # advance by the full batch even if it overshoots (guarantees progress)
        t_cursor += float(cs[-1])

        # If we didn’t add any valid spikes, increase batch size (very high rate / t_max corner)
        if not mask.any():
            expected = int(expected * 1.5)

    if len(batches) == 0:
        return np.array([])

    spikes_backend = np.concatenate(batches, axis=0)
    # Return on CPU for compatibility (downstream code expects NumPy)
    return spikes_backend


def exponential_func(x, mult, gamma, baseline):
    """Exponential-like discount function: mult*gamma^x + baseline"""
    return mult*np.power(gamma, x) + baseline

def curve_fit_calc(
    x_data, 
    y_observed, 
    model_function,
    p0=[max_input_FR, 0.5, 2.0],
    bounds=([0, 1e-6, -1], [max_input_FR*2, 1-1e-9, max_input_FR//4]),
    sigma=None,
    absolute_sigma=False,
    param_names=None
):
    """
    Fits the model and returns parameter estimates along with standard errors,
    t-stats, and p-values for each parameter.
    """
    # Fit
    popt, pcov = curve_fit(
        model_function,
        x_data,
        y_observed,
        p0=p0,
        bounds=bounds,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        maxfev=10000
    )
    y_expected = model_function(x_data, *popt)

    # DoF and residual variance
    n = len(y_observed)
    p = len(popt)
    dof = max(n - p, 1)  # avoid divide-by-zero

    residuals = y_observed - y_expected
    # If sigma not given or absolute_sigma=False, scale covariance by reduced chi^2
    if (sigma is None) or (absolute_sigma is False):
        s_sq = np.sum(residuals**2) / dof
        pcov = pcov * s_sq

    # Standard errors, t-stats, p-values
    with np.errstate(divide='ignore', invalid='ignore'):
        se = np.sqrt(np.diag(pcov))
        t_stats = popt / se
        p_vals = 2 * student_t.sf(np.abs(t_stats), df=dof)

    # Package results
    if param_names is None:
        param_names = [f"param_{i}" for i in range(p)]
    results = {
        name: {
            "estimate": float(val),
            "std_err": float(err),
            "t_stat": float(t),
            "p_value": float(pv)
        }
        for name, val, err, t, pv in zip(param_names, popt, se, t_stats, p_vals)
    }

    return popt, y_expected, results

# ========== process Brian2 monitors ==========

def preprocess_monitors(SM, RM, calc_ISI=True):
    """
    preprocess Brian's SpikeMonitor and PopulationRateMonitor data for further analysis and plotting
    :param SM: Brian2 SpikeMonitor
    :param RM: Brian2 PopulationRateMonitor
    :param calc_ISI: bool for calculating ISIs
    :return spike_times, spiking_neurons: 2 lists: spike times and corresponding neuronIDs
            rate: firing rate of the population
            ISI_hist and ISI_bin_edges: bin heights and edges of the histogram of the ISI of the population
    """

    spike_times = np.array(SM.t_) * 1000.  # *1000 ms conversion
    spiking_neurons = np.array(SM.i_)
    tmp_spike_times = SM.spike_trains().items()
    rate = np.array(RM.rate_).reshape(-1, 10).mean(axis=1)

    if calc_ISI:
        ISIs = np.hstack([np.diff(spikes_i*1000) for i, spikes_i in tmp_spike_times])  # *1000 ms conversion
        ISI_hist, bin_edges = np.histogram(ISIs, bins=20, range=(0,1000))

        return spike_times, spiking_neurons, rate, ISI_hist, bin_edges
    else:
        return spike_times, spiking_neurons, rate


def _estimate_LFP(StateM, subset):
    """
    Estimates LFP by summing synaptic currents to PCs (assuming that all neurons are at equal distance (10 um) from the electrode)
    :param StateM: Brian2 StateMonitor object (of the PC population)
    :param subset: IDs of the recorded neurons
    :return: t, LFP: estimated LFP (in uV) and corresponding time points (in ms)
    """

    t = StateM.t_ * 1000.  # *1000 ms conversion
    LFP = np.zeros_like(t)*pA

    for i in subset:
        v = StateM[i].vm
        g_exc = StateM[i].g_ampa*nS + StateM[i].g_ampaMF*nS
        i_exc = g_exc * (v - (Erev_E * np.ones_like(v/mV)))  # pA
        g_inh = StateM[i].g_gaba*nS
        i_inh = g_inh * (v - (Erev_I * np.ones_like(v/mV)))  # pA
        LFP += -(i_exc + i_inh)  # (this is still in pA)

    LFP *= 1 / (4 * np.pi * volume_cond)

    return t, LFP/mV




# ========== 2 environments ==========

def reorder_spiking_neurons(spiking_neurons, pklf_name_tuning_curves):
    """
    Reorders spiking neurons based on the intermediate (non-ordered) place fields
    :param spiking_neurons: list of spiking neurons (ordered)
    :param pklf_name_tuning_curves: file name of the tuning curves (in the non-ordered env.) - used only for idx
    :return: reordered_spiking_neurons: same spiking neurons list with neuron idx swapped to the non-ordered env. ones
    """

    with open(pklf_name_tuning_curves, "rb") as f:
        place_fields = pickle.load(f, encoding="latin1")

    # create a mapping between gids in the ordered env. and the non-ordered one
    PF_idx = np.asarray(list(place_fields.keys()))
    PF_starts = np.asarray(list(place_fields.values()))
    sort_idx = np.argsort(PF_starts, kind="mergesort")
    sorted_PF_idx = PF_idx[sort_idx]
    # key: ordered, val: non-ordered
    id_map_PF = {neuron_id: PF_idx[i] for i, neuron_id in enumerate(sorted_PF_idx)}
    assert np.sum(list(id_map_PF.keys())) == np.sum(list(id_map_PF.values()))

    # create a random mapping for gids which don't have place fields in the non-ordered env.
    # in order to get rid of "ghost" replays - replays in the other env. in the raster plot
    # TODO investigate why this is needed!
    non_PFs = np.array([neuron_id for neuron_id in range(num_CA3_neurons) if neuron_id not in id_map_PF])
    tmp = deepcopy(non_PFs)
    np.random.shuffle(tmp)
    id_map_nonPF = {neuron_id: tmp[i] for i, neuron_id in enumerate(non_PFs)}
    assert np.sum(list(id_map_nonPF.keys())) == np.sum(list(id_map_nonPF.values()))

    reordered_spiking_neurons = np.zeros_like(spiking_neurons)
    for neuron_id in np.unique(spiking_neurons):
        if neuron_id in id_map_PF:  # place cells
            reordered_spiking_neurons[spiking_neurons == neuron_id] = id_map_PF[neuron_id]
        else:
            reordered_spiking_neurons[spiking_neurons == neuron_id] = id_map_nonPF[neuron_id]
    return reordered_spiking_neurons


# ========== saving & loading ==========


def create_dir(dir_name):
    """
    Deletes dir (if exists) and creates a new one with
    :param dir_name: string: full path of the directory to be created
    """
    if os.path.isdir(dir_name):
        rmtree(dir_name)
        os.mkdir(dir_name)
    else:
        os.mkdir(dir_name)


def save_place_fields(place_fields, pklf_name):
    """
    Save place field starts and corresponding neuron IDs for further analysis (see `bayesian_decoding.py`)
    :param place_fields: dict: neuron id:place field start
    :param pklf_name: name of saved file
    """

    with open(pklf_name, "wb") as f:
        pickle.dump(place_fields, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_vars(SM, RM, StateM, subset, seed, f_name="sim_vars_PC"):
    """
    Saves PC pop spikes, firing rate, membrane voltage, adaptation current and PSCs
    from a couple of recorded neurons after the simulation
    :param SM, RM: Brian2 SpikeMonitor, PopulationRateMonitor and StateMonitor
    :param subset: IDs of the recorded neurons
    :param seed: random seed used to run the simulation - here used only for naming
    :param f_name: name of saved file
    """

    spike_times, spiking_neurons, rate = preprocess_monitors(SM, RM, calc_ISI=False)
    # get PSCs from recorded voltage and conductances (and adaptation current)
    vs, PSCs, ws = {}, {}, {}
    for i in subset:
        v = StateM[i].vm
        vs[i] = v/mV
        g_exc = StateM[i].g_ampa*nS
        i_exc = -g_exc * (v - (Erev_E * np.ones_like(v/mV)))  # pA
        # separate outer (mossy fiber) input, from AMPA cond from local cells
        g_MF = StateM[i].g_ampaMF*nS
        i_MF = -g_MF * (v - (Erev_E * np.ones_like(v / mV)))  # pA
        g_inh = StateM[i].g_gaba*nS
        i_inh = -g_inh * (v - (Erev_I * np.ones_like(v/mV)))  # pA
        PSCs[i] = {"i_exc": i_exc/pA, "i_MF": i_MF/pA, "i_inh": i_inh/pA}
        ws[i] = StateM[i].w/pA
    # (shouldn't really be saved to PSCs only but keeping it for consistency)
    PSCs["t"] = StateM.t_ * 1000.  # *1000 ms conversion

    results = {"spike_times": spike_times, "spiking_neurons": spiking_neurons, "rate": rate,
               "vs": vs, "PSCs": PSCs, "ws": ws}
    pklf_name = os.path.join(base_path, "files", "%s_%s.pkl" % (f_name, seed))
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_PSD(f_PC, Pxx_PC, f_BC, Pxx_BC, f_LFP, Pxx_LFP, seed, f_name="PSD"):
    """
    Saves PSDs for PC and BC pop as well as LFP
    :params: f*, Pxx*: freqs and PSD (see `analyse_rate()` and `analyse_estimated_LFP()`)
    :param seed: random seed used to run the simulation - here used only for naming
    :param f_name: name of saved file
    """

    results = {"f_PC":f_PC, "Pxx_PC":Pxx_PC, "f_BC":f_BC, "Pxx_BC":Pxx_BC, "f_LFP":f_LFP, "Pxx_LFP":Pxx_LFP}
    pklf_name = os.path.join(base_path, "files", "%s_%s.pkl"%(f_name, seed))
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_TFR(f_PC, coefs_PC, f_BC, coefs_BC, f_LFP, coefs_LFP, seed, f_name="TFR"):
    """
    Saves TFR for PC and BC pop as well as LFP
    :params: f*, coefs*: freqs and coefficients (see `calc_TFR()`)
    :param seed: random seed used to run the simulation - here used only for naming
    :param f_name: name of saved file
    """

    results = {"f_PC":f_PC, "coefs_PC":coefs_PC, "f_BC":f_BC, "coefs_BC":coefs_BC, "f_LFP":f_LFP, "coefs_LFP":coefs_LFP}
    pklf_name = os.path.join(base_path, "files", "%s_%s.pkl"%(f_name, seed))
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_LFP(t, LFP, seed, f_name="LFP"):
    """
    Saves estimated LFP
    :params: t, LFP: time and LFP (see `analyse_estimated_LFP()`)
    :param seed: random seed used to run the simulation - here used only for naming
    :param f_name: name of saved file
    """

    results = {"t":t, "LFP":LFP}
    pklf_name = os.path.join(base_path, "files", "%s_%s.pkl"%(f_name, seed))
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_step_sizes(trajectories, step_sizes, avg_step_sizes, gamma_filtered_LFPs, f_name="step_sizes"):
    """
    Saves estimated trajectory, calculated step sizes and slow gamma filtered LFP
    :param trajectories: estimated (from posterior matrix) trajectories
    :param step_sizes: event step sizes calculated from estimated trajectories
    :param avg_step_size: average step sizes calculated from distance and time of trajectories
    :param gamma_filtered_LFPs: gamma freq filtered and sliced LFP
    """

    results = {"trajectories":trajectories, "step_sizes":step_sizes,
               "avg_step_sizes":avg_step_sizes, "gamma_filtered_LFPs":gamma_filtered_LFPs}
    pklf_name = os.path.join(base_path, "files", "%s.pkl"%f_name)
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_gavg_step_sizes(step_sizes, phases, avg_step_sizes, seeds, f_name="gavg_step_sizes"):
    """
    Saves estimated step sizes and phases from sims with multiple seeds
    :param step_sizes: event step sizes calculated from estimated trajectories
    :param phases: calculated phases for every step size
    :param avg_step_size: average step sizes calculated from distance and time of trajectories
    :param seeds: seeds of different sims
    """

    results = {"step_sizes":step_sizes, "phases":phases,
               "avg_step_sizes":avg_step_sizes, "seeds":seeds}
    pklf_name = os.path.join(base_path, "files", "%s.pkl"%f_name)
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_wmx(weightmx, npzf_name):
    """
    Saves excitatory weight matrix (as a `scipy.sparse` matrix)
    :param weightmx: synaptic weight matrix to save
    :param npzf_name: file name of the saved weight matrix
    """
    np.fill_diagonal(weightmx, 0.0)  # just to make sure
    sparse_weightmx = coo_matrix(weightmx)  # convert to COO
    save_npz(npzf_name, sparse_weightmx)


def load_wmx(npzf_name):
    """
    Dummy function to load in the excitatory weight matrix and make python clear the memory
    :param npzf_name: file name of the saved weight matrix
    :return: excitatory weight matrix
    """
    return load_npz(npzf_name)


def load_spikes(pklf_name):
    """
    Loads in saved spikes from simulations
    param pklf_name: name of saved file
    return: spike_times, spiking_neurons, rate
    """

    with open(pklf_name, "rb") as f:
        tmp = pickle.load(f, encoding="latin1")
    return tmp["spike_times"], tmp["spiking_neurons"], tmp["rate"]


def load_LFP(pklf_name):
    """
    Loads in saved LFP from simulations
    param pklf_name: name of saved file
    return: t, LFP
    """

    with open(pklf_name, "rb") as f:
        tmp = pickle.load(f, encoding="latin1")
    return tmp["t"], tmp["LFP"]





# ========== misc. ==========


def refractoriness(spike_trains, ref_per=5e-3):
    """
    Delete spikes (from generated train) which are too close to each other
    :param spike_trains: list of lists representing individual spike trains
    :param ref_per: refractory period (in sec)
    :return spike_trains: same structure, but with some spikes deleted
    """

    spike_trains_updated = []; count = 0
    if type(spike_trains) != list:
        spike_trains = [spike_trains]
    for single_spike_train in spike_trains:
        tmp = np.diff(single_spike_train)  # calculate ISIs
        idx = np.where(tmp < ref_per)[0] + 1
        if idx.size:
            count += idx.size
            single_spike_train_updated = np.delete(single_spike_train, idx).tolist()  # delete spikes which are too close
        else:
            single_spike_train_updated = single_spike_train
        spike_trains_updated.append(single_spike_train_updated)

    # print("%i spikes deleted because of too short refractory period" % count)

    return spike_trains_updated

def _get_consecutive_sublists(list_):
    """
    Groups list into sublists of consecutive numbers
    :param list_: input list to group
    :return cons_lists: list of lists with consecutive numbers
    """

    # get upper bounds of consecutive sublists
    ubs = [x for x,y in zip(list_, list_[1:]) if y-x != 1]

    cons_lists = []; lb = 0
    for ub in ubs:
        tmp = [x for x in list_[lb:] if x <= ub]
        cons_lists.append(tmp)
        lb += len(tmp)
    cons_lists.append([x for x in list_[lb:]])

    return cons_lists

def argmin_time_arrays(time_short, time_long):
    """
    Finds closest elements in differently sampled time arrays (used for step size analysis...)
    TODO: add some error management here....
    :param time_short: time array with less elements
    :param time_long: time array with more elements (in the same range)
    :return: idx of long array, to get closest elements to short array
    """

    return [np.argmin(np.abs(time_long-t)) for t in time_short]


def generate_cue_spikes():
    """Generates short (200ms) Poisson spike train at 20Hz (with brian2's `PoissonGroup()` one can't specify the duration)"""

    spike_times = np.asarray(hom_poisson(20.0, 10, t_max=0.2, seed=12345))
    spiking_neurons = np.zeros_like(spike_times)
    for neuron_id in range(1, 300):
        spike_times_tmp = np.asarray(hom_poisson(20.0, 10, t_max=0.2, seed=12345+neuron_id))
        spike_times = np.concatenate((spike_times, spike_times_tmp), axis=0)
        spiking_neurons_tmp = neuron_id * np.ones_like(spike_times_tmp)
        spiking_neurons = np.concatenate((spiking_neurons, spiking_neurons_tmp), axis=0)

    return spike_times, spiking_neurons


# def calc_spiketrain_ISIs():
#     """Calculates inter spike intervals within the generated spike trains (separately for place cells, non-place cells)"""

#     # just to get place cell idx
#     pklf_name = os.path.join(base_path, "files", "PFstarts_0.5_linear.pkl")
#     with open(pklf_name, "rb") as f:
#         PFs = pickle.load(f, encoding="latin1")

#     npzf_name = os.path.join(base_path, "files", "spike_trains_0.5_linear.npz")
#     npz_f = np.load(npzf_name)
#     spike_trains = npz_f["spike_trains"]

#     place_cell_ISIs = []
#     nplace_cell_ISIs = []
#     for i in range(num_CA3_neurons):
#         if i in PFs:
#             place_cell_ISIs.extend(np.diff(spike_trains[i]).tolist())
#         else:
#             nplace_cell_ISIs.extend(np.diff(spike_trains[i]).tolist())

#     results = {"PCs":np.asarray(place_cell_ISIs), "num_CA3_neurons":np.asarray(nplace_cell_ISIs)}
#     pklf_name = os.path.join(base_path, "files", "spiketrain_ISIs.pkl")
#     with open(pklf_name, "wb") as f:
#         pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


# def calc_single_cell_rates(seed):
#     """Calculates single cell firing rates for cells (separately for place cells, non-place cells and BCs)"""

#     # just to get place cell idx
#     pklf_name = os.path.join(base_path, "files", "PFstarts_0.5_linear.pkl")
#     with open(pklf_name, "rb") as f:
#         PFs = pickle.load(f, encoding="latin1")

#     pklf_name = os.path.join(base_path, "files", "sim_vars_PC_%s.pkl" % seed)
#     spike_times, spiking_neurons, rate = load_spikes(pklf_name)

#     place_cell_rates = []
#     nplace_cell_rates = []
#     for i in range(num_CA3_neurons):
#         spikes = spike_times[spiking_neurons == i]
#         if i in PFs:
#             place_cell_rates.append(len(spikes)/(len_sim/1000.))
#         else:
#             nplace_cell_rates.append(len(spikes)/(len_sim/1000.))

#     pklf_name = os.path.join(base_path, "files", "sim_vars_BC_%s.pkl" % seed)
#     spike_times, spiking_neurons, _ = load_spikes(pklf_name)

#     BC_rates = []
#     for i in range(num_IN_neurons):
#         spikes = spike_times[spiking_neurons == i]
#         BC_rates.append(len(spikes)/(len_sim/1000.))

#     results = {"PCs": np.asarray(place_cell_rates), "num_CA3_neurons": np.asarray(nplace_cell_rates), "BCs": np.asarray(BC_rates)}
#     pklf_name = os.path.join(base_path, "files", "single_cell_rates_%s.pkl" % seed)
#     with open(pklf_name, "wb") as f:
#         pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


# def calc_ISIs(seed):
#     """Calculates inter spike intervals for cells (separately for place cells, non-place cells and BCs)"""

#     # just to get place cell idx
#     pklf_name = os.path.join(base_path, "files", "PFstarts_0.5_linear.pkl")
#     with open(pklf_name, "rb") as f:
#         PFs = pickle.load(f, encoding="latin1")

#     pklf_name = os.path.join(base_path, "files", "sim_vars_PC_%s.pkl"%seed)
#     spike_times, spiking_neurons, _ = load_spikes(pklf_name)

#     place_cell_ISIs = []
#     nplace_cell_ISIs = []
#     for i in range(num_CA3_neurons):
#         idx = np.where(spiking_neurons == i)[0]
#         spikes = spike_times[idx]
#         if i in PFs:
#             place_cell_ISIs.extend(np.diff(spikes).tolist())
#         else:
#             nplace_cell_ISIs.extend(np.diff(spikes).tolist())

#     pklf_name = os.path.join(base_path, "files", "sim_vars_BC_%s.pkl"%seed)
#     spike_times, spiking_neurons, _ = load_spikes(pklf_name)

#     BC_ISIs = []
#     for i in range(num_IN_neurons):
#         idx = np.where(spiking_neurons == i)[0]
#         spikes = spike_times[idx]
#         BC_ISIs.extend(np.diff(spikes).tolist())

#     results = {"PCs":np.asarray(place_cell_ISIs), "num_CA3_neurons":np.asarray(nplace_cell_ISIs), "BCs":np.asarray(BC_ISIs)}
#     pklf_name = os.path.join(base_path, "files", "ISIs_%s.pkl"%seed)
#     with open(pklf_name, "wb") as f:
#         pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


# def calc_LFP_TFR(seed):
#     """Calculates TFR of the full LFP (not sliced, not downsampled)"""

#     pklf_name = os.path.join(base_path, "files", "LFP_%s.pkl"%seed)
#     t, LFP = load_LFP(pklf_name)
#     fs = 10000.

#     scales = np.concatenate((np.linspace(25, 80, 250), np.linspace(80, 300, 250)[1:]))  # 27-325 Hz  pywt.scale2frequency("morl", scale) / (1/fs)
#     coefs, freqs = pywt.cwt(LFP, scales, "morl", 1/fs)

#     results = {"coefs": coefs, "freqs": freqs}
#     pklf_name = os.path.join(base_path, "files", "LFP_TFR_%s.pkl" % seed)
#     with open(pklf_name, "wb") as f:
#         pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

def sigmoid(x, a, b):
    x = np.asarray(x)
    return 1 / (1 + np.exp(-b * (x - a)))

def normalized_sigmoid(x, a, b):
    x = np.asarray(x)
    s = sigmoid(x, a, b)
    s0 = sigmoid(0, a, b)
    s1 = sigmoid(1, a, b)
    return (s - s0) / (s1 - s0 + 1e-12)

def softplus(x):
    return np.log(1 + np.exp(x))



# Detect replay

def _avg_rate(rate, bin_, len_sim):
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

def emperical_tuning_curve(activity_arr):
    return {"%d"%ID: activity_arr[:,ID]/infield_rate for ID in range(len(activity_arr))}

def slice_high_activity(rate, th, min_len, len_sim, bin_=10):
    """
    Slices out high network activity - which will be candidates for replay detection
    :param rate: firing rate of the population
    :param th: rate threshold (above which is 'high activity')
    :param min_len: minimum length of continuous high activity (in ms)
    :param bin_: bin size for rate averaging (see `helper/_avg_rate()`)
    """

    assert min_len >= 128, "Spectral analysis won't work on sequences shorter than 128 ms"
    idx = np.where(_avg_rate(rate, bin_, len_sim) >= th)[0]
    high_act = _get_consecutive_sublists(idx.tolist())
    slice_idx = []
    for tmp in high_act:
        if len(tmp) >= np.floor(min_len/bin_):
            slice_idx.append((tmp[0]*bin_, (tmp[-1]+1)*bin_))
    # if not slice_idx:
    #     print("Sustained high network activity can't be detected"
    #           "(bin size:%i, min length:%.1f and threshold:%.2f)!" % (bin_, min_len, th))
    return slice_idx

def behavior_markov(P, total_time=600, start_state=0, end_state=[]):
    trajectory = np.zeros(total_time, dtype=int)
    trajectory[0] = start_state
    for t in range(1, total_time):
        trajectory[t] = np.random.choice(np.arange(P.shape[0]), p=P[trajectory[t-1],:])
        if trajectory[t] in end_state:
            goal_time = t
            goal_state = trajectory[t]
            trajectory[t:] = goal_state
            return trajectory, goal_time, goal_state
    return trajectory, np.nan, np.nan

# Detect oscillations

# def _autocorrelation(time_series):
    
    #Computes the autocorrelation of a time series
    #R(\tau) = \frac{E[(X_t - \mu)(X_{t+\tau} - \mu)]}{\sigma^2}
    #:param time_series: time series to analyse
    #:return: autocorrelation
    
    # var = np.var(time_series)
    # time_series = time_series - np.mean(time_series)
    # autocorrelation = np.correlate(time_series, time_series, mode="same") / var
    # return autocorrelation[int(len(autocorrelation)/2):]


def _calc_spectrum(time_series, fs, nperseg):
    """
    Estimates the power spectral density of the signal using Welch's method
    :param time_series: time series to analyse
    :param fs: sampling frequency
    :param nperseg: length of segments used in periodogram averaging
    :return f: frequencies used to evaluate PSD
            Pxx: estimated PSD
    """
    f, Pxx = signal.welch(time_series, fs=fs, window="hann", nperseg=nperseg)
    return f, Pxx


def analyse_rate(rate, fs, slice_idx=[], calc_pxx=True, rest_time=3000):
    """
    Basic analysis of firing rate: autocorrelation and PSD
    :param rate: firing rate of the neuron population
    :param fs: sampling frequency (for the spectral analysis)
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :return: mean_rate, rate_ac: mean rate, autocorrelation of the rate
             max_ac, t_max_ac: maximum autocorrelation, time interval of maxAC
             f, Pxx: sample frequencies and power spectral density (results of PSD analysis)
    """
    
    if slice_idx:
        print("slice idx exists")
        t = np.arange(0, rest_time); rates = []; highact_rates = []; lowact_rates = []
        prev_ub = 0
        for bounds in slice_idx:  # iterate through sustained high activity periods
            lb = bounds[0]; ub = bounds[1]
            rates.append(rate[np.where((lb <= t) & (t < ub))[0]])
            highact_rates+=list(rate[lb:ub])
            lowact_rates+=list(rate[prev_ub:lb])
            prev_ub = ub
        lowact_rates+=list(rate[prev_ub:rest_time])
        if not lowact_rates: lowact_rates = [0.]
        if not calc_pxx: return np.mean(rate)

        # AC and PSD are only analysed in the selected parts...
        # rate_acs = [_autocorrelation(rate_tmp) for rate_tmp in rates]
        # max_acs = [rate_ac[1:].max() for rate_ac in rate_acs]
        # t_max_acs = [rate_ac[1:].argmax()+1 for rate_ac in rate_acs]
        PSDs = [_calc_spectrum(rate_tmp, fs=fs, nperseg=256) for rate_tmp in rates]
        #f = PSDs[0][0]
        f_arr = [tmp[0] for tmp in PSDs]
        Pxxs = [tmp[1] for tmp in PSDs]

        return np.mean(highact_rates), np.mean(lowact_rates), f_arr, Pxxs
    else:
        print("No slice idx")
        if not calc_pxx: return np.mean(rate)
        # if len(rate) > 0:
        #     rate_ac = _autocorrelation(rate)
        # else: rate_ac = 0
        f, Pxx = _calc_spectrum(rate, fs=fs, nperseg=512)
        return 0, np.mean(rate), f, Pxx


def calc_TFR(rate, fs, slice_idx=[]):
    """
    Creates time-frequency representation using wavelet analysis
    :param rate: firing rate of the neuron population
    :param fs: sampling frequency (for the spectral analysis)
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :return: coefs, freqs: coefficients from wavelet transform and frequencies used
    """
    import pywt

    scales = np.linspace(3.5, 5, 300)  # 162-232 Hz  pywt.scale2frequency("morl", scale) / (1/fs)
    # 27-325 Hz for 10 kHz sampled LFP...
    # scales = np.concatenate((np.linspace(25, 80, 150), np.linspace(80, 300, 150)[1:]))

    if slice_idx:
        t = np.arange(0, 10000); rates = []
        for bounds in slice_idx:  # iterate through sustained high activity periods
            lb = bounds[0]; ub = bounds[1]
            rates.append(rate[np.where((lb <= t) & (t < ub))[0]])
        wts = [pywt.cwt(rate, scales, "morl", 1/fs) for rate in rates]
        coefs = [tmp[0] for tmp in wts]
        freqs = wts[0][1]
    else:
        coefs, freqs = pywt.cwt(rate, scales, "morl", 1/fs)
    return coefs, freqs


def ripple_AC(rate_acs, slice_idx=[]):
    """
    Analyses AC of rate (in the ripple freq)
    :param rate_acs: auto correlation function(s) of rate see (`analyse_rate()`)
    :return: max_ac_ripple, t_max_ac_ripple: maximum autocorrelation in ripple range, time interval of maxACR
    """
    if slice_idx:
        max_ac_ripple = [rate_ac[3:9].max() for rate_ac in rate_acs]  # hard coded values in ripple range (works with 1ms binning...)
        t_max_ac_ripple = [rate_ac[3:9].argmax()+3 for rate_ac in rate_acs]
        return np.mean(max_ac_ripple), np.mean(t_max_ac_ripple)
    else:
        return rate_acs[3:9].max(), rate_acs[3:9].argmax()+3


def _fisher(Pxx):
    """
    Performs Fisher g-test on PSD (see Fisher 1929: http://www.jstor.org/stable/95247?seq=1#page_scan_tab_contents)
    :param Pxx: power spectral density (see `_calc_spectrum()`)
    :return p_val: p-value
    """
    if np.sum(Pxx) == 0: return 1.0
    fisher_g = Pxx.max() / np.sum(Pxx)
    n = len(Pxx); upper_lim = int(np.floor(1. / fisher_g))
    p_val = np.sum([np.power(-1, i-1) * comb(n, i) * np.power((1-i*fisher_g), n-1) for i in range(1, upper_lim)])
    return p_val


def ripple(fs, Pxx, slice_idx=[], p_th=0.05):
    """
    Decides if there is a significant high freq. ripple oscillation by applying Fisher g-test on the power spectrum
    :param f, Pxx: calculated power spectrum of the neural activity and frequencies used to calculate it (see `analyse_rate()`)
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :param p_th: significance threshold for Fisher g-test
    :return: avg_ripple_freq, ripple_power: average frequency and power of ripple band oscillation
    """

    
    if slice_idx:
        p_vals, freqs, ripple_powers = [], [], []
        for i in range(len(Pxx)):
            f = np.asarray(fs[i])
            if sum(Pxx[i]) == 0:
                p_vals.append(1.0)
                freqs.append(np.nan)
                ripple_powers.append(0.0)
                continue
            Pxx_ripple = Pxx[i][np.where((150 < f) & (f < 220))]
            p_vals.append(_fisher(Pxx_ripple))
            freqs.append(Pxx_ripple.argmax())
            ripple_powers.append((sum(Pxx_ripple) / sum(Pxx[i])) * 100)
        idx = np.where(np.asarray(p_vals) <= p_th)[0].tolist()
        if len(idx) >= 0.25*len(slice_idx):  # if at least 25% are significant
            avg_freq = np.mean(np.asarray(freqs)[idx])
            try: avg_ripple_freq = f[np.where(150 < f)[0][0] + int(avg_freq)]
            except: avg_ripple_freq = (np.nan)
        else:
            avg_ripple_freq = np.nan
        return avg_ripple_freq, np.mean(ripple_powers)
    else:
        if sum(Pxx) == 0:
            p_vals = (1.0)
            avg_ripple_freq = (np.nan)
            ripple_power = (0.0)
        else:
            f = np.asarray(fs)
            Pxx = np.asarray(Pxx)
            Pxx_ripple = Pxx[np.where((150 < f) & (f < 220))]
            p_val = _fisher(Pxx_ripple)
            avg_ripple_freq = f[np.where(150 < f)[0][0] + Pxx_ripple.argmax()] if p_val < p_th else np.nan
            ripple_power = (sum(Pxx_ripple) / sum(Pxx)) * 100
        return avg_ripple_freq, ripple_power


def gamma(fs, Pxx, slice_idx=[], p_th=0.05):
    """
    Decides if there is a significant gamma freq. oscillation by applying Fisher g-test on the power spectrum
    :param f, Pxx: calculated power spectrum of the neural activity and frequencies used to calculate it (see `analyse_rate()`)
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :param p_th: significance threshold for Fisher g-test
    :return: avg_gamma_freq, gamma_power: average frequency and power of the oscillation
    """

    if slice_idx:
        p_vals, freqs, gamma_powers = [], [], []
        for i in range(len(Pxx)):
            f = np.asarray(fs[i])
            if sum(Pxx[i]) == 0:
                p_vals.append(1.0)
                freqs.append(np.nan)
                gamma_powers.append(0.0)
                continue
            Pxx_gamma = Pxx[i][np.where((30 < f) & (f < 100))]
            p_vals.append(_fisher(Pxx_gamma))
            freqs.append(Pxx_gamma.argmax())
            gamma_powers.append((sum(Pxx_gamma) / sum(Pxx[i])) * 100)
        idx = np.where(np.asarray(p_vals) <= p_th)[0].tolist()
        if len(idx) >= 0.25*len(slice_idx):  # if at least 25% are significant
            avg_freq = np.mean(np.asarray(freqs)[idx])
            try:
                avg_gamma_freq = f[np.where(30 < f)[0][0] + int(avg_freq)]
            except:
                avg_gamma_freq = np.nan
        else:
            avg_gamma_freq = np.nan
        return avg_gamma_freq, np.mean(gamma_powers)
    else:
        f = np.asarray(fs)
        Pxx = np.asarray(Pxx)
        Pxx_gamma = Pxx[np.where((30 < f) & (f < 100))]
        p_val = _fisher(Pxx_gamma)
        avg_gamma_freq = f[np.where(30 < f)[0][0] + Pxx_gamma.argmax()] if p_val < p_th else np.nan
        gamma_power = (sum(Pxx_gamma) / sum(Pxx)) * 100
        return avg_gamma_freq, gamma_power


def lowfreq(fs, Pxx, slice_idx=[], p_th=0.05):
    """
    Decides if there is a significant sub gamma (alpha, beta) freq. oscillation by applying Fisher g-test on the power spectrum
    (This function is only used during optimizations to supress low freq. oscillations and ensure that the oscillation
    we get is not a harmonic of those)
    :param f, Pxx: calculated power spectrum of the neural activity and frequencies used to calculate it (see `analyse_rate()`)
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :param p_th: significance threshold for Fisher g-test
    :return: avg_subgamma_freq, subgamma_power: average frequency and power of the oscillations
    """

    if slice_idx:
        p_vals, freqs, subgamma_powers = [], [], []
        for i in range(len(Pxx)):
            f = np.asarray(fs[i])
            if sum(Pxx[i]) == 0:
                p_vals.append(1.0)
                freqs.append(np.nan)
                subgamma_powers.append(0.0)
                continue
            Pxx_subgamma = Pxx[i][np.where(f < 30)]
            p_vals.append(_fisher(Pxx_subgamma))
            freqs.append(Pxx_subgamma.argmax())
            subgamma_powers.append((sum(Pxx_subgamma) / sum(Pxx[i])) * 100)
        idx = np.where(np.asarray(p_vals) <= p_th)[0].tolist()
        if len(idx) >= 0.25*len(slice_idx):  # if at least 25% are significant
            avg_subgamma_freq = int(np.mean(np.asarray(freqs)[idx]))
        else:
            avg_subgamma_freq = np.nan
        return avg_subgamma_freq, np.mean(subgamma_powers)
    else:
        f = np.asarray(fs)
        Pxx = np.asarray(Pxx)
        Pxx_subgamma = Pxx[f < 30]
        p_val = _fisher(Pxx_subgamma)
        avg_subgamma_freq = np.max(Pxx_subgamma) if p_val < p_th else np.nan
        subgamma_power = (sum(Pxx_subgamma) / sum(Pxx)) * 100
        return avg_subgamma_freq, subgamma_power


def lowpass_filter(time_series, fs=10000., cut=500.):
    """
    Low-pass filters time series (3rd order Butterworth filter) - (used for LFP)
    :param time_series: time series to analyse
    :param fs: sampling frequency
    :param cut: cut off frequency
    :return: filtered time_series
    """
    b, a = signal.butter(3, cut/(fs/2.), btype="lowpass")
    return signal.filtfilt(b, a, time_series, axis=0)


def bandpass_filter(time_series, fs=10000., cut=np.array([25., 60.])):
    """
    Band-pass filters time series (3rd order Butterworth filter) - (used for LFP)
    :param time_series: time series to analyse
    :param fs: sampling frequency
    :param cut: cut off frequencies
    :return: filtered time_series
    """
    b, a = signal.butter(3, cut/(fs/2.), btype="bandpass")
    return signal.filtfilt(b, a, time_series, axis=0)


def calc_phase(time_series):
    """
    Gets phase of the signal from the Hilbert transform
    :param time_series: time series to analyse
    :return: exctracted phase of the time_series
    """
    z = signal.hilbert(time_series)
    return np.angle(z)


def analyse_estimated_LFP(StateM, subset, slice_idx=[], fs=10000.):
    """
    Analyses estimated LFP (see also `_calculate_LFP()`)
    :param StateM, subset: see `_calculate_LFP()`
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :param fs: sampling frequency
    :return: t, LFP: estimated LFP and corresponding time vector
             f, Pxx: sample frequencies and power spectral density (results of PSD analysis)
    """

    t, LFP = _estimate_LFP(StateM, subset)
    LFP = lowpass_filter(LFP, fs)

    LFPs = []
    if slice_idx:
        for bounds in slice_idx:  # iterate through sustained high activity periods
            lb = bounds[0]; ub = bounds[1]
            LFPs.append(LFP[np.where((lb <= t) & (t < ub))[0]])

            PSDs = [_calc_spectrum(LFP_tmp, fs, nperseg=2048) for LFP_tmp in LFPs]
        f = PSDs[0][0]
        Pxxs = np.array([tmp[1] for tmp in PSDs])
        # for comparable results cut spectrum at 500 Hz
        f = np.asarray(f)
        idx = np.where(f < 500)[0]
        f = f[idx]
        Pxxs = Pxxs[:, idx]
        return t, LFP, f, Pxxs
    else:
        f, Pxx = _calc_spectrum(LFP, fs, nperseg=4096)
        # for comparable results cut spectrum at 500 Hz
        f = np.asarray(f)
        idx = np.where(f < 500)[0]
        f = f[idx]
        Pxx = Pxx[idx]
        return t, LFP, f, Pxx

# Bayesian decoding

def extract_binspikecount(lb, ub, delta_t, t_incr, spike_times, spiking_neurons, tuning_curves):
    """
    Builds container of spike counts in a given interval (bin)
    In order to save time in likelihood calculation only neurons which spike are taken into account
    :param lb, ub: lower and upper bounds for time binning
    :param delta_t: window size (in ms)
    :param t_incr: increment or step size (if less than delta_t than it's an overlapping sliding window)
    :param spike_times: np.array of ordered spike times (saved and loaded in ms)
    :param spiking_neurons: np.array (same shape as spike_times) with corresponding neuron IDx
    :param tuning_curves: dictionary of tuning curves {neuronID: tuning curve} (see `load_tuning_curves()`)
    :return: list (1 entry for every time bin) of dictionaries {i: n_i}
    """

    assert delta_t >= t_incr
    bin_spike_counts = []
    t_start = lb; t_end = lb + delta_t
    while t_end < ub + t_incr:
        n_spikes = {}
        neuron_idx, counts = np.unique(spiking_neurons[np.where((t_start <= spike_times) & (spike_times < t_end))],
                                       return_counts=True)
        for i, count in zip(neuron_idx, counts):
            if i in tuning_curves:
                n_spikes[i] = count
        bin_spike_counts.append(n_spikes)
        t_start += t_incr; t_end += t_incr
    return bin_spike_counts


def calc_posterior(bin_spike_counts, tuning_curves, delta_t):
    """
    Calculates posterior distribution of decoded place Pr(x|spikes) based on Davison et al. 2009
    Pr(spikes|x) = \prod_{i=1}^N \frac{(\Delta t*tau_i(x))^n_i}{n_i!} e^{-\Delta t*tau_i(x)} (* uniform prior...)
    (It actually implements it via log(likelihoods) for numerical stability)
    Assumptions: independent neurons; firing rates modeled with Poisson processes
    Vectorized implementation using only the spiking neurons in each bin
    (plus taking only the highest fraction before summing...)
    :param bin_spike_counts: list (1 entry for every time bin) of spike dictionaries {i: n_i} (see `extract_binspikecount()`)
    :param tuning_curves: dictionary of tuning curves {neuronID: tuning curve} (see `helper.py/load_tuning_curves()`)
    :param delta_t: delta t used for binning spikes (in ms)
    return: X_posterior: spatial_resolution*temporal_resolution array with calculated posterior probability Pr(x|spikes)
    """
    import random as pyrandom
    delta_t *= 1e-3  # convert back to second
    n_spatial_points = pyrandom.sample(list(tuning_curves.values()), 1)[0].shape[0]
    X_posterior = np.zeros((n_spatial_points, len(bin_spike_counts)))  # dim:x*t

    # could be a series of 3d array operations instead of this for loop...
    # ...but since only a portion of the 8000 neurons are spiking in every bin this one might be even faster
    for t, spikes in enumerate(bin_spike_counts):
        # prepare broadcasted variables
        n_spiking_neurons = len(spikes)
        expected_spikes = np.zeros((n_spatial_points, n_spiking_neurons))  # dim:x*i_spiking
        n_spikes = np.zeros_like(expected_spikes)  # dim:x*i_spiking
        n_factorials = np.ones_like(expected_spikes)  # dim:x*i_spiking
        for j, (neuron_id, n_spike) in enumerate(spikes.items()):
            tuning_curve = tuning_curves[neuron_id] * infield_rate
            tuning_curve[np.where(tuning_curve <= 0.1)] = 0.1
            expected_spikes[:, j] = tuning_curve * delta_t
            n_spikes[:, j] = n_spike
            n_factorials[:, j] = factorial(n_spike).item()
        # calculate log(likelihood)
        likelihoods = np.multiply(expected_spikes, 1.0/n_factorials)
        likelihoods = np.multiply(n_spikes, np.log(likelihoods))
        likelihoods = likelihoods - delta_t * expected_spikes
        likelihoods.sort(axis=1, kind="mergsort")
        if likelihoods.shape[1] > 100:
            likelihoods = likelihoods[:, -100:]  # take only the 100 highest values for numerical stability
        likelihoods = np.sum(likelihoods, axis=1)
        likelihoods -= np.max(likelihoods)  # normalize before exp()
        likelihoods = np.exp(likelihoods)
        # calculate posterior
        X_posterior[:, t] = likelihoods / np.sum(likelihoods)
    return X_posterior


def _line(x, a, b):
    """
    Dummy function used for line fitting
    :param x: independent variable
    :param a, b: slope and intercept
    """
    return a*x + b


def _evaluate_fit(X_posterior, y, band_size=2):
    """
    Calculates the goodness of fit based on Davison et al. 2009 (line fitting in a probability matrix)
    R(v, rho) = \frac{1}{n} \sum_{k=1}^n-1 Pr(|pos - (rho + v*k*\Delta t)| < d)
    Masking matrix is based on Olafsdottir et al. 2016's MATLAB implementation
    :param X_posterior: posterior matrix (see `get_posterior()`)
    :param y: candidate fitted line
    :param band_size: distance (up and down) from fitted line to consider
    :return: R: goodness of fit (in [0, 1])
    """

    n_spatial_points = X_posterior.shape[0]
    t = np.arange(0, X_posterior.shape[1])
    line_idx = np.clip(np.round(y)+n_spatial_points, 0, n_spatial_points*3-1).astype(int)  # convert line to matrix idx
    # check if line is "long enough"
    if len(np.where((n_spatial_points <= line_idx) & (line_idx < n_spatial_points*2))[0]) < n_spatial_points / 3.0:
        return 0.0
    mask = np.zeros((n_spatial_points*3, X_posterior.shape[1]))  # extend on top and bottom
    mask[line_idx, t] = 1
    # convolve with kernel to get the desired band width
    mask = convolve2d(mask, np.ones((2*band_size+1, 1)), mode="same")
    mask = mask[int(n_spatial_points):int(n_spatial_points*2), :]  # remove extra padding to get X_posterior's shape
    R = np.sum(np.multiply(X_posterior, mask)) / np.sum(X_posterior)
    return R


def fit_trajectory(X_posterior, slope_lims=(0.5, 3), grid_res=100):
    """
    Brute force trajectory fit in the posterior matrix (based on Davison et al. 2009, see: `_evaluate_fit()`)
    :param X_posterior: posterior matrix (see `get_posterior()`)
    :param slope_lims: lower and upper bounds of splopes to test
    :param grid_res: number of points to try along one dimension
    :return: highest_R: best goodness of fit (see `_evaluate_fit()`)
             fit: fitted line
             best_params: slope and offset parameter corresponding to the highest R
    """

    slopes = np.concatenate((np.linspace(-slope_lims[1], -slope_lims[0], int(grid_res/2.)),
                             np.linspace(slope_lims[0], slope_lims[1], int(grid_res/2.))))
    offsets = np.linspace(-0.5*X_posterior.shape[0], X_posterior.shape[0]*1.5, grid_res)
    t = np.arange(0, X_posterior.shape[1])
    best_params = (slopes[0], offsets[0]); highest_R = 0.0
    for a in slopes:
        for b in offsets:
            y = _line(t, a, b)
            R = _evaluate_fit(X_posterior, y)
            if R > highest_R:
                highest_R = R
                best_params = (a, b)
    fit = _line(t, *best_params)
    return highest_R, fit, best_params


def _shuffle_tuning_curves(tuning_curves, seed):
    """
    Shuffles neuron IDx and corresponding tuning curves (used for significance test)
    :param tuning_curves: {neuronID: tuning curve}
    :param seed: random seed for shuffling
    """
    keys = list(tuning_curves.keys())
    vals = list(tuning_curves.values())
    np.random.seed(seed)
    np.random.shuffle(keys)
    return {key: vals[i] for i, key in enumerate(keys)}


def _test_significance_subprocess(inputs):
    """
    Subprocess used by multiprocessing pool for significance test: log(likelihood) calculation and line fit
    :param inputs: see `calc_log_likelihoods()`
    :return: R: see `fit_trajectory()`
    """
    X_posterior = calc_posterior(*inputs)
    R, _, _ = fit_trajectory(X_posterior)
    return R


def test_significance(bin_spike_counts, tuning_curves, delta_t, R, N):
    """
    Test significance of fitted trajectory (and detected sequence replay) by shuffling the data and re-fitting many times
    :param delta_t, bin_spike_counts, tuning_curves: see `calc_log_likelihoods()`
    :param R: reference goodness of fit (from unshuffled data)
    :param N: number of shuffled versions tested
    :return: Rs: list of goodness of fits from the shuffled events
    """
    import multiprocessing as mp
    orig_tuning_curves = deepcopy(tuning_curves)  # just to make sure...
    shuffled_tuning_curves = [_shuffle_tuning_curves(orig_tuning_curves, seed=12345+i) for i in range(N)]
    n = N if mp.cpu_count()-1 > N else mp.cpu_count()-1
    pool = mp.Pool(processes=n)
    Rs = pool.map(_test_significance_subprocess,
                  zip([bin_spike_counts for _ in range(N)], shuffled_tuning_curves, [delta_t for _ in range(N)]))
    pool.terminate()
    significance = 1 if R > np.percentile(Rs, 95) else np.nan
    return significance, sorted(Rs)
    
