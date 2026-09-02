"""
run_offline.py
==============
Offline (rest/sleep) learning pipeline for the hippocampal BTSP model.

Overview
--------
After each online (active exploration) lap the network can undergo an offline
consolidation phase that models hippocampal sharp-wave ripple (SWR) replay.
The pipeline has three stages:

  Stage 1 — Parameter optimisation  (find_optimal_parameters)
      A BluePyOpt CMA-ES/DEAP evolutionary algorithm searches for synaptic
      weight scale factors that make the recurrent Brian2 network produce
      SWR-like population bursts (150–220 Hz ripple, low background rate).
      Optimal parameters are saved to results/<task>/optimization/.

  Stage 2 — Offline spike simulation  (simulate_offline_spikes)
      The Brian2 spiking network (CA3 + CA1 + interneurons) is run using the
      optimised parameters and the weight matrices learned online.  Spike
      trains and population rates are saved for downstream analysis.

  Stage 3 — STDP weight update  (train_network)
      Spike trains from Stage 2 are replayed through a rate-based STDP rule
      to update w_CA3_CA1 (the cognitive map weights), consolidating the
      spatial memory formed during exploration.

Brian2 network used in stages 1 and 2:
    CA3_PCs  (num_CA3_neurons)  — place cells, AdExpIF model
    CA3_INs  (num_IN_neurons)   — fast-spiking interneurons, AdExpIF model
    CA1_PCs  (num_CA1_neurons)  — place cells, AdExpIF model
    CA1_INs  (num_IN_neurons)   — fast-spiking interneurons, AdExpIF model

Spike sources during offline simulation:
    MF  — mossy-fibre background (PoissonGroup to CA3_PCs, rate = rate_MF Hz)
    cue — location-specific cue input (PoissonGroup to a subset of CA3_PCs
          centred on pause_state), present only when pause_state >= 0.

Attribution
-----------
Several functions in this file are adapted from the ca3net codebase
(https://github.com/KaliLab/ca3net).  Individual functions are annotated.

Public entry points
-------------------
    find_optimal_parameters(mode, target_trial, target_lap, ...)
    simulate_offline_spikes(mode, target_lap, trial_number, ...)
    train_network(mode, target_lap, trial_number, ...)
    detect_replay_type(mode, target_lap, trial_number, ...)
    find_replay_onset(mode, trial_names, ...)  — earliest lap with a
        spontaneous CA3 replay burst, for CA3-only (no CA1) network variants
        such as the plain-STDP `STDP<N>` control trials.
    simulate_potentiated_replay(trial_number, data_dir, ...)  — T-maze-only;
        selective-potentiation CA3 replay for the Supple_replay_Carey figure.
    simulate_fatigued_replay(trial_number, data_dir, ...)  — linear-shock-only;
        selective-fatigue CA3 replay for the Supple_replay_Wu figure.
"""

from tabnanny import verbose
import os, sys, logging, traceback, gc
import numpy as np
import bluepyopt as bpop
import multiprocessing as mp
import warnings
from scipy.sparse import coo_matrix
from brian2 import *
from tqdm import tqdm
import copy as cp
warnings.filterwarnings('ignore')

base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
sys.path.insert(0, os.path.join(base_path, "code/functions"))

from common_functions import preprocess_monitors, find_PF_peak, save_place_fields
from global_variables import background_rate, num_IN_neurons, IN_connection_prob, CA3_CA3_connection_prob, CA3_CA1_connection_prob, infield_rate, place_thr_FR, theta_mod_factor
from spiking_neurons_params import *
from common_functions import hom_poisson, analyse_rate, ripple, gamma, slice_high_activity, input_driven_rate

# ---------------------------------------------------------------------------
# REVISED FROM ca3net (https://github.com/KaliLab/ca3net)
# Original: generated cue spikes for a single burst event.
# Changes: generalised to accept an arbitrary spike_rates vector and
#          an explicit duration parameter; removed Brian2 dependency.
# ---------------------------------------------------------------------------
def generate_cue_spikes(spike_rates, duration, round_dec=3):
    """
    Generate Poisson spike trains for a population of cue neurons.

    Parameters
    ----------
    spike_rates : array-like
        Firing rates [Hz] for each neuron in the cue population.
    duration : float
        Simulation duration [ms].
    round_dec : int
        Decimal places to round spike times to (avoids floating-point collisions).

    Returns
    -------
    spike_times : np.ndarray
        Concatenated spike times [s] across all neurons.
    spiking_neurons : np.ndarray
        Neuron index for each entry in spike_times.
    """
    sec_dur = duration/1000
    spike_times = np.asarray(hom_poisson(spike_rates[0], t_max=sec_dur, seed=12345))
    spike_times = np.unique(np.round(spike_times,decimals=round_dec))
    spiking_neurons = np.zeros_like(spike_times)
    
    for neuron_id in range(1, len(spike_rates)):
        spike_times_tmp = np.asarray(hom_poisson(spike_rates[neuron_id], t_max=sec_dur, seed=12345+neuron_id))
        spike_times_tmp = np.unique(np.round(spike_times_tmp,decimals=round_dec))
        spike_times = np.concatenate((spike_times, spike_times_tmp), axis=0)
        spiking_neurons_tmp = neuron_id * np.ones_like(spike_times_tmp)
        spiking_neurons = np.concatenate((spiking_neurons, spiking_neurons_tmp), axis=0)

    return spike_times, spiking_neurons

def simulate_offline_spikes(mode, target_lap, trial_number, pause_state=None):
    """
    Run the offline (rest/sleep) Brian2 simulation and save spike data.

    Loads pre-optimised synaptic parameters (Stage 1) and the weight matrices
    learned after `target_lap` online laps, then runs `offline_simulation` once
    per trial.  Spike times and population rates for both CA3 and CA1 are saved
    to compressed .npz files for downstream analysis and STDP training.

    Parameters
    ----------
    mode : int
        Task selector (0=linear_reward, 1=Tmaze, 2=linear_shock).
    target_lap : int
        Which online lap's weight matrices to use as the starting point.
    trial_number : int
        Number of trials to simulate.
    pause_state : int or None
        Maze state ID where the cue input is injected (models "replay seeding"
        from a specific location).  None → spontaneous replay without a cue.
    """
    if mode == 0:
        from linear_reward_functions import load_PF_starts
        file_dir = os.path.join(base_path,"results/linear_reward")
        CA3_PF = load_PF_starts()
    elif mode == 1:
        from Tmaze_functions import load_PF_starts
        file_dir = os.path.join(base_path,"results/Tmaze")
        CA3_PF = load_PF_starts()
    elif mode == 2:
        from linear_shock_functions import load_PF_starts
        file_dir = os.path.join(base_path,"results/linear_shock")
        CA3_PF = load_PF_starts()

    for trial in range(trial_number):
        foldername = "trial%d"%trial
        offline_param = np.load(os.path.join(file_dir,"optimization/parameter_lap_%d.npz" % target_lap))
        f_in = os.path.join(file_dir,foldername,"lap_%d.npz"%(target_lap))
        w_CA3 = coo_matrix(np.load(f_in)["w_CA3_CA3"])
        w_CA1 = coo_matrix(np.load(f_in)["w_CA3_CA1"])

        SM_PC_CA3, SM_IN_CA3, RM_PC_CA3, RM_IN_CA3,\
        SM_PC_CA1, SM_IN_CA1, RM_PC_CA1, RM_IN_CA1,\
        _, _, _, _, _, _, _ = offline_simulation(mode, w_CA3, w_CA1, pause_state, CA3_PF, **offline_param)
        
        spike_times_CA3_PC, spiking_neurons_CA3_PC, rate_CA3_PC = preprocess_monitors(SM_PC_CA3, RM_PC_CA3, calc_ISI=False)
        spike_times_CA3_IN, spiking_neurons_CA3_IN, _ = preprocess_monitors(SM_IN_CA3, RM_IN_CA3, calc_ISI=False)
        file_out = os.path.join(file_dir,foldername,"CA3_replay_lap_%d_pause_%d.npz"%(target_lap, pause_state)) if pause_state!=None else os.path.join(file_dir,foldername,"CA3_replay_lap_%d.npz"%(target_lap))
        np.savez(file_out,
                spike_times_CA3_PC=spike_times_CA3_PC, spiking_neurons_CA3_PC=spiking_neurons_CA3_PC,
                rate_CA3_PC=rate_CA3_PC,spike_times_CA3_IN=spike_times_CA3_IN, spiking_neurons_CA3_IN=spiking_neurons_CA3_IN)
        
        spike_times_CA1_PC, spiking_neurons_CA1_PC, rate_CA1_PC = preprocess_monitors(SM_PC_CA1, RM_PC_CA1, calc_ISI=False)
        spike_times_CA1_IN, spiking_neurons_CA1_IN, _ = preprocess_monitors(SM_IN_CA1, RM_IN_CA1, calc_ISI=False)
        file_out = os.path.join(file_dir,foldername,"CA1_replay_lap_%d_pause_%d.npz"%(target_lap, pause_state)) if pause_state!=None else os.path.join(file_dir,foldername,"CA1_replay_lap_%d.npz"%(target_lap))
        np.savez(file_out,
                spike_times_CA1_PC=spike_times_CA1_PC, spiking_neurons_CA1_PC=spiking_neurons_CA1_PC,
                rate_CA1_PC=rate_CA1_PC,spike_times_CA1_IN=spike_times_CA1_IN, spiking_neurons_CA1_IN=spiking_neurons_CA1_IN)

def ca3_offline_simulation(mode, wmx_CA3, duration_ms, w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3, wmx_mult_CA3, w_PC_MF_CA3, rate_MF, cue_state=-1, verbose=False):
    """
    Run a CA3-only offline Brian2 simulation (no CA1), seeded with a
    location-specific cue at `cue_state`.

    Parameters
    ----------
    wmx_CA3 : scipy.sparse.coo_matrix
        CA3-CA3 recurrent weight matrix (already scaled/potentiated if desired).
    PF : dict {neuron_id: [row, col]}
        CA3 place field peak positions.
    cue_state : int
        0-indexed T-maze state at which the replay cue is injected.
    duration_ms : float
        Simulation duration [ms].
    w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3, wmx_mult_CA3, w_PC_MF_CA3, rate_MF :
        Same meaning as in `offline_simulation`.
    verbose : bool
        If True, Brian2 reports simulation progress.

    Returns
    -------
    SM_PC, RM_PC, SM_IN, RM_IN : Brian2 SpikeMonitor/PopulationRateMonitor for CA3 PCs/INs.
    cue_ids : np.ndarray, CA3 neuron IDs that received the cue.
    """
    if mode == 0:
        from linear_reward_variables import num_CA3_neurons, state_position
        from linear_reward_functions import generate_place_cell_ID_list, get_tuning_curve, load_PF_starts
        
    if mode == 1:
        from Tmaze_variables import num_CA3_neurons, state_position
        from Tmaze_functions import generate_place_cell_ID_list, get_tuning_curve, load_PF_starts
    if mode == 2:
        from linear_shock_variables import num_CA3_neurons, state_position
        from linear_shock_functions import generate_place_cell_ID_list, get_tuning_curve, load_PF_starts

    start_scope()

    CA3_PCs = NeuronGroup(num_CA3_neurons, model=eqs_PC, threshold="vm>spike_th_PC",
                      reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    CA3_PCs.vm = Vrest_PC; CA3_PCs.g_ampa = 0.0; CA3_PCs.g_ampaMF = 0.0; CA3_PCs.g_gaba = 0.0
    CA3_INs = NeuronGroup(num_IN_neurons, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    CA3_INs.vm = Vrest_BC; CA3_INs.g_ampa = 0.0; CA3_INs.g_gaba = 0.0

    wmx_CA3 = wmx_CA3.copy(); wmx_CA3 *= wmx_mult_CA3
    Syn_CA3_PC = Synapses(CA3_PCs, CA3_PCs, "w_exc:1", on_pre="x_ampa+=norm_PC_E*w_exc", delay=delay_PC_E)
    Syn_CA3_PC.connect(i=wmx_CA3.row, j=wmx_CA3.col)
    Syn_CA3_PC.w_exc = wmx_CA3.data
    del wmx_CA3

    Syn_CA3_IN = Synapses(CA3_INs, CA3_PCs, on_pre="x_gaba+=norm_PC_I*w_PC_IN_CA3", delay=delay_PC_I)
    Syn_CA3_IN.connect(p=IN_connection_prob)

    Syn_IN_CA3 = Synapses(CA3_PCs, CA3_INs, on_pre="x_ampa+=norm_BC_E*w_IN_PC_CA3", delay=delay_BC_E)
    Syn_IN_CA3.connect(p=CA3_CA3_connection_prob)

    Syn_CA3_IN_IN = Synapses(CA3_INs, CA3_INs, on_pre="x_gaba+=norm_BC_I*w_IN_IN_CA3", delay=delay_BC_I)
    Syn_CA3_IN_IN.connect(p=IN_connection_prob)

    MF = PoissonGroup(num_CA3_neurons, rate_MF*Hz)
    Syn_CA3_MF = Synapses(MF, CA3_PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF_CA3")
    Syn_CA3_MF.connect(j="i")

    if cue_state >= 0:
        PF = load_PF_starts()
        CA3_place_cell_ID_list = generate_place_cell_ID_list(np.array(list(PF.keys()), dtype=int), np.array(list(PF.values())))
        cue_ids = CA3_place_cell_ID_list[cue_state]
        spike_rates = infield_rate*get_tuning_curve(state_position[cue_state], [PF[key] for key in cue_ids])
        cue_times, cue_local = generate_cue_spikes(spike_rates, duration_ms)
        sort_idx = np.argsort(cue_times)
        cue_src = SpikeGeneratorGroup(len(cue_ids), cue_local[sort_idx].astype(int), cue_times[sort_idx]*second)
        Syn_cue = Synapses(cue_src, CA3_PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF_CA3")
        Syn_cue.connect(i=np.arange(len(cue_ids)), j=cue_ids)

    SM_PC = SpikeMonitor(CA3_PCs); RM_PC = PopulationRateMonitor(CA3_PCs)
    SM_IN = SpikeMonitor(CA3_INs); RM_IN = PopulationRateMonitor(CA3_INs)

    if verbose:
        run(duration_ms*ms, report="text")
    else:
        run(duration_ms*ms)

    if cue_state >= 0:
        return SM_PC, RM_PC, SM_IN, RM_IN, cue_ids
    else: 
        return SM_PC, RM_PC, SM_IN, RM_IN

def find_replay_onset(trial_names, param_lap=5, th=2.0, min_len=150, descending=True, save=True, verbose=True):
    """
    Find, for each trial, the replay-onset lap: the point in training at which
    a spontaneous CA3 replay burst (per `slice_high_activity`) first appears
    in a CA3-only offline simulation.

    Intended for network variants that only save a CA3 recurrent weight matrix
    and have no CA1 layer (e.g. the plain-STDP `STDP<N>` control trials in
    results/linear_reward — see `ca3_offline_simulation`).

    Two-stage search:

    Stage 1 (coarse, over saved laps) — saved laps ("lap_<N>.npz") are tested
    in DESCENDING order (highest lap first), since replay is expected to be
    present once training is far enough along and the goal is to find where
    it disappears going backwards. Each tested lap's `w_CA3_CA3` is loaded, a
    spontaneous CA3-only offline simulation is run for `rest_time` ms using
    the optimised offline network parameters from
    optimization/parameter_lap_<param_lap>.npz, and the resulting CA3
    population rate is checked with `slice_high_activity`. Testing continues
    to the next (lower) lap as long as a burst is detected, and stops at the
    first lap with no detected burst. Let `lap_a` be the lowest lap tested so
    far that DID show a burst, and `lap_b` be the lap where testing stopped
    (the highest lap with no burst). If every tested lap shows a burst,
    `lap_a` is the lowest saved lap and `lap_b` is undefined (the true onset
    is at or below the lowest saved lap; stage 2 is skipped). If the highest
    lap already shows no burst, `lap_a` is undefined and no onset can be
    determined from this trial (stage 2 is skipped).

    Stage 2 (fine, binary search between lap_b and lap_a) — `lap_a` and
    `lap_b` need not be consecutive integers (only some laps are saved), so
    intermediate "laps" are approximated by linearly interpolating the CA3-CA3
    weight matrix between the matrices at `lap_b` (no burst) and `lap_a`
    (burst): for an integer `search_lap` strictly between them,
        w_CA3(search_lap) = w_CA3(lap_b)
            + (w_CA3(lap_a) - w_CA3(lap_b)) * (search_lap - lap_b) / (lap_a - lap_b)
    All other offline network parameters (including wmx_mult_CA3) are kept
    fixed at the values in `ca3_param`. A binary search over `search_lap`
    (assuming burst occurrence is monotonic in the interpolated weight) finds
    the lowest integer `search_lap` at which a burst is detected; this is
    reported as the trial's (interpolated) replay-onset lap.

    Results are saved per trial to <file_dir>/<trial_name>/replay_onset.npz.

    Parameters
    ----------
    trial_names : list of str
        Trial folder names to process (e.g. ["STDP0", ..., "STDP5"]).
    param_lap : int
        Which optimized parameter file (parameter_lap_<param_lap>.npz) to use
        for the offline network's CA3 parameters.
    th, min_len : float
        Passed to `slice_high_activity` (rate threshold [Hz] and minimum
        burst duration [ms]).
    verbose : bool
        Print per-trial / per-lap progress.

    Returns
    -------
    onset_laps : dict {trial_name: int or None}
        Replay-onset lap per trial (interpolated integer lap from stage 2 if
        a [lap_b, lap_a] bracket was found; the lowest saved lap if every
        tested lap showed a burst; None if no burst was ever detected).
    """
    import pickle as pkl

    file_dir = os.path.join(base_path, "results/linear_reward")

    rest_time = 3000

    offline_param_full = np.load(os.path.join(file_dir, "optimization", "parameter_lap_%d.npz" % param_lap))
    ca3_keys = ["w_PC_IN_CA3", "w_IN_PC_CA3", "w_IN_IN_CA3", "wmx_mult_CA3", "w_PC_MF_CA3", "rate_MF"]
    ca3_param = {k: float(offline_param_full[k]) for k in ca3_keys}

    def run_and_detect(wmx_dense):
        SM_PC, RM_PC, SM_IN, RM_IN = ca3_offline_simulation(0, coo_matrix(wmx_dense), rest_time, **ca3_param)
        spike_times, spiking_neurons, rate_PC = preprocess_monitors(SM_PC, RM_PC, calc_ISI=False)
        del SM_PC, RM_PC, SM_IN, RM_IN
        gc.collect()
        slice_idx = slice_high_activity(rate_PC, th=th, min_len=min_len, len_sim=rest_time)
        return bool(slice_idx), slice_idx, spike_times, spiking_neurons

    onset_laps = {}
    for trial_name in trial_names:
        trial_dir = os.path.join(file_dir, trial_name)
        lap_files = sorted(
            (fn for fn in os.listdir(trial_dir) if fn.startswith("lap_") and fn.endswith(".npz")),
            key=lambda fn: int(fn[len("lap_"):-len(".npz")])
        )
        sorted_laps = [int(fn[len("lap_"):-len(".npz")]) for fn in lap_files]
        descending_laps = sorted(sorted_laps, reverse=True) if descending else sorted_laps

        # --- Stage 1: coarse descending scan over saved laps ---
        lap_a, lap_b, wmx_a, wmx_b = None, None, None, None
        before_spk_times, before_spk_neurons, onset_spk_times, onset_spk_neurons = [], [], [], []
        for lap in tqdm(descending_laps, desc="%s (coarse)" % trial_name, disable=not verbose):
            wmx_dense = np.load(os.path.join(trial_dir, "lap_%d.npz" % lap))["w_CA3_CA3"].astype(float)
            detected, _, spk_t, spk_n = run_and_detect(wmx_dense)
            if verbose:
                print("%s: lap %d -> %s" % (trial_name, lap, "burst" if detected else "no burst"))

            if descending:
                if not detected:
                    lap_b, wmx_b = lap, wmx_dense
                    before_spk_times, before_spk_neurons = spk_t, spk_n
                    break
                lap_a, wmx_a = lap, wmx_dense
                onset_spk_times, onset_spk_neurons = spk_t, spk_n
            else:
                if detected:
                    lap_a, wmx_a = lap, wmx_dense
                    break
                lap_b, wmx_b = lap, wmx_dense
                before_spk_times, before_spk_neurons = spk_t, spk_n

        # --- Stage 2: binary search between lap_b (no burst) and lap_a (burst) ---
        onset_lap = None
        if lap_a is not None and lap_b is not None:
            weight_diff = wmx_a - wmx_b
            low, high = lap_b, lap_a
            while high - low > 1:
                mid = (low + high) // 2
                fraction = (mid - lap_b) / (lap_a - lap_b)
                wmx_mid = wmx_b + weight_diff * fraction
                detected, _, spk_t, spk_n = run_and_detect(wmx_mid)
                if verbose:
                    print("%s: search_lap %d (fraction %.3f) -> %s" %
                          (trial_name, mid, fraction, "burst" if detected else "no burst"))
                if detected:
                    high = mid
                    onset_spk_times, onset_spk_neurons = spk_t, spk_n
                else:
                    low = mid
                    before_spk_times, before_spk_neurons = spk_t, spk_n
            onset_lap = high
        elif lap_a is not None and lap_b is None:
            onset_lap = lap_a
            if verbose:
                print("%s: burst detected at every tested lap; onset at or below lowest saved lap %d" %
                      (trial_name, lap_a))
        elif verbose:
            print("%s: no replay burst detected at the highest tested lap %d" % (trial_name, descending_laps[0]))
        if save:
            f_out = os.path.join(trial_dir, "replay_onset.npz")
            with open(f_out, 'wb') as file:
                pkl.dump(
                    {"onset_lap": onset_lap if onset_lap is not None else -1,
                    "found": onset_lap is not None,
                    "before_spk_times": before_spk_times,
                    "before_spk_neurons": before_spk_neurons,
                    "onset_spk_times": onset_spk_times,
                    "onset_spk_neurons": onset_spk_neurons,
                    },
                    file)

        onset_laps[trial_name] = onset_lap

    return onset_laps

def simulate_potentiated_replay(trial_number, potentiated_states=list(range(7)), potentiation_factor=1.15,
                                 cue_state=0, duration_ms=20000, 
                                 folder_fmt="trial%d", online_fname="lap_10.npz"):
    """
    Selective-potentiation CA3 replay (T-maze only) for the Supple_replay_Carey
    supplementary figure.

    For each trial, potentiates every recurrent CA3-CA3 synapse whose both
    endpoints lie in `potentiated_states` by `potentiation_factor`
    (`Tmaze_functions.build_potentiated_weight_matrix`), then runs a
    cue-seeded CA3-only offline replay (`ca3_offline_simulation`) once with
    the potentiated matrix and once with the original (control) matrix. Both
    spike trains are saved next to the online weights in
    `data_dir/<folder_fmt % trial>/`.

    Parameters
    ----------
    trial_number : int
    data_dir : str
        Directory holding one subfolder per trial (`folder_fmt % trial`),
        each containing `online_fname` (online-learned w_CA3_CA3) and
        `PF_peak_data.pkl` (CA3 place fields).
    potentiated_states : list of int
        0-indexed T-maze states whose mutual recurrent synapses are potentiated.
    potentiation_factor : float
    cue_state : int
        0-indexed T-maze state at which the replay cue is injected.
    duration_ms : float
    w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3, wmx_mult_CA3, w_PC_MF_CA3, rate_MF :
        Offline network parameters (see `offline_simulation`).
    potentiated_wmx_mult_scale : float
        Extra scale applied to `wmx_mult_CA3` for the potentiated run only
        (compensates for the higher overall drive after potentiation).
    folder_fmt : str
        Per-trial subfolder name pattern.
    online_fname : str
        Filename (within each trial folder) holding the online-learned weights.
    """
    import pickle
    from Tmaze_functions import build_potentiated_weight_matrix, data_path, load_PF_starts, sample_spatial_points, get_tuning_curve

    simulation_time = 20000 # [ms]
    mode = 1
    PF_dict = load_PF_starts()
    place_peak_position = sample_spatial_points()

    offline_param = np.load(os.path.join(data_path, "optimization/parameter_lap_10.npz"))
    ca3_keys = ["w_PC_IN_CA3", "w_IN_PC_CA3", "w_IN_IN_CA3", "wmx_mult_CA3", "w_PC_MF_CA3", "rate_MF"]
    ca3_param = {k: float(offline_param[k]) for k in ca3_keys}
    
    for trial in range(trial_number):
        trial_dir = os.path.join(data_path, folder_fmt % trial)
        print("Trial %d/%d — %s" % (trial+1, trial_number, trial_dir))

        online_data = np.load(os.path.join(trial_dir, online_fname), allow_pickle=True)
        W_raw = online_data["w_CA3_CA3"].copy()
        W_CA1 = online_data["w_CA3_CA1"].copy()
        connectivity = np.load(os.path.join(trial_dir, "simulation_information.npz"))["connectivity_CA3_CA1"]
        
        W_pot, _, n_pot = build_potentiated_weight_matrix(W_raw, PF_dict, potentiated_states, potentiation_factor)
        n_total = int((W_raw > 0).sum())
        print("  Potentiated synapses: %d / %d (%.1f%%)" % (n_pot, n_total, 100*n_pot/n_total))

        # SM_PC, RM_PC, SM_IN, RM_IN, cue_ids = ca3_offline_simulation(
        #     mode, coo_matrix(W_pot), duration_ms, w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3, wmx_mult_CA3*potentiated_wmx_mult_scale, w_PC_MF_CA3, rate_MF, cue_state=cue_state)

        SM_PC_CA3, _, RM_PC_CA3, _, SM_PC_CA1, _, RM_PC_CA1, _, _, _, _, _, _, _, cue_ids = offline_simulation(mode, coo_matrix(W_pot), coo_matrix(W_CA1),
                                                                                                                    cue_state, PF_dict, **offline_param,
                                                                                                                    simulation_time=simulation_time)

        spike_times_CA3_PC, spiking_neurons_CA3_PC, _ = preprocess_monitors(SM_PC_CA3, RM_PC_CA3, calc_ISI=False)
        spike_times_CA1_PC, spiking_neurons_CA1_PC, _ = preprocess_monitors(SM_PC_CA1, RM_PC_CA1, calc_ISI=False)

        print("  CA3 PC spikes: %d" % (SM_PC_CA3.num_spikes))
        print("Now training network with generated spike train...")

        updated_CA1 = update_STDP(W_CA1, spiking_neurons_CA3_PC, spiking_neurons_CA1_PC,
                    spike_times_CA3_PC, spike_times_CA1_PC,
                    connectivity, np.array([simulation_time]))
        updated_CA1 = updated_CA1[0]

        CA1_activity = np.zeros((place_peak_position.shape[0],updated_CA1.shape[1]))
        
        for ll in range(place_peak_position.shape[0]):
            position = place_peak_position[ll]
            CA3_FR = get_tuning_curve(position, list(PF_dict.values()))*infield_rate*theta_mod_factor
            CA1_activity[ll,:] = input_driven_rate(np.arange(updated_CA1.shape[1]), CA3_FR, updated_CA1)

        np.savez(os.path.join(trial_dir, "CA3_replay_potentiated_cue1.npz"),
                         updated_CA1 = updated_CA1[0], CA1_activity=CA1_activity,
                         spike_times_CA3_PC=spike_times_CA3_PC, spiking_neurons_CA3_PC=spiking_neurons_CA3_PC,
                         spike_times_CA1_PC=spike_times_CA1_PC, spiking_neurons_CA1_PC=spiking_neurons_CA1_PC,
                         cue_neuron_ids=cue_ids, cue_state=cue_state, simulation_time=simulation_time,
                         potentiated_states=np.array(potentiated_states), potentiation_factor=potentiation_factor)

        del W_pot, updated_CA1, SM_PC_CA3, RM_PC_CA3, SM_PC_CA1, RM_PC_CA1, CA1_activity

        # SM_PC_CA3, _, RM_PC_CA3, _, SM_PC_CA1, _, RM_PC_CA1, _, _, _, _, _, _, _, cue_ids = offline_simulation(mode, coo_matrix(W_raw), coo_matrix(W_CA1),
        #                                                                                                                     cue_state, PF_dict, **offline_param,
        #                                                                                                                     simulation_time=simulation_time)
        
        # spike_times_CA3_PC, spiking_neurons_CA3_PC, _ = preprocess_monitors(SM_PC_CA3, RM_PC_CA3, calc_ISI=False)
        # spike_times_CA1_PC, spiking_neurons_CA1_PC, _ = preprocess_monitors(SM_PC_CA1, RM_PC_CA1, calc_ISI=False)
        SM_PC_CA3, RM_PC_CA3, _, _, cue_ids = ca3_offline_simulation(
            mode, coo_matrix(W_raw), simulation_time, **ca3_param, cue_state=cue_state)
        spike_times_CA3_PC, spiking_neurons_CA3_PC, _ = preprocess_monitors(SM_PC_CA3, RM_PC_CA3, calc_ISI=False)
        
        print("  [control] CA3 PC spikes: %d " % (SM_PC_CA3.num_spikes))
        # print("Now training network with generated spike train...")

        # updated_CA1 = update_STDP(W_CA1, spiking_neurons_CA3_PC, spiking_neurons_CA1_PC,
        #             spike_times_CA3_PC, spike_times_CA1_PC,
        #             connectivity, np.array([simulation_time]))
        # updated_CA1 = updated_CA1[0]

        # CA1_activity = np.zeros((place_peak_position.shape[0],updated_CA1.shape[1]))
        
        # for ll in range(place_peak_position.shape[0]):
        #     position = place_peak_position[ll]
        #     CA3_FR = get_tuning_curve(position, list(PF_dict.values()))*infield_rate*theta_mod_factor
        #     CA1_activity[ll,:] = input_driven_rate(np.arange(updated_CA1.shape[1]), CA3_FR, updated_CA1)

        np.savez(os.path.join(trial_dir, "CA3_replay_ctrl_cue1.npz"),
                            # updated_CA1 = updated_CA1[0], CA1_activity=CA1_activity,
                            spike_times_CA3_PC=spike_times_CA3_PC, spiking_neurons_CA3_PC=spiking_neurons_CA3_PC,
                            # spike_times_CA1_PC=spike_times_CA1_PC, spiking_neurons_CA1_PC=spiking_neurons_CA1_PC,
                            cue_neuron_ids=cue_ids, cue_state=cue_state, simulation_time=simulation_time)
        del W_raw, SM_PC_CA3, RM_PC_CA3


        # SM_PC, RM_PC, SM_IN, RM_IN, cue_ids = ca3_offline_simulation(
        #     mode, coo_matrix(W_raw), duration_ms, w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3, wmx_mult_CA3, w_PC_MF_CA3, rate_MF, cue_state=cue_state)
        # spike_times_PC, spiking_neurons_PC, rate_PC = preprocess_monitors(SM_PC, RM_PC, calc_ISI=False)
        # spike_times_IN, spiking_neurons_IN, _ = preprocess_monitors(SM_IN, RM_IN, calc_ISI=False)
        # print("  [control] CA3 PC spikes: %d   CA3 IN spikes: %d" % (SM_PC.num_spikes, SM_IN.num_spikes))
        # np.savez(os.path.join(trial_dir, "CA3_replay_ctrl_cue1.npz"),
        #          spike_times_CA3_PC=spike_times_PC, spiking_neurons_CA3_PC=spiking_neurons_PC,
        #          spike_times_CA3_IN=spike_times_IN, spiking_neurons_CA3_IN=spiking_neurons_IN,
        #          cue_neuron_ids=cue_ids, cue_state_1indexed=cue_state+1, potentiation_factor=1.0)
        # del W_raw, PF_dict, SM_PC, RM_PC, SM_IN, RM_IN

def ca3_offline_fatigued(wmx_CA3, PF, fatigue_states, cue_state, duration_ms, w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3, wmx_mult_CA3, w_PC_MF_CA3, rate_MF,
                          a_PC_fatigued=-0.01*nS, b_PC_fatigued=40000*pA, w_init_fatigue=10000*pA, verbose=False):
    """
    Run a CA3-only offline Brian2 simulation (no CA1) in which place cells in
    `fatigue_states` carry elevated per-neuron spike-triggered adaptation
    (heterogeneous `eqs_PC_het`), modelling post-burst neuronal fatigue, while
    a Poisson cue is injected at `cue_state`.

    Linear-shock-only variant used for the Wu et al. replay-fatigue
    supplementary figure (Supple_replay_Wu.ipynb / generate_data.ipynb). Pass
    `a_PC_fatigued=a_PC, b_PC_fatigued=b_PC, w_init_fatigue=0*pA` for a
    matched control run with fatigue disabled.

    Parameters
    ----------
    wmx_CA3 : scipy.sparse.coo_matrix
        CA3-CA3 recurrent weight matrix.
    PF : dict {neuron_id: [row, col]}
        CA3 place field peak positions.
    fatigue_states : list of int
        0-indexed linear-track states whose neurons get elevated adaptation.
    cue_state : int
        0-indexed state at which the replay cue is injected.
    duration_ms : float
        Simulation duration [ms].
    w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3, wmx_mult_CA3, w_PC_MF_CA3, rate_MF :
        Same meaning as in `offline_simulation`.
    a_PC_fatigued, b_PC_fatigued, w_init_fatigue : Brian2 quantities
        Subthreshold adaptation conductance, spike-triggered adaptation
        current, and initial adaptation current for the fatigued neurons.
    verbose : bool
        If True, Brian2 reports simulation progress.

    Returns
    -------
    SM_PC, RM_PC, SM_IN, RM_IN : Brian2 SpikeMonitor/PopulationRateMonitor for CA3 PCs/INs.
    fatigue_ids, cue_ids : np.ndarray of global CA3 neuron IDs.
    """
    from linear_shock_variables import num_CA3_neurons, state_position
    from linear_shock_functions import generate_place_cell_ID_list, get_tuning_curve

    start_scope()

    CA3_PCs = NeuronGroup(num_CA3_neurons, model=eqs_PC_het, threshold="vm>spike_th_PC",
                      reset="vm=Vreset_PC; w+=b_adapt", refractory=tref_PC, method="exponential_euler")
    CA3_PCs.vm = Vrest_PC; CA3_PCs.g_ampa = 0.0; CA3_PCs.g_ampaMF = 0.0; CA3_PCs.g_gaba = 0.0; CA3_PCs.w = 0.0*pA
    CA3_INs = NeuronGroup(num_IN_neurons, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    CA3_INs.vm = Vrest_BC; CA3_INs.g_ampa = 0.0; CA3_INs.g_gaba = 0.0

    CA3_PCs.a_adapt = a_PC; CA3_PCs.b_adapt = b_PC
    CA3_place_cell_ID_list = generate_place_cell_ID_list(np.array(list(PF.keys()), dtype=int), np.array(list(PF.values())))
    fatigue_ids = np.concatenate([CA3_place_cell_ID_list[s] for s in fatigue_states])
    CA3_PCs.a_adapt[fatigue_ids] = a_PC_fatigued
    CA3_PCs.b_adapt[fatigue_ids] = b_PC_fatigued
    CA3_PCs.w[fatigue_ids] = w_init_fatigue

    wmx_CA3 = wmx_CA3.copy(); wmx_CA3 *= wmx_mult_CA3
    Syn_CA3_PC = Synapses(CA3_PCs, CA3_PCs, "w_exc:1", on_pre="x_ampa+=norm_PC_E*w_exc", delay=delay_PC_E)
    Syn_CA3_PC.connect(i=wmx_CA3.row, j=wmx_CA3.col)
    Syn_CA3_PC.w_exc = wmx_CA3.data
    del wmx_CA3

    Syn_CA3_IN = Synapses(CA3_INs, CA3_PCs, on_pre="x_gaba+=norm_PC_I*w_PC_IN_CA3", delay=delay_PC_I)
    Syn_CA3_IN.connect(p=IN_connection_prob)

    Syn_IN_CA3 = Synapses(CA3_PCs, CA3_INs, on_pre="x_ampa+=norm_BC_E*w_IN_PC_CA3", delay=delay_BC_E)
    Syn_IN_CA3.connect(p=CA3_CA3_connection_prob)

    Syn_CA3_IN_IN = Synapses(CA3_INs, CA3_INs, on_pre="x_gaba+=norm_BC_I*w_IN_IN_CA3", delay=delay_BC_I)
    Syn_CA3_IN_IN.connect(p=IN_connection_prob)

    MF = PoissonGroup(num_CA3_neurons, rate_MF*Hz)
    Syn_CA3_MF = Synapses(MF, CA3_PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF_CA3")
    Syn_CA3_MF.connect(j="i")

    cue_ids = CA3_place_cell_ID_list[cue_state]
    spike_rates = infield_rate*get_tuning_curve(state_position[cue_state], [PF[key] for key in cue_ids])
    cue_times, cue_local = generate_cue_spikes(spike_rates, duration_ms)
    sort_idx = np.argsort(cue_times)
    cue_src = SpikeGeneratorGroup(len(cue_ids), cue_local[sort_idx].astype(int), cue_times[sort_idx]*second)
    Syn_cue = Synapses(cue_src, CA3_PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF_CA3")
    Syn_cue.connect(i=np.arange(len(cue_ids)), j=cue_ids)

    SM_PC = SpikeMonitor(CA3_PCs); RM_PC = PopulationRateMonitor(CA3_PCs)
    SM_IN = SpikeMonitor(CA3_INs); RM_IN = PopulationRateMonitor(CA3_INs)

    if verbose:
        run(duration_ms*ms, report="text")
    else:
        run(duration_ms*ms)

    return SM_PC, RM_PC, SM_IN, RM_IN, fatigue_ids, cue_ids


def simulate_EC_replay(trial_number, cue_state=3, duration_ms=5000, EC_coeff=1,
                              folder_fmt="trial%d", online_fname="lap_4.npz"):
    """

    Parameters
    ----------
    trial_number : int
    data_dir : str
        Directory holding one subfolder per trial (`folder_fmt % trial`),
        each containing `online_fname` (online-learned w_CA3_CA3) and
        `PF_peak_data.pkl` (CA3 place fields).
    cue_state : int
        0-indexed state at which the replay cue is injected.
    duration_ms : float
    rate_coeff : float
        Coefficient scaling rate_MF.
    folder_fmt : str
        Per-trial subfolder name pattern.
    online_fname : str
        Filename (within each trial folder) holding the online-learned weights.
    """
    from linear_shock_functions import data_path

    offline_param_full = np.load(os.path.join(data_path, "optimization", "parameter_lap_4.npz"))
    ca3_keys = ["w_PC_IN_CA3", "w_IN_PC_CA3", "w_IN_IN_CA3", "wmx_mult_CA3", "w_PC_MF_CA3", "rate_MF"]
    ca3_param = {k: float(offline_param_full[k]) for k in ca3_keys}
    ca3_param["rate_MF"] *= EC_coeff

    for trial in range(trial_number):
        trial_dir = os.path.join(data_path, folder_fmt % trial)
        print("Trial %d/%d — %s" % (trial+1, trial_number, trial_dir))

        online_data = np.load(os.path.join(trial_dir, online_fname), allow_pickle=True)
        w_CA3_CA3 = online_data["w_CA3_CA3"]

        SM_PC, RM_PC, SM_IN, RM_IN, cue_ids = ca3_offline_simulation(2, coo_matrix(w_CA3_CA3), duration_ms,
                                                                     **ca3_param, cue_state=cue_state, verbose=False)
                                                                    #  w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3,
                                                                    #  wmx_mult_CA3, w_PC_MF_CA3*EC_coeff, rate_MF,
                                                                     

        spike_times_PC, spiking_neurons_PC, rate_PC = preprocess_monitors(SM_PC, RM_PC, calc_ISI=False)

        np.savez(os.path.join(trial_dir, "CA3_replay_EC_%.1f.npz"%EC_coeff),
                 spike_times_CA3_PC=spike_times_PC, spiking_neurons_CA3_PC=spiking_neurons_PC, 
                 rate_PC=rate_PC, EC_coeff=EC_coeff, cue_neuron_ids=cue_ids)
        del SM_PC, RM_PC, SM_IN, RM_IN


def simulate_fatigued_replay(trial_number, fatigue_states=[0, 1, 2], cue_state=3, duration_ms=20000,
                              w_PC_IN_CA3=1.646, w_IN_PC_CA3=1.715, w_IN_IN_CA3=14.732,
                              wmx_mult_CA3=2.306, w_PC_MF_CA3=15.843, rate_MF=13.038,
                              a_PC_fatigued=-0.01*nS, b_PC_fatigued=40000*pA, w_init_fatigue=10000*pA,
                              folder_fmt="trial%d", online_fname="lap_4.npz"):
    """
    Selective-fatigue CA3 replay (linear shock only) for the Supple_replay_Wu
    supplementary figure.

    For each trial, runs a cue-seeded CA3-only offline replay
    (`ca3_offline_fatigued`) with elevated per-neuron adaptation applied to
    `fatigue_states`, saved as `CA3_replay_wu_fatigue.npz`.
    A matched control run (fatigue disabled) is additionally simulated for the
    trials listed in `control_trials`, saved as `CA3_replay_wu_ctrl.npz`.

    Parameters
    ----------
    trial_number : int
    data_dir : str
        Directory holding one subfolder per trial (`folder_fmt % trial`),
        each containing `online_fname` (online-learned w_CA3_CA3) and
        `PF_peak_data.pkl` (CA3 place fields).
    fatigue_states : list of int
        0-indexed linear-track states whose neurons are fatigued.
    cue_state : int
        0-indexed state at which the replay cue is injected.
    duration_ms : float
    w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3, wmx_mult_CA3, w_PC_MF_CA3, rate_MF :
        Offline network parameters (see `offline_simulation`).
    a_PC_fatigued, b_PC_fatigued, w_init_fatigue : Brian2 quantities
        Fatigue parameters for the fatigued run (see `ca3_offline_fatigued`).
    control_trials : list of int or None
        Trials to additionally simulate with fatigue disabled. None (default)
        runs no control trials.
    ctrl_seed : int
        Random seed reset right before each control run, for reproducibility.
    folder_fmt : str
        Per-trial subfolder name pattern.
    online_fname : str
        Filename (within each trial folder) holding the online-learned weights.
    """
    import pickle
    from linear_shock_functions import load_PF_starts, data_path
    PF_dict = load_PF_starts()

    for trial in range(trial_number):
        trial_dir = os.path.join(data_path, folder_fmt % trial)
        print("Trial %d/%d — %s" % (trial+1, trial_number, trial_dir))

        online_data = np.load(os.path.join(trial_dir, online_fname), allow_pickle=True)
        w_CA3_CA3 = online_data["w_CA3_CA3"]
        
        SM_PC, RM_PC, SM_IN, RM_IN, fatigue_ids, cue_ids = ca3_offline_fatigued(
            coo_matrix(w_CA3_CA3), PF_dict, fatigue_states, cue_state, duration_ms,
            w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3, wmx_mult_CA3, w_PC_MF_CA3, rate_MF,
            a_PC_fatigued=a_PC_fatigued, b_PC_fatigued=b_PC_fatigued, w_init_fatigue=w_init_fatigue)
        spike_times_PC, spiking_neurons_PC, rate_PC = preprocess_monitors(SM_PC, RM_PC, calc_ISI=False)
        spike_times_IN, spiking_neurons_IN, _ = preprocess_monitors(SM_IN, RM_IN, calc_ISI=False)
        print("  CA3 PC spikes: %d   CA3 IN spikes: %d" % (SM_PC.num_spikes, SM_IN.num_spikes))
        np.savez(os.path.join(trial_dir, "CA3_replay_wu_fatigue.npz"),
                 spike_times_CA3_PC=spike_times_PC, spiking_neurons_CA3_PC=spiking_neurons_PC, rate_CA3_PC=rate_PC,
                 spike_times_CA3_IN=spike_times_IN, spiking_neurons_CA3_IN=spiking_neurons_IN,
                 fatigue_neuron_ids=fatigue_ids, cue_neuron_ids=cue_ids,
                 fatigue_state_1indexed=np.array([s+1 for s in fatigue_states]), cue_state_1indexed=cue_state+1,
                 sim_duration_ms=duration_ms,
                 w_PC_IN_CA3=w_PC_IN_CA3, w_IN_PC_CA3=w_IN_PC_CA3, w_IN_IN_CA3=w_IN_IN_CA3,
                 wmx_mult_CA3=wmx_mult_CA3, w_PC_MF_CA3=w_PC_MF_CA3, rate_MF=rate_MF,
                 a_PC_fatigued_nS=float(a_PC_fatigued/nS), b_PC_fatigued_pA=float(b_PC_fatigued/pA),
                 w_init_fatigue_pA=float(w_init_fatigue/pA))
        del SM_PC, RM_PC, SM_IN, RM_IN

        SM_PC, RM_PC, SM_IN, RM_IN, _, cue_ids = ca3_offline_fatigued(
            coo_matrix(w_CA3_CA3), PF_dict, fatigue_states, cue_state, duration_ms,
            w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3, wmx_mult_CA3, w_PC_MF_CA3, rate_MF,
            a_PC_fatigued=a_PC, b_PC_fatigued=b_PC, w_init_fatigue=0*pA)
        spike_times_PC, spiking_neurons_PC, rate_PC = preprocess_monitors(SM_PC, RM_PC, calc_ISI=False)
        spike_times_IN, spiking_neurons_IN, _ = preprocess_monitors(SM_IN, RM_IN, calc_ISI=False)
        print("  [control] CA3 PC spikes: %d   CA3 IN spikes: %d" % (SM_PC.num_spikes, SM_IN.num_spikes))
        np.savez(os.path.join(trial_dir, "CA3_replay_wu_ctrl.npz"),
                    spike_times_CA3_PC=spike_times_PC, spiking_neurons_CA3_PC=spiking_neurons_PC, rate_CA3_PC=rate_PC,
                    spike_times_CA3_IN=spike_times_IN, spiking_neurons_CA3_IN=spiking_neurons_IN,
                    cue_neuron_ids=cue_ids, cue_state_1indexed=cue_state+1, fatigue_applied=False,
                    sim_duration_ms=duration_ms, w_PC_IN_CA3=w_PC_IN_CA3, w_IN_PC_CA3=w_IN_PC_CA3, w_IN_IN_CA3=w_IN_IN_CA3,
                    wmx_mult_CA3=wmx_mult_CA3, w_PC_MF_CA3=w_PC_MF_CA3, rate_MF=rate_MF)
        del SM_PC, RM_PC, SM_IN, RM_IN

        del w_CA3_CA3, online_data

def CA3_find_optimal_parameters(target_trial, target_lap, offspring_size=50, max_ngen=10):
    """
    Optimise the CA3 recurrent network's offline parameters using a DEAP
    evolutionary algorithm.

    CA3-only counterpart of `find_optimal_parameters`: optimises just the CA3
    recurrent network (no CA1 population, no CA3→CA1 feedforward projection).
    Intended for network variants that only maintain a CA3 recurrent weight
    matrix and have no CA1 layer (e.g. the plain-STDP `STDP<N>` control
    trials — see `ca3_offline_simulation`, `find_replay_onset`).

    The fitness function (`ca3NetworkEvaluator.init_simulator_and_evaluate_with_lists`)
    runs `ca3_offline_simulation` and scores the resulting CA3
    population activity on six SWR criteria:
        rippleE / rippleI       — ripple-band (150–220 Hz) power in PC / IN rate
        no_gamma_peakI          — absence of gamma-band peak in IN rate
        ripple_ratioE / ratioI  — ripple-to-gamma power ratio
        low_rateE               — mean PC rate close to `background_rate` Hz

    Parameters to optimise (with search bounds):
        w_PC_IN_CA3   — excitatory weight onto interneurons
        w_IN_PC_CA3   — inhibitory weight onto place cells
        w_IN_IN_CA3   — inhibitory weight between interneurons
        wmx_mult_CA3  — global scale factor for the learned weight matrix
        w_PC_MF_CA3   — mossy-fibre drive weight
        rate_MF       — mossy-fibre background rate [Hz]

    Best parameters are saved to results/<task>/optimization/parameter_lap_<N>.npz.

    Parameters
    ----------
    target_trial : int
        Which trial's (STDP<target_trial>) weight matrix to use for optimisation.
    target_lap : int
        Which lap's weight matrix to use.
    offspring_size : int
        Population size for the evolutionary algorithm (≥ num_workers is ideal).
    max_ngen : int
        Maximum number of evolutionary generations.

    Returns
    -------
    best : list of float
        Optimal parameter values in the order defined by `optconf`.
    """

    # if mode == 0:
    file_dir = os.path.join(base_path,"results/linear_reward")
    # elif mode == 1:
    #     file_dir = os.path.join(base_path,"results/Tmaze")
    # elif mode == 2:
    #     file_dir = os.path.join(base_path,"results/linear_shock")

    foldername = "STDP%d"%target_trial
    online_result = np.load(os.path.join(file_dir,foldername,"lap_%d.npz"%target_lap))
    weightmx_CA3_CA3 = online_result["w_CA3_CA3"]
    wmx_CA3 = coo_matrix(weightmx_CA3_CA3)

    prev_coeff = np.load(os.path.join(file_dir, "optimization", "parameter_lap_5.npz"))

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    os.makedirs(os.path.join(file_dir,"optimization"), exist_ok=True)
    cp_f_name = os.path.join(file_dir, "optimization", "checkpoint_lap_%d.pkl" % target_lap)
    params_fname = os.path.join(file_dir, "optimization", "parameter_lap_%d.npz" % target_lap)
    optconf = [("w_PC_IN_CA3", 0.1, 2.0),
                ("w_IN_PC_CA3", 0.1, 2.0),
                ("w_IN_IN_CA3", 1.0, 8.0),
                ("wmx_mult_CA3", 0.5, 2.0),
                ("w_PC_MF_CA3", 15.0, 25.0),
                ("rate_MF", 5.0, 20.0)]

    pnames = [name for name, _, _ in optconf]

    # Create multiprocessing pool for parallel evaluation of fitness function
    n_proc = np.min([offspring_size, mp.cpu_count()-1])
    pool = mp.Pool(processes=n_proc)
    # Create BluePyOpt optimization and run

    evaluator = ca3NetworkEvaluator(0, wmx_CA3, optconf)
    opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=offspring_size, map_function=pool.map,
                                                eta=20, mutpb=0.3, cxpb=0.7)

    print("Started running %i simulations on %i cores..." % (offspring_size*max_ngen, n_proc))
    _, hof, _, _ = opt.run(max_ngen=max_ngen, cp_filename=cp_f_name,verbose=False)
    del pool, opt

    best = hof[0]
    for pname, value in zip(pnames, best):
        print("%s = %.3f" % (pname, value))
    _ = evaluator.evaluate_with_lists(best)
    np.savez(params_fname, **dict(zip(pnames, best)))
    return best

def find_optimal_parameters(mode, target_trial, target_lap, pause_state=-1, offspring_size=50, max_ngen=10):
    """
    Optimise the offline network parameters using a DEAP evolutionary algorithm.

    The fitness function (`NetworkEvaluator.init_simulator_and_evaluate_with_lists`)
    runs `offline_simulation` and scores the resulting population activity on
    six SWR criteria for each of CA3 and CA1 (12 objectives total):
        rippleE / rippleI       — ripple-band (150–220 Hz) power in PC / IN rate
        no_gamma_peakI          — absence of gamma-band peak in IN rate
        ripple_ratioE / ratioI  — ripple-to-gamma power ratio
        low_rateE               — mean PC rate close to `background_rate` Hz

    Parameters to optimise (with search bounds):
        w_PC_IN_CA3/CA1   — excitatory weight onto interneurons
        w_IN_PC_CA3/CA1   — inhibitory weight onto place cells
        w_IN_IN_CA3/CA1   — inhibitory weight between interneurons
        wmx_mult_CA3/CA1  — global scale factor for the learned weight matrix
        w_PC_MF_CA3       — mossy-fibre drive weight
        rate_MF           — mossy-fibre background rate [Hz]

    Best parameters are saved to results/<task>/optimization/parameter_lap_<N>.npz.

    Parameters
    ----------
    mode : int
        Task selector.
    target_trial : int
        Which trial's weight matrices to use for optimisation.
    target_lap : int
        Which lap's weight matrices to use.
    pause_state : int
        Maze state index for cue injection (-1 = no cue; spontaneous replay).
    offspring_size : int
        Population size for the evolutionary algorithm (≥ num_workers is ideal).
    max_ngen : int
        Maximum number of evolutionary generations.

    Returns
    -------
    best : list of float
        Optimal parameter values in the order defined by `optconf`.
    """
    
    if mode == 0:
        from linear_reward_functions import load_PF_starts
        file_dir = os.path.join(base_path,"results/linear_reward")
    elif mode == 1:
        from Tmaze_functions import load_PF_starts
        file_dir = os.path.join(base_path,"results/Tmaze")
    elif mode == 2:
        from linear_shock_functions import load_PF_starts
        file_dir = os.path.join(base_path,"results/linear_shock")

    foldername = "trial%d"%target_trial
    online_result = np.load(os.path.join(file_dir,foldername,"lap_%d.npz"%target_lap))
    weightmx_CA3_CA3 = online_result["w_CA3_CA3"]
    wmx_CA3 = coo_matrix(weightmx_CA3_CA3)

    weightmx_CA3_CA1 = online_result["w_CA3_CA1"]
    wmx_CA1 = coo_matrix(weightmx_CA3_CA1)

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    os.makedirs(os.path.join(file_dir,"optimization"), exist_ok=True)
    cp_f_name = os.path.join(file_dir, "optimization", "checkpoint_lap_%d.pkl" % target_lap)
    params_fname = os.path.join(file_dir, "optimization", "parameter_lap_%d.npz" % target_lap)
    optconf = [("w_PC_IN_CA3", 0.5, 20.0),
                ("w_IN_PC_CA3", 0.2, 5.0),
                ("w_IN_IN_CA3", 1.0, 15.0),
                ("wmx_mult_CA3", 1.0, 2.5),
                ("w_PC_MF_CA3", 0.5, 40.0),
                ("rate_MF", 5.0, 12.0),
                ("w_PC_IN_CA1", 1.0, 10.0),
                ("w_IN_PC_CA1", 0.2, 5.0),
                ("w_IN_IN_CA1", 1.0, 15.0),
                ("wmx_mult_CA1", 1.0, 2.5)]

    pnames = [name for name, _, _ in optconf]


    # Create multiprocessing pool for parallel evaluation of fitness function
    n_proc = np.min([offspring_size, mp.cpu_count()-1])
    pool = mp.Pool(processes=n_proc)
    # Create BluePyOpt optimization and run

    CA3_PF = load_PF_starts()

    evaluator = NetworkEvaluator(mode, wmx_CA3, wmx_CA1, pause_state, CA3_PF, optconf)
    opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=offspring_size, map_function=pool.map,
                                                eta=20, mutpb=0.3, cxpb=0.7)

    print("Started running %i simulations on %i cores..." % (offspring_size*max_ngen, n_proc))
    _, hof, _, _ = opt.run(max_ngen=max_ngen, cp_filename=cp_f_name, verbose=False)
    del pool, opt

    best = hof[0]
    for pname, value in zip(pnames, best):
        print("%s = %.3f" % (pname, value))
    _ = evaluator.evaluate_with_lists(best)
    np.savez(params_fname, **dict(zip(pnames, best)))
    return best

def detect_replay_type(mode, target_lap, trial_number):
    """
    Classify offline CA3 replay events by trajectory type (T-maze only).

    Loads saved CA3 spike data and calls `analyse_replay_type` to check
    which of the predefined replay trajectories (e.g., left-arm, right-arm,
    or full-loop) each detected SWR event corresponds to.  Results are
    pickled per replay type in the trial directory.

    Parameters
    ----------
    mode : int
        Must be 1 (T-maze); raises ValueError otherwise.
    target_lap : int
        Lap whose saved offline spikes are analysed.
    trial_number : int
        Number of trials to process.
    """

    if mode == 1:
        from Tmaze_functions import analyse_replay_type
        file_dir = os.path.join(base_path,"results/Tmaze")
    else: raise ValueError("This function is only supported to analyze T-maze learning scenario!")

    for trial in tqdm(range(trial_number),desc="Trial: "):
        foldername = "trial%d"%trial
        save_path = os.path.join(file_dir,foldername)
        ff = np.load(os.path.join(save_path,"CA3_replay_lap_%d_pause_4.npz"%(target_lap)))
        spike_times = ff['spike_times_CA3_PC']; spiking_neurons = ff['spiking_neurons_CA3_PC']; rate = ff['rate_CA3_PC']
        del ff
        
        analyse_replay_type(spike_times, spiking_neurons, rate, save_path=save_path)

# ---------------------------------------------------------------------------
# REVISED FROM ca3net (https://github.com/KaliLab/ca3net)
# Original: single-population (CA3 only) recurrent network with SWR optimisation.
# Changes:  added a second CA1 population (CA1_PCs + CA1_INs) driven by CA3→CA1
#           weight matrix; added optional pause-state cue input (PoissonGroup)
#           seeded from place-field tuning curves at a specific maze location.
# ---------------------------------------------------------------------------
def offline_simulation(mode, wmx_CA3, wmx_CA1, pause_state, PF, w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3, wmx_mult_CA3, w_PC_MF_CA3, rate_MF, w_PC_IN_CA1, w_IN_PC_CA1, w_IN_IN_CA1, wmx_mult_CA1, simulation_time=None, verbose=False):
    """
    Run a single offline Brian2 simulation and return all monitors.

    The network consists of four Brian2 NeuronGroups:
        CA3_PCs / CA3_INs — CA3 place cells and fast-spiking interneurons
        CA1_PCs / CA1_INs — CA1 place cells and fast-spiking interneurons

    Driving inputs:
        MF   — mossy-fibre background (PoissonGroup, `rate_MF` Hz to all CA3_PCs)
        cue  — location-specific cue (PoissonGroup, rates from place-field tuning
                curves at `pause_state`); only added when pause_state >= 0.

    Synaptic connections built:
        CA3_PCs → CA3_PCs  via wmx_CA3 (scaled by wmx_mult_CA3)
        CA3_PCs → CA3_INs  (random, CA3_CA3_connection_prob)
        CA3_INs → CA3_PCs  (random, IN_connection_prob)
        CA3_INs → CA3_INs  (random, IN_connection_prob)
        CA3_PCs → CA1_PCs  via wmx_CA1 (scaled by wmx_mult_CA1)
        CA1_PCs → CA1_INs  (random, CA3_CA1_connection_prob)
        CA1_INs → CA1_PCs  (random, IN_connection_prob)
        CA1_INs → CA1_INs  (random, IN_connection_prob)

    Parameters
    ----------
    wmx_CA3, wmx_CA1 : scipy.sparse.coo_matrix
        Learned online weight matrices (sparse COO format).
    pause_state : int
        Maze state to inject as cue (≥0) or -1 for no cue.
    PF : dict {neuron_id: position}
        CA3 place field peak positions.
    w_PC_IN_*, w_IN_PC_*, w_IN_IN_* : float
        Scalar synaptic weights (uniform within population pair).
    wmx_mult_CA3/CA1 : float
        Global multiplier applied to the sparse weight matrices before simulation.
    w_PC_MF_CA3 : float
        Mossy-fibre synapse weight.
    rate_MF : float
        Mossy-fibre background rate [Hz].
    verbose : bool
        If True, Brian2 reports simulation progress.

    Returns
    -------
    Tuple of Brian2 monitor objects (SpikeMonitor, PopulationRateMonitor,
    StateMonitor) for CA3 and CA1 PC and IN populations, plus the array of
    recorded neuron indices and the cue-target cell IDs.
    """

    if mode == 0:
        from linear_reward_variables import rest_time, num_CA3_neurons, num_CA1_neurons, state_position
        from linear_reward_functions import generate_place_cell_ID_list, get_tuning_curve
    elif mode == 1:
        from Tmaze_variables import rest_time, num_CA3_neurons, num_CA1_neurons, state_position
        from Tmaze_functions import generate_place_cell_ID_list, get_tuning_curve
    elif mode == 2:
        from linear_shock_variables import rest_time, num_CA3_neurons, num_CA1_neurons, state_position
        from linear_shock_functions import generate_place_cell_ID_list, get_tuning_curve

    if simulation_time != None: rest_time = simulation_time

    CA3_PCs = NeuronGroup(num_CA3_neurons, model=eqs_PC, threshold="vm>spike_th_PC",
                      reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    CA3_PCs.vm = Vrest_PC; CA3_PCs.g_ampa = 0.0; CA3_PCs.g_ampaMF = 0.0; CA3_PCs.g_gaba = 0.0
    CA3_INs = NeuronGroup(num_IN_neurons, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    CA3_INs.vm  = Vrest_BC; CA3_INs.g_ampa = 0.0; CA3_INs.g_gaba = 0.0


    CA1_PCs = NeuronGroup(num_CA1_neurons, model=eqs_PC, threshold="vm>spike_th_PC",
                      reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    CA1_PCs.vm = Vrest_PC; CA1_PCs.g_ampa = 0.0; CA1_PCs.g_ampaMF = 0.0; CA1_PCs.g_gaba = 0.0
    CA1_INs = NeuronGroup(num_IN_neurons, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    CA1_INs.vm  = Vrest_BC; CA1_INs.g_ampa = 0.0; CA1_INs.g_gaba = 0.0
    

    if pause_state>=0:
        CA3_place_cell_ID_list = generate_place_cell_ID_list(np.array(list(PF.keys()),dtype=int), np.array(list(PF.values())))
        place_cell_num = len(CA3_place_cell_ID_list[pause_state])
        spike_rates = infield_rate*get_tuning_curve(state_position[pause_state], [PF[key] for key in CA3_place_cell_ID_list[pause_state]])
        
        targ_cell_order = np.random.choice(np.arange(0, place_cell_num), size=place_cell_num, replace=False)
        targ_cell_id = CA3_place_cell_ID_list[pause_state][targ_cell_order]

        cue_input = PoissonGroup(place_cell_num, spike_rates[targ_cell_order]*Hz)
        Syn_CA3_cue = Synapses(cue_input, CA3_PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF_CA3")
        Syn_CA3_cue.connect(i=np.arange(0, place_cell_num), j=targ_cell_id)

        MF = PoissonGroup(num_CA3_neurons, rate_MF*Hz)

    else:
        MF = PoissonGroup(num_CA3_neurons, rate_MF*Hz)
        targ_cell_id = np.array([])
    
    Syn_CA3_MF = Synapses(MF, CA3_PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF_CA3")
    Syn_CA3_MF.connect(j="i")

    # weight matrix used here
    wmx_CA3 *= wmx_mult_CA3
    wmx_CA1 *= wmx_mult_CA1
    Syn_CA3_PC = Synapses(CA3_PCs, CA3_PCs, "w_exc:1", on_pre="x_ampa+=norm_PC_E*w_exc", delay=delay_PC_E)
    Syn_CA3_PC.connect(i=wmx_CA3.row, j=wmx_CA3.col)
    Syn_CA3_PC.w_exc = wmx_CA3.data
    del wmx_CA3

    Syn_CA3_IN = Synapses(CA3_INs, CA3_PCs, on_pre="x_gaba+=norm_PC_I*w_PC_IN_CA3", delay=delay_PC_I)
    Syn_CA3_IN.connect(p=IN_connection_prob)

    Syn_IN_CA3 = Synapses(CA3_PCs, CA3_INs, on_pre="x_ampa+=norm_BC_E*w_IN_PC_CA3", delay=delay_BC_E)
    Syn_IN_CA3.connect(p=CA3_CA3_connection_prob)

    Syn_CA3_IN_IN = Synapses(CA3_INs, CA3_INs, on_pre="x_gaba+=norm_BC_I*w_IN_IN_CA3", delay=delay_BC_I)
    Syn_CA3_IN_IN.connect(p=IN_connection_prob)

    Syn_CA1_PC = Synapses(CA3_PCs, CA1_PCs, "w_exc:1", on_pre="x_ampa+=norm_PC_E*w_exc", delay=delay_PC_E)
    Syn_CA1_PC.connect(i=wmx_CA1.row, j=wmx_CA1.col)
    Syn_CA1_PC.w_exc = wmx_CA1.data
    del wmx_CA1

    Syn_CA1_IN = Synapses(CA1_INs, CA1_PCs, on_pre="x_gaba+=norm_PC_I*w_PC_IN_CA1", delay=delay_PC_I)
    Syn_CA1_IN.connect(p=IN_connection_prob)

    Syn_IN_CA1 = Synapses(CA1_PCs, CA1_INs, on_pre="x_ampa+=norm_BC_E*w_IN_PC_CA1", delay=delay_BC_E)
    Syn_IN_CA1.connect(p=CA3_CA1_connection_prob)

    Syn_CA1_IN_IN = Synapses(CA1_INs, CA1_INs, on_pre="x_gaba+=norm_BC_I*w_IN_IN_CA1", delay=delay_BC_I)
    Syn_CA1_IN_IN.connect(p=IN_connection_prob)


    SM_PC_CA3 = SpikeMonitor(CA3_PCs); SM_IN_CA3 = SpikeMonitor(CA3_INs)
    RM_PC_CA3 = PopulationRateMonitor(CA3_PCs); RM_IN_CA3 = PopulationRateMonitor(CA3_INs)

    SM_PC_CA1 = SpikeMonitor(CA1_PCs); SM_IN_CA1 = SpikeMonitor(CA1_INs)
    RM_PC_CA1 = PopulationRateMonitor(CA1_PCs); RM_IN_CA1 = PopulationRateMonitor(CA1_INs)

    selection_CA3 = np.arange(0, num_CA3_neurons, 10)
    StateM_PC_CA3 = StateMonitor(CA3_PCs, variables=["vm", "w", "g_ampa", "g_ampaMF", "g_gaba"], record=selection_CA3.tolist(), dt=0.1*ms)
    StateM_IN_CA3 = StateMonitor(CA3_INs, "vm", record=[num_IN_neurons/2], dt=0.1*ms)

    selection_CA1 = np.arange(0, num_CA1_neurons, 10)
    StateM_PC_CA1 = StateMonitor(CA1_PCs, variables=["vm", "w", "g_ampa", "g_ampaMF", "g_gaba"], record=selection_CA1.tolist(), dt=0.1*ms)
    StateM_IN_CA1 = StateMonitor(CA1_INs, "vm", record=[num_IN_neurons/2], dt=0.1*ms)

    if verbose:
        run(rest_time*ms, report="text")
    else:
        run(rest_time*ms)

    return SM_PC_CA3, SM_IN_CA3, RM_PC_CA3, RM_IN_CA3,\
            SM_PC_CA1, SM_IN_CA1, RM_PC_CA1, RM_IN_CA1,\
            StateM_PC_CA3, StateM_IN_CA3, selection_CA3,\
            StateM_PC_CA1, StateM_IN_CA1, selection_CA1, targ_cell_id

# ---------------------------------------------------------------------------
# REVISED FROM ca3net (https://github.com/KaliLab/ca3net)
# Original: evaluated a single CA3 network; defined objectives for CA3 only.
# Changes:  extended to handle both CA3 and CA1 populations (12 objectives
#           instead of 6); passes W_CA1 and mode to offline_simulation.
# ---------------------------------------------------------------------------
class NetworkEvaluator(bpop.evaluators.Evaluator):
    """
    BluePyOpt Evaluator that wraps `offline_simulation` as the fitness function.

    The evolutionary algorithm minimises 12 error objectives (negated scores),
    six per hippocampal region (CA3 and CA1):
        rippleE, rippleI        — peak ripple power in PC / IN population
        no_gamma_peakI          — reward absence of gamma in IN population
        ripple_ratioE/I         — ripple-to-gamma power ratio
        low_rateE               — mean PC rate near background (rewarded)
    """

    def __init__(self, mode, W_CA3, W_CA1, cue, PF, params):
        """
        Parameters
        ----------
        mode : int
            Task selector passed through to offline_simulation.
        W_CA3, W_CA1 : scipy.sparse.coo_matrix
            Online-learned weight matrices (serialised to workers via cPickle).
        cue : int
            pause_state index for the cue injection (≥0) or -1.
        PF : dict
            CA3 place field dictionary {neuron_id: position}.
        params : list of (name, lower_bound, upper_bound) tuples
            Parameters to optimise and their search bounds.
        """
        super(NetworkEvaluator, self).__init__()
        self.mode = mode
        self.W_CA3 = W_CA3
        self.W_CA1 = W_CA1
        self.cue = cue
        self.PF = PF
        self.params = params

        # Parameters to be optimized
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval))
                       for name, minval, maxval in self.params]
        self.objectives = ["rippleE", "rippleI", "no_gamma_peakI", "ripple_ratioE", "ripple_ratioI", "low_rateE",
                           "rippleE_CA1", "rippleI_CA1", "no_gamma_peakI_CA1", "ripple_ratioE_CA1", "ripple_ratioI_CA1", "low_rateE_CA1",]

    def generate_model(self, individual):
        """Runs single simulation (see `run_sim.py`) and returns monitors"""
        SM_PC_CA3, SM_BC_CA3, RM_PC_CA3, RM_BC_CA3,\
            SM_PC_CA1, SM_BC_CA1, RM_PC_CA1, RM_BC_CA1,\
                _,_,_,_,_,_,_ = offline_simulation(self.mode, self.W_CA3, self.W_CA1, self.cue, self.PF, *individual)
        return SM_PC_CA3, SM_BC_CA3, RM_PC_CA3, RM_BC_CA3, SM_PC_CA1, SM_BC_CA1, RM_PC_CA1, RM_BC_CA1

    def init_simulator_and_evaluate_with_lists(self, individual):
        """Fitness error used by BluePyOpt for the optimization"""
        SM_PC_CA3, SM_BC_CA3, RM_PC_CA3, RM_BC_CA3,\
            SM_PC_CA1, SM_BC_CA1, RM_PC_CA1, RM_BC_CA1 = self.generate_model(individual)

        try:
            wc_errors = list(np.zeros((12),dtype=float))  # worst case scenario
            if SM_PC_CA3.num_spikes > 0 and SM_BC_CA3.num_spikes > 0 and SM_PC_CA1.num_spikes > 0 and SM_BC_CA1.num_spikes > 0:  # check if there is any activity
                
                rip_peakE_CA3, rip_peakI_CA3,\
                    no_gammaI_CA3, rip_ratE_CA3, rip_ratI_CA3, \
                        low_rE_CA3 = analyze_related_params(RM_PC_CA3, RM_BC_CA3)

                rip_peakE_CA1, rip_peakI_CA1,\
                    no_gammaI_CA1, rip_ratE_CA1, rip_ratI_CA1, \
                        low_rE_CA1 = analyze_related_params(RM_PC_CA1, RM_BC_CA1)

                # *-1 since the algorithm tries to minimize...
                errors = -1. * np.array([rip_peakE_CA3, rip_peakI_CA3, no_gammaI_CA3, \
                                        rip_ratE_CA3, rip_ratI_CA3, low_rE_CA3, \
                                        rip_peakE_CA1, rip_peakI_CA1, no_gammaI_CA1, \
                                        rip_ratE_CA1, rip_ratI_CA1, low_rE_CA1])

                return errors.tolist()
            else:
                return wc_errors
        except Exception:
            # Make sure exception and backtrace are thrown back to parent process
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))

class ca3NetworkEvaluator(bpop.evaluators.Evaluator):
    """
    BluePyOpt Evaluator that wraps `ca3_offline_simulation` as the
    fitness function.

    CA3-only counterpart of `NetworkEvaluator`: optimises the CA3 recurrent
    network in isolation, without building a CA1 population or a CA3→CA1
    feedforward projection. The evolutionary algorithm minimises 6 error
    objectives (negated scores):
        rippleE, rippleI        — peak ripple power in PC / IN population
        no_gamma_peakI          — reward absence of gamma in IN population
        ripple_ratioE/I         — ripple-to-gamma power ratio
        low_rateE               — mean PC rate near background (rewarded)
    """

    def __init__(self, mode, W_CA3, params):
        """
        Parameters
        ----------
        mode : int
            Task selector (0=linear_reward, 1=Tmaze, 2=linear_shock) — used to
            pick rest_time (offline simulation duration) and num_CA3_neurons.
        W_CA3 : scipy.sparse.coo_matrix
            Online-learned CA3-CA3 weight matrix (serialised to workers via cPickle).
        params : list of (name, lower_bound, upper_bound) tuples
            Parameters to optimise and their search bounds.
        """
        super(ca3NetworkEvaluator, self).__init__()
        self.mode = mode
        self.W_CA3 = W_CA3
        self.params = params

        if mode == 0:
            from linear_reward_variables import rest_time
        elif mode == 1:
            from Tmaze_variables import rest_time
        elif mode == 2:
            from linear_shock_variables import rest_time
        self.duration_ms = rest_time

        # Parameters to be optimized
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval))
                       for name, minval, maxval in self.params]
        self.objectives = ["rippleE", "rippleI", "no_gamma_peakI", "ripple_ratioE", "ripple_ratioI", "low_rateE",]

    def generate_model(self, individual):
        """Runs single simulation (spontaneous CA3-only replay) and returns monitors"""
        SM_PC_CA3, RM_PC_CA3, SM_IN_CA3, RM_IN_CA3 = ca3_offline_simulation(
            self.mode, self.W_CA3, self.duration_ms, *individual)
        return SM_PC_CA3, SM_IN_CA3, RM_PC_CA3, RM_IN_CA3

    def init_simulator_and_evaluate_with_lists(self, individual):
        """Fitness error used by BluePyOpt for the optimization"""
        SM_PC_CA3, SM_IN_CA3, RM_PC_CA3, RM_IN_CA3 = self.generate_model(individual)

        try:
            wc_errors = list(np.zeros((6),dtype=float))  # worst case scenario
            if SM_PC_CA3.num_spikes > 0 and SM_IN_CA3.num_spikes > 0:  # check if there is any activity

                rip_peakE, rip_peakI, no_gammaI, \
                    rip_ratE, rip_ratI, low_rE = analyze_related_params(RM_PC_CA3, RM_IN_CA3, len_sim=self.duration_ms)

                # *-1 since the algorithm tries to minimize...
                errors = -1. * np.array([rip_peakE, rip_peakI, no_gammaI, rip_ratE, rip_ratI, low_rE])

                return errors.tolist()
            else:
                return wc_errors
        except Exception:
            # Make sure exception and backtrace are thrown back to parent process
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))


# ---------------------------------------------------------------------------
# REVISED FROM ca3net (https://github.com/KaliLab/ca3net)
# Original: computed a subset of these metrics for CA3 only.
# Changes:  made the function generic (takes any RM_PC/RM_BC pair) so it can
#           be called for both CA3 and CA1 in the evaluator.
# ---------------------------------------------------------------------------
def analyze_related_params(RM_PC, RM_BC, len_sim=5000):
    """
    Compute SWR fitness metrics from Brian2 PopulationRateMonitors.

    Bins the raw Brian2 rate (1 kHz) into 10 ms bins, then identifies periods
    of sustained high activity (`slice_high_activity`), computes the power
    spectral density via Welch's method, and extracts:
        ripple_peakE/I   — Gaussian score for ripple frequency near 180 Hz
        no_gamma_peakI   — 1 if no significant gamma peak in INs, else 0
        ripple_ratioE/I  — clipped ripple/gamma power ratio
        low_rateE        — Gaussian score for background PC rate near baseline

    Parameters
    ----------
    RM_PC : Brian2 PopulationRateMonitor
        Rate monitor for place cells.
    RM_BC : Brian2 PopulationRateMonitor
        Rate monitor for basket / fast-spiking interneurons.
    len_sim : int
        Simulation duration [ms] (default 5000 ms).

    Returns
    -------
    Six float fitness scores (all positively oriented; the evaluator negates them).
    """
    rate_PC = np.array(RM_PC.rate_).reshape(-1, 10).mean(axis=1)
    rate_BC = np.array(RM_BC.rate_).reshape(-1, 10).mean(axis=1)
    gc.collect()

    # analyse rates
    slice_idx = slice_high_activity(rate_PC, th=2.0, min_len=150, len_sim=len_sim)
    _, mean_lowact_rate_PC, f_PC, Pxx_PC = analyse_rate(rate_PC, 1000.0, slice_idx)
    _, _, f_BC, Pxx_BC = analyse_rate(rate_BC, 1000.0, slice_idx)
    avg_ripple_freq_PC, ripple_power_PC = ripple(f_PC, Pxx_PC, slice_idx)
    avg_ripple_freq_BC, ripple_power_BC = ripple(f_BC, Pxx_BC, slice_idx)
    _, gamma_power_PC = gamma(f_PC, Pxx_PC, slice_idx)
    avg_gamma_freq_BC, gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx)

    # look for significant ripple peak close to 180 Hz
    ripple_peakE = np.exp(-1/2*(avg_ripple_freq_PC-180.)**2/20**2) if not np.isnan(avg_ripple_freq_PC) else 0.
    ripple_peakI = 2*np.exp(-1/2*(avg_ripple_freq_BC-180.)**2/20**2) if not np.isnan(avg_ripple_freq_BC) else 0.

    # penalize gamma peak (in inhibitory pop) - binary variable, which might not be the best for this algo.
    no_gamma_peakI = 1. if np.isnan(avg_gamma_freq_BC) else 0.

    # look for high ripple/gamma power ratio
    ripple_ratioE = np.clip(ripple_power_PC/gamma_power_PC, 0., 5.)
    ripple_ratioI = np.clip(2*ripple_power_BC/gamma_power_BC, 0., 10.)
    
    # look for "low" exc. population rate (around 2.5 Hz)
    low_rateE = np.exp(-1/2*(mean_lowact_rate_PC-background_rate)**2/background_rate**2)
    
    return ripple_peakE, ripple_peakI, no_gamma_peakI, ripple_ratioE, ripple_ratioI, low_rateE

def train_network(mode, target_lap, trial_number, pause_state=None, replay_type=False):
    """
    Update w_CA3_CA1 using STDP applied to offline replay spike trains.

    Loads the CA3 and CA1 spike trains generated by `simulate_offline_spikes`,
    then calls `update_STDP` to replay them through an STDP rule that
    strengthens connections whose CA3 spikes precede CA1 spikes.  Optionally
    restricts the update to specific replay time windows identified by
    `detect_replay_type`.

    The updated weight matrix at each time point of interest is used to
    recompute the CA1 activity map and identify new place cells.  Outputs:
        results/<task>/trial<N>/lap_<L>_0_<T>_replayed.npz  — final weights
        results/<task>/trial<N>/activity/CA1_activity_lap_<L>_replay_<T>.npz

    Parameters
    ----------
    mode : int
        Task selector.
    target_lap : int
        Lap whose weights / spike data to use.
    trial_number : int
        Number of trials to process.
    pause_state : int or None
        Which pause-state spike file to load (None = end-of-lap replay).
    replay_type : bool
        If True, restrict STDP to time intervals of classified replay events
        (requires prior `detect_replay_type` output).
    """
    
    if mode == 0:
        file_dir = os.path.join(base_path,"results/linear_reward")
        from linear_reward_variables import rest_time
        from linear_reward_functions import load_PF_starts, sample_spatial_points, get_tuning_curve
        place_peak_position = sample_spatial_points()
    elif mode == 1:
        file_dir = os.path.join(base_path,"results/Tmaze")
        from Tmaze_variables import rest_time, replay_trajectory
        from Tmaze_functions import load_PF_starts, sample_spatial_points, get_tuning_curve
        place_peak_position = sample_spatial_points()
    elif mode == 2:
        file_dir = os.path.join(base_path,"results/linear_shock")
        from linear_shock_variables import rest_time
        from linear_shock_functions import load_PF_starts, sample_spatial_points, get_tuning_curve
        place_peak_position = sample_spatial_points()
    
    CA3_PF = load_PF_starts()

    for trial in tqdm(range(trial_number)):
        foldername = "trial%d"%trial
        init_w = np.load(os.path.join(file_dir,foldername,"lap_%d.npz"%target_lap))["w_CA3_CA1"]
        connectivity=np.load(os.path.join(file_dir,foldername,"simulation_information.npz"))["connectivity_CA3_CA1"]

        if pause_state is None: CA3_spike_data = np.load(os.path.join(file_dir,foldername,"CA3_replay_after_lap_%d.npz"%target_lap))
        else: CA3_spike_data = np.load(os.path.join(file_dir,foldername,"CA3_replay_lap_%d_pause_%d.npz"%(target_lap,pause_state)))
        spiking_neurons_CA3 = CA3_spike_data["spiking_neurons_CA3_PC"]; spike_times_CA3 = CA3_spike_data["spike_times_CA3_PC"]
        spiking_neurons_CA3 = spiking_neurons_CA3.astype(int)
        spike_times_CA3 = np.round(spike_times_CA3).astype(int)
        del CA3_spike_data

        if pause_state is None: CA1_spike_data = np.load(os.path.join(file_dir,foldername,"CA1_replay_after_lap_%d.npz"%target_lap))
        else: CA1_spike_data = np.load(os.path.join(file_dir,foldername,"CA1_replay_lap_%d_pause_%d.npz"%(target_lap,pause_state)))
        spiking_neurons_CA1 = CA1_spike_data["spiking_neurons_CA1_PC"]; spike_times_CA1 = CA1_spike_data["spike_times_CA1_PC"]
        spiking_neurons_CA1 = spiking_neurons_CA1.astype(int)
        spike_times_CA1 = np.round(spike_times_CA1).astype(int)
        del CA1_spike_data

        timepoint_interest = [rest_time]
        if replay_type:
            import pickle
            for replay_type in range(len(replay_trajectory)):
                fname = os.path.join(file_dir,foldername,'replay_type_%d.pkl'%replay_type)
                with open(fname, 'rb') as fp: replays = pickle.load(fp)
                for time_interest in replays.keys(): timepoint_interest += list(time_interest) 
        timepoint_interest = np.unique(np.array(timepoint_interest))

        w_array = update_STDP(init_w, spiking_neurons_CA3, spiking_neurons_CA1, spike_times_CA3, spike_times_CA1, connectivity, timepoint_interest)
        file_out = os.path.join(file_dir,foldername,"lap_%d_%d_%d_replayed.npz"%(target_lap,0,rest_time))
        np.savez_compressed(file_out,w_CA3_CA1=w_array[-1])

        for tt in range(len(timepoint_interest)):
            w = w_array[tt]
            CA1_activity = np.zeros((place_peak_position.shape[0],w.shape[1]))
            
            for ll in range(place_peak_position.shape[0]):
                position = place_peak_position[ll]
                CA3_FR = get_tuning_curve(position, list(CA3_PF.values()))*infield_rate*theta_mod_factor
                CA1_activity[ll,:] = input_driven_rate(np.arange(w.shape[1]), CA3_FR, w)

            np.savez_compressed(os.path.join(file_dir,foldername, "activity/CA1_activity_lap_%d_replay_%d.npz"%(target_lap,timepoint_interest[tt])),CA1_activity=CA1_activity, place_thr_FR=place_thr_FR, unit_gran=4)

def update_STDP(init_w, spk_neuron_pre, spk_neuron_post, spk_time_pre, spk_time_post, connectivity, timepoint_interest, tau=20, eta=1e-3):
    """
    Apply a simplified all-to-all STDP rule to offline replay spike trains.

    Both pre- and post-synaptic eligibility traces accumulate at each spike
    event and decay exponentially with time constant `tau`.  Weight updates:
        w[pre, :]  += dApre  when CA3 neuron fires   (pre-synaptic trace)
        w[:, post] += dApost when CA1 neuron fires   (post-synaptic trace)

    Weights are clipped to [0, wmax] at every step and zeroed at unconnected
    synapses (masked by `connectivity`).  Snapshots are taken at each time
    point in `timepoint_interest` (typically the end of each replay event).

    Parameters
    ----------
    init_w : np.ndarray, shape (num_CA3, num_CA1)
        Initial weight matrix (copy is made; input is not modified).
    spk_neuron_pre / spk_neuron_post : np.ndarray of int
        Neuron indices of CA3 / CA1 spike events (from offline simulation).
    spk_time_pre / spk_time_post : np.ndarray of int
        Spike times [ms] corresponding to the neuron index arrays.
    connectivity : np.ndarray of bool, shape (num_CA3, num_CA1)
        Structural connectivity mask; unconnected entries stay zero.
    timepoint_interest : np.ndarray of int
        Time points [ms] at which to snapshot the weight matrix.
    tau : float
        Eligibility trace decay time constant [ms].
    eta : float
        Learning rate (dApre = dApost = eta).

    Returns
    -------
    w_array : np.ndarray, shape (len(timepoint_interest), num_CA3, num_CA1)
        Weight matrix snapshots at each time point of interest.
    """
    from global_variables import wmax
    dt = 1

    w = cp.deepcopy(init_w)
    w_array = np.zeros((len(timepoint_interest),w.shape[0],w.shape[1]))
    ET_pre = np.zeros((w.shape[0]), dtype=float)
    ET_post = np.zeros((w.shape[1]), dtype=float)

    dApre = eta; dApost = eta
    taupre = tau; taupost = tau

    target_idx_pre = np.where((spk_time_pre<timepoint_interest.max()))[0]
    target_neurons_pre = spk_neuron_pre[target_idx_pre]
    target_times_pre = spk_time_pre[target_idx_pre]

    target_idx_post = np.where((spk_time_post<timepoint_interest.max()))[0]
    target_neurons_post = spk_neuron_post[target_idx_post]
    target_times_post = spk_time_post[target_idx_post]

    count = 0
    for tt in np.arange(1,timepoint_interest.max()+1,dt,dtype=int):
        spiking_CA3 = target_neurons_pre[np.where((target_times_pre//dt)*dt==tt)[0]]
        spiking_CA1 = target_neurons_post[np.where((target_times_post//dt)*dt==tt)[0]]

        ET_pre[spiking_CA3] += dApre
        ET_post[spiking_CA1] += dApost

        w[spiking_CA3,:] += ET_pre[None, :]
        w[:,spiking_CA1] += ET_post[:, None]
        w = np.clip(w, 0, wmax*2)

        ET_pre -= ET_pre * (dt / taupre)
        ET_post -= ET_post * (dt / taupost)

        ET_pre = np.maximum(ET_pre,0)
        ET_post = np.maximum(ET_post,0)

        if np.isin(tt, timepoint_interest): w[connectivity == False] = 0; w_array[count] = w; count+=1
    
    return w_array
