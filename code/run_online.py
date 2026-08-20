"""
run_online.py
=============
Online (awake, active-exploration) learning simulation for the hippocampal BTSP model.

Overview
--------
The virtual animal navigates a maze over multiple laps. At each millisecond time step
the model generates Poisson spike trains for CA3 and CA1 populations, then updates
three synaptic weight matrices via two complementary learning rules:

  1. BTSP (Behavioral Time Scale Plasticity)  — updates w_CA3_CA3 and w_CA3_CA1
  2. Delta rule (predictive coding)            — updates w_CA1_feat

Network connectivity:
    CA3 --[w_CA3_CA3]--> CA3        recurrent; self-organizes place-cell sequences
    CA3 --[w_CA3_CA1]--> CA1        feedforward; builds the cognitive map
    CA1 --[w_CA1_feat]--> features  predicts which environmental cues are present

BTSP terminology:
    ET  – Eligibility Trace   : integral of recent pre-synaptic firing, decays ~tpre
    PT  – Plateau Potential   : dendritic event that gates BTSP weight changes;
                                triggered probabilistically by post-synaptic rate
    PS  – Perceived Salience  : weighted prediction-error signal; scales CA1 plateau
                                probability so the network learns faster in novel contexts
    f_presence – binary/real feature vector: which cues (reward, shock, location) are active

Supported task modes (select via the `mode` argument):
    0 – Linear Reward track  (unidirectional run; rewarded goal location)
    1 – T-maze               (two-arm choice, each arm has a distinct outcome)
    2 – Linear Shock track   (aversive zone at one end of the track)

Public entry points
-------------------
    run_online(mode, simul_trial, save_lap, ...)
        Main simulation loop; writes .npz snapshots under results/<task>/trial<N>/.

    find_place_cells(mode, trial_number)
        Post-hoc identification of CA1 place cells from saved weight matrices.

    run_factorial_shock(trial_number, ...)
        Linear-shock-only; factorial (speed, MI) control simulation for the
        shock-cue feature, branching from a shared online-lap checkpoint.
"""

import os, warnings, sys, copy
import numpy as np
import random as pyrandom
from tqdm import tqdm

warnings.filterwarnings("ignore")

base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
data_path = os.path.join(base_path,"results")
sys.path.insert(0, os.path.join(base_path, "code/functions"))

from global_variables import *
from common_functions import init_weights, init_layervars, save_place_fields
from common_functions import concat_spike_trains, generate_spike_byInput
from common_functions import ET_update, plateau_update, BTSP_update, PS_update, feat_weight_update
from common_functions import run_lap_prefix, run_lap_from_prefix, CHECKPOINT_FIELDS

def run_online(mode, simul_trial,save_lap, pause_state=[], seed=12345, verbose=False):
    """
    Simulate online BTSP learning for the selected task environment.

    Parameters
    ----------
    mode : int
        Task selector: 0=linear_reward, 1=Tmaze, 2=linear_shock.
    simul_trial : int
        Number of independent simulation trials (each gets a fresh seed offset).
    save_lap : int
        Checkpoint interval: weights + error signals are written every this many laps.
    pause_state : list of int, optional
        Maze state IDs where an extra mid-lap snapshot is saved.  Used later by
        run_offline.py to seed offline replay from a specific pause location.
    seed : int, optional
        Base random seed; each trial shifts it by trial*1e5, each lap by lap*1e6.
    verbose : bool, optional
        Print per-step firing rates and error signals (slow; for debugging only).
    """

    # -----------------------------------------------------------------
    # Task-specific imports: each mode provides the same interface
    # (same variable/function names) so the loop body is mode-agnostic.
    # -----------------------------------------------------------------
    if mode == 0:
        file_dir = os.path.join(data_path,"linear_reward")
        from linear_reward_variables import actions, num_CA3_neurons, num_CA1_neurons
        from linear_reward_variables import tot_lap, detailed_laps, exploration_actions, start, feature_speed, MI_vector, num_features
        from linear_reward_functions import retreive_ID_from_position, generate_place_field, presence_update
        from linear_reward_functions import generate_spike_byPlaceAndInput, load_PF_starts

    elif mode == 1:
        file_dir = os.path.join(data_path,"Tmaze")
        from Tmaze_variables import actions, num_CA3_neurons, num_CA1_neurons
        from Tmaze_variables import tot_lap, detailed_laps, exploration_actions, start, feature_speed, MI_vector, num_features
        from Tmaze_functions import retreive_ID_from_position, generate_place_field, presence_update
        from Tmaze_functions import generate_spike_byPlaceAndInput, load_PF_starts

    elif mode == 2:
        file_dir = os.path.join(data_path,"linear_shock")
        from linear_shock_variables import actions, num_CA3_neurons, num_CA1_neurons
        from linear_shock_variables import tot_lap, detailed_laps, exploration_actions, start, feature_speed, MI_vector, num_features
        from linear_shock_functions import retreive_ID_from_position, generate_place_field, presence_update
        from linear_shock_functions import generate_spike_byPlaceAndInput, load_PF_starts


    for trial in range(simul_trial):
        print("Running %dth simulation"%(trial+1))
        foldername = "trial"+str(trial)
        os.makedirs(os.path.join(file_dir,foldername), exist_ok=True)

        # Derive a unique, reproducible seed for this trial so trials are
        # independent but fully deterministic given the base seed.
        initial_seed = int(seed+trial*1e5)

        # Initialize all three weight matrices and the random sparse connectivity masks.
        # w_CA3_CA3  : (num_CA3, num_CA3) — recurrent CA3 synapses
        # w_CA3_CA1  : (num_CA3, num_CA1) — CA3→CA1 feedforward synapses
        # w_CA1_feat : (num_CA1, num_features) — CA1→feature prediction weights
        w_CA3_CA3, w_CA3_CA1, w_CA1_feat, connectivity_CA3_CA3, connectivity_CA3_CA1 = init_weights(num_CA3_neurons,num_CA1_neurons,num_features)

        # Load pre-generated CA3 place field centers from disk; generate them if missing.
        # CA3_place_fields : dict {neuron_id: 2D position of place field peak}
        pklf_name = os.path.join(file_dir, "PF_peak_data.pkl")
        try: CA3_place_fields = load_PF_starts()
        except: CA3_place_fields, _, _ = generate_place_field(initial_seed,num_CA3_neurons)
        del pklf_name

        init_w_CA3_CA3 = copy.deepcopy(w_CA3_CA3); init_w_CA3_CA1 = copy.deepcopy(w_CA3_CA1)
        w_CA3_CA3 = copy.deepcopy(init_w_CA3_CA3); w_CA3_CA1 = copy.deepcopy(init_w_CA3_CA1)

        # Per-layer dynamic variables (reset at the start of each trial):
        #   ET  – eligibility trace amplitude (pre-synaptic; shape: num_neurons)
        #   PT  – plateau trace amplitude    (post-synaptic; shape: num_neurons)
        #   plateau_flag     – bool mask: neuron is currently in plateau state
        #   plateau_refractory – countdown (ms) until a neuron can enter plateau again
        #   CA*_FR – instantaneous population firing rates [Hz]
        ET_CA3, PT_CA3, plateau_flag_CA3, plateau_refractory_CA3, CA3_FR = init_layervars(num_CA3_neurons)
        _, PT_CA1, plateau_flag_CA1, plateau_refractory_CA1, CA1_FR = init_layervars(num_CA1_neurons)

        step_error = np.zeros((num_features))  # prediction error accumulated over one dA_granularity window

        for lap in tqdm(range(1,tot_lap+1)):

            # Each lap gets its own seed so laps are independent within a trial.
            seed = int(1e6*lap+initial_seed)

            np.random.seed(seed)
            pyrandom.seed(seed)

            current_position = start  # reset animal position to start of maze
            PS_list = []; error_list = []  # per-lap logs of salience and prediction error

            # ---------------------------------------------------------------
            # Step loop: the animal takes one discrete action per step
            # (e.g., move one maze unit left/right/up/down).
            # exploration_actions[lap-1] gives the pre-defined action sequence
            # for this lap (shape: num_steps_per_lap).
            # ---------------------------------------------------------------
            for step in range(exploration_actions.shape[1]):

                action_ID = exploration_actions[lap-1,step]
                # Identify which maze state (discrete location) the animal is in
                # at the midpoint of the current step.
                current_unit_ID, _ = retreive_ID_from_position(current_position + actions[action_ID]/2)

                # f_presence: binary vector (length num_features) indicating
                # which cues/reward/shock features are currently perceivable.
                f_presence = presence_update(current_unit_ID, lap)
                # Effective running speed depends on the type of current feature
                # (e.g., the animal slows at reward/shock locations).
                mice_speed = v_mice*feature_speed[np.where(f_presence==1)[0][0]]
                # Duration of this step in ms, scaled by movement speed.
                current_T = int(step_time_length*np.linalg.norm(actions[action_ID])/(mice_speed*sec))
                if verbose: print("Moving through state %d for %dms"%(current_unit_ID,current_T))

                # ---------------------------------------------------------------
                # Time loop: 1 ms resolution; runs for current_T ms per step.
                # Spike trains are regenerated every dA_granularity ms (default 100 ms).
                # ---------------------------------------------------------------
                for tt in range(current_T):

                    if tt%dA_granularity == 0:
                        # --- CA3 spike generation ---
                        # Each CA3 neuron fires according to its place-field tuning curve
                        # at the animal's current position PLUS recurrent input from w_CA3_CA3.
                        # Returns a list of spike trains (one per neuron) over dA_granularity ms.
                        spike_trains_CA3 = generate_spike_byPlaceAndInput(
                            np.arange(num_CA3_neurons),
                            CA3_place_fields,
                            current_position+actions[action_ID]*tt/current_T,      # start of sub-interval
                            current_position+actions[action_ID]*(tt+dA_granularity)/current_T,  # end
                            dA_granularity/sec,w_CA3_CA3, CA3_FR,
                            mice_speed=mice_speed,
                            seed=seed)
                        seed += 1
                        # Convert spike trains to an instantaneous population rate vector [Hz].
                        CA3_FR = (sec/dA_granularity) * np.array([len(spikes) for spikes in spike_trains_CA3])
                        # Flatten the per-neuron spike-train lists into two arrays:
                        #   spiking_neurons_CA3 — neuron index of each spike event
                        #   spike_times_CA3     — time of each spike event (ms, offset by tt)
                        spiking_neurons_CA3, spike_times_CA3 = concat_spike_trains(spike_trains_CA3, num_CA3_neurons)
                        spiking_neurons_CA3 = spiking_neurons_CA3.astype(int)
                        spike_times_CA3 = tt + np.round(spike_times_CA3,decimals=(-np.log10(dt*1e-3)).astype(int))*sec

                        # --- CA1 spike generation ---
                        # CA1 neurons fire purely based on synaptic input from CA3 (no place field).
                        spike_trains_CA1 = generate_spike_byInput(
                            np.arange(num_CA1_neurons),
                            dA_granularity/sec,w_CA3_CA1,CA3_FR,
                            seed=seed)
                        seed += 1
                        CA1_FR = (sec/dA_granularity) * np.array([len(spikes) for spikes in spike_trains_CA1])
                        spiking_neurons_CA1, spike_times_CA1 = concat_spike_trains(spike_trains_CA1, num_CA1_neurons)
                        spiking_neurons_CA1 = spiking_neurons_CA1.astype(int)
                        spike_times_CA1 = tt + np.round(spike_times_CA1,decimals=(-np.log10(dt*1e-3)).astype(int))*sec

                        # --- Log prediction error and perceived salience ---
                        # mean_error: average per-feature prediction error over the last window.
                        mean_error = (step_error/dA_granularity)
                        error_list.append(mean_error)
                        step_error = np.zeros((num_features))  # reset for next window

                        # PS_update computes a scalar salience: high PS → large prediction error
                        # in salient features → faster CA1 synaptic plasticity.
                        mean_perceived_salience = PS_update(f_presence,MI_vector,np.abs(mean_error))
                        PS_list.append(mean_perceived_salience)

                        if verbose:
                            print("act. CA3: %.4f"%(np.average(CA3_FR)))
                            print("act. CA1: %.4f"%(np.average(CA1_FR)))
                            print("mean error:", mean_error)
                            print("Mean perceived salience:", mean_perceived_salience)
                            print("--")

                    # --- CA3 BTSP update (recurrent synapses) ---
                    # Step 1: Update eligibility trace for CA3 pre-synaptic neurons.
                    #         ET_CA3[i] is incremented when neuron i fires; decays with tpre.
                    ET_CA3 = ET_update(tt, spike_times_CA3, spiking_neurons_CA3, ET_CA3, ET=ET_amp)
                    # Step 2: Stochastically trigger plateau potentials for CA3 post-synaptic neurons.
                    #         Probability scales with each neuron's firing rate vs target_FR_CA3.
                    PT_CA3, plateau_flag_CA3, plateau_refractory_CA3 = plateau_update(CA3_FR, PT_CA3, target_FR_CA3,
                                                                                        plateau_flag_CA3, plateau_refractory_CA3,
                                                                                        base_prob=base_prob_CA3, p_slope=firing_prob_slope_CA3,
                                                                                        seed=seed)
                    seed += 1
                    # Step 3: Apply BTSP rule: w += f(ET_pre ⊗ PT_post) for neurons in plateau.
                    w_CA3_CA3 = BTSP_update(ET_CA3,PT_CA3,plateau_flag_CA3,w_CA3_CA3,connectivity_CA3_CA3,BTSP_scaling_CA3)

                    # --- CA1 feature-prediction update (delta rule) ---
                    # w_CA1_feat learns to predict f_presence from CA1 firing rates.
                    # error = f_presence − w_CA1_feat·CA1_FR
                    w_CA1_feat, error = feat_weight_update(w_CA1_feat, CA1_FR, f_presence)
                    step_error += np.abs(error)  # accumulate error magnitude for the current window
                    perceived_salience = PS_update(f_presence,MI_vector,np.abs(error))

                    # --- CA1 BTSP update (CA3→CA1 feedforward synapses) ---
                    # CA1 plateau probability is additionally gated by perceived salience (PS):
                    # higher PS → more plateaus → faster map formation in novel environments.
                    PT_CA1, plateau_flag_CA1, plateau_refractory_CA1 = plateau_update(CA1_FR, PT_CA1, target_FR_CA1,
                                                                                        plateau_flag_CA1,
                                                                                        plateau_refractory_CA1, min_prob=min_prob_CA1,
                                                                                        PS=perceived_salience,
                                                                                        seed=seed)
                    seed += 1
                    # ET_CA3 serves as the pre-synaptic trace here (CA3→CA1 synapses).
                    w_CA3_CA1 = BTSP_update(ET_CA3,PT_CA1,plateau_flag_CA1,w_CA3_CA1,connectivity_CA3_CA1,BTSP_scaling_CA1)

                    # --- Exponential decay of eligibility and plateau traces ---
                    ET_CA3 -= ET_CA3 * (dt / tpre)
                    PT_CA3 -= PT_CA3 * (dt / tpre)
                    PT_CA1 -= PT_CA1 * (dt / tpost)

                # Save intermediate snapshot when the animal pauses at a designated state
                # (used to seed offline replay from a specific location).
                if (lap%save_lap == 0)and(current_unit_ID+1 in pause_state):
                    file_out = os.path.join(file_dir,foldername,"lap_%d_pause_%d.npz"%(lap,current_unit_ID+1))
                    np.savez_compressed(file_out,
                        error_list=error_list,PS_list=PS_list,
                        w_CA3_CA3=w_CA3_CA3,w_CA3_CA1=w_CA3_CA1,w_CA1_feat=w_CA1_feat)
                    del file_out
                current_position = current_position + actions[action_ID]

            # Save end-of-lap snapshot.
            if lap%save_lap == 0:
                file_out = os.path.join(file_dir,foldername,"lap_%d.npz"%lap)
                if lap in detailed_laps:
                    np.savez_compressed(file_out,
                        error_list=error_list,PS_list=PS_list,
                        w_CA3_CA3=w_CA3_CA3,w_CA3_CA1=w_CA3_CA1,w_CA1_feat=w_CA1_feat,
                        ET_CA3=ET_CA3,PT_CA3=PT_CA3,plateau_flag_CA3=plateau_flag_CA3,plateau_refractory_CA3=plateau_refractory_CA3,CA3_FR=CA3_FR,
                        PT_CA1=PT_CA1,plateau_flag_CA1=plateau_flag_CA1,plateau_refractory_CA1=plateau_refractory_CA1,CA1_FR=CA1_FR)
                else:
                    np.savez_compressed(file_out,
                        error_list=error_list,PS_list=PS_list,
                        w_CA3_CA3=w_CA3_CA3,w_CA3_CA1=w_CA3_CA1,w_CA1_feat=w_CA1_feat)
                del file_out

def find_place_cells(mode, trial_number):
    """
    Identify CA1 place cells from saved weight matrices.

    For each trial and each saved lap, this function sweeps a grid of spatial
    positions, computes the CA1 activity map (using the stored w_CA3_CA1 and
    the fixed CA3 place fields), and labels neurons whose peak rate exceeds
    `place_thr_FR` as place cells.  Results are written to:
        <task>/trial<N>/activity/CA1_activity_lap_<L>.npz  — full activity map
        <task>/trial<N>/detected_PC/CA1_PF_lap_<L>.pkl     — place field dict

    Parameters
    ----------
    mode : int
        Task selector (0=linear_reward, 1=Tmaze, 2=linear_shock).
    trial_number : int
        Number of trials to analyze (processes trial0 … trial<N-1>).
    """

    if mode == 0:
        file_dir = os.path.join(data_path,"linear_reward")
        from linear_reward_variables import num_CA1_neurons, tot_lap
        from linear_reward_functions import sample_spatial_points, get_tuning_curve, load_PF_starts
    
    elif mode == 1:
        file_dir = os.path.join(data_path,"Tmaze")
        from Tmaze_variables import num_CA1_neurons, tot_lap
        from Tmaze_functions import sample_spatial_points, get_tuning_curve, load_PF_starts

    elif mode == 2:
        file_dir = os.path.join(data_path,"linear_shock")
        from linear_shock_variables import num_CA1_neurons, tot_lap
        from linear_shock_functions import sample_spatial_points, get_tuning_curve, load_PF_starts

    
    print("Identify place cells...")
    place_peak_position = sample_spatial_points(unit_gran)

    for seed in tqdm(range(trial_number)):
        foldername = "trial" + str(seed)
        CA3_place_fields = load_PF_starts(os.path.join(data_path, "PF_peak_data.pkl"))

        for load_episode in range(1,tot_lap+1):
            w_CA3_CA1 = np.load(os.path.join(file_dir,foldername,"lap_%d.npz"%load_episode))["w_CA3_CA1"]
            CA1_activity = np.zeros((place_peak_position.shape[0],num_CA1_neurons))

            for ll in range(place_peak_position.shape[0]):
                position = place_peak_position[ll]
                CA3_FR = get_tuning_curve(position, list(CA3_place_fields.values()))*infield_rate*theta_mod_factor
                CA1_activity[ll,:] = input_driven_rate(np.arange(num_CA1_neurons), CA3_FR, w_CA3_CA1, rate_shift=5)

            np.savez_compressed(os.path.join(file_dir, foldername, "activity/CA1_activity_lap_"+str(load_episode)+".npz"),CA1_activity=CA1_activity, place_thr_FR=place_thr_FR, unit_gran=unit_gran)

            place_cell_idx = np.where((np.max(CA1_activity,axis=0)>=place_thr_FR))[0]
            place_field_sorted_idx = place_cell_idx[np.argsort(np.argmax(CA1_activity[:,place_cell_idx],axis=0))]
            place_peak_idx = find_PF_peak(CA1_activity)
            CA1_place_fields = {neuron_id:np.array(place_peak_position[place_peak_idx[neuron_id]]) for neuron_id in place_field_sorted_idx}
            print("%d place cell identified after lap %d!"%(len(place_cell_idx),load_episode))

            pklf_name = os.path.join(file_dir, foldername, "detected_PC/CA1_PF_lap_"+str(load_episode)+".pkl")
            save_place_fields(CA1_place_fields, pklf_name)

def run_online_STDP(simul_trial, lap_increment=[500,500,500,500,200,200,200,200,200],
                    _tau=20, _eta=1e-3, seed=12345, verbose=False):
    """
    Simulate online STDP learning for the selected task environment.

    Parameters
    ----------
    simul_trial : int
        Number of independent simulation trials (each gets a fresh seed offset).
    save_lap : int
        Checkpoint interval: weights + error signals are written every this many laps.
    seed : int, optional
        Base random seed; each trial shifts it by trial*1e5, each lap by lap*1e6.
    verbose : bool, optional
        Print per-step firing rates and error signals (slow; for debugging only).
    """

    # -----------------------------------------------------------------
    # Task-specific imports: each mode provides the same interface
    # (same variable/function names) so the loop body is mode-agnostic.
    # -----------------------------------------------------------------

    file_dir = os.path.join(data_path,"linear_reward")
    from linear_reward_variables import actions, num_CA3_neurons
    from linear_reward_variables import num_state_total
    from linear_reward_functions import retreive_ID_from_position, generate_place_field
    from linear_reward_functions import generate_spike_byPlaceAndInput, load_PF_starts
    dApre = _eta; tpre = _tau

    dA_granularity = 100

    for trial in range(simul_trial):
        
        print("Running %dth simulation"%(trial+1))
        foldername = "STDP"+str(trial)
        os.makedirs(os.path.join(file_dir,foldername), exist_ok=True)

        # Derive a unique, reproducible seed for this trial so trials are
        # independent but fully deterministic given the base seed.
        initial_seed = int(seed+trial*1e5)
        lap = 0

        # Initialize all three weight matrices and the random sparse connectivity masks.
        # w_CA3_CA3  : (num_CA3, num_CA3) — recurrent CA3 synapses
        w_CA3_CA3, _, _, connectivity_CA3_CA3, _ = init_weights(num_CA3_neurons,1,1)

        # Load pre-generated CA3 place field centers from disk; generate them if missing.
        # CA3_place_fields : dict {neuron_id: 2D position of place field peak}
        try: CA3_place_fields = load_PF_starts()
        except: CA3_place_fields, _, _ = generate_place_field(initial_seed,num_CA3_neurons)

        # Per-layer dynamic variables (reset at the start of each trial):
        #   ET  – eligibility trace amplitude (pre-synaptic; shape: num_neurons)
        #   CA3_FR – instantaneous population firing rates [Hz]
        ET_CA3, _, _, _, CA3_FR = init_layervars(num_CA3_neurons)

        for lap_incr in tqdm(lap_increment):
            # try: 
            # ff = np.load(os.path.join(file_dir, foldername, "lap_%d.npz" % lap))
            # init_w_CA3_CA3=ff["w_CA3_CA3"];ET_CA3=ff["ET_CA3"];CA3_FR=ff["CA3_FR"];connectivity_CA3_CA3=ff["connectivity_CA3_CA3"];del ff
            # except: pass

            lap += lap_incr

            w_CA3_CA3[~connectivity_CA3_CA3] = 0  # enforce sparse connectivity mask
            init_w_CA3_CA3 = copy.deepcopy(w_CA3_CA3); w_CA3_CA3 = copy.deepcopy(init_w_CA3_CA3)

            # Each lap gets its own seed so laps are independent within a trial.
            seed = int(1e3*lap+initial_seed)

            np.random.seed(seed)
            pyrandom.seed(seed)

            current_position = np.array([0,0])  # reset animal position to start of maze

            # ---------------------------------------------------------------
            # Step loop: the animal takes one discrete action per step
            # (e.g., move one maze unit left/right/up/down).
            # for this lap (shape: num_steps_per_lap).
            # ---------------------------------------------------------------
            for _ in range(num_state_total):

                action_ID = 1
                # Identify which maze state (discrete location) the animal is in
                # at the midpoint of the current step.
                current_unit_ID, _ = retreive_ID_from_position(current_position + actions[action_ID]/2)

                # Effective running speed depends on the type of current feature.
                mice_speed = v_mice
                # Duration of this step in ms, scaled by movement speed.
                current_T = sec
                if verbose: print("Moving through state %d for %dms"%(current_unit_ID,current_T))

                # ---------------------------------------------------------------
                # Time loop: 1 ms resolution; runs for current_T ms per step.
                # Spike trains are regenerated every dA_granularity ms (default 100 ms).
                # ---------------------------------------------------------------
                for tt in range(sec):

                    if tt%dA_granularity == 0:
                        # --- CA3 spike generation ---
                        # Each CA3 neuron fires according to its place-field tuning curve
                        # at the animal's current position PLUS recurrent input from w_CA3_CA3.
                        # Returns a list of spike trains (one per neuron) over dA_granularity ms.
                        spike_trains_CA3 = generate_spike_byPlaceAndInput(
                            np.arange(num_CA3_neurons),
                            CA3_place_fields,
                            current_position+actions[action_ID]*tt/current_T,      # start of sub-interval
                            current_position+actions[action_ID]*(tt+dA_granularity)/current_T,  # end
                            dA_granularity/sec, w_CA3_CA3, CA3_FR,
                            mice_speed=v_mice,
                            seed=seed)
                        seed += 1
                        # Convert spike trains to an instantaneous population rate vector [Hz].
                        CA3_FR = (sec/dA_granularity) * np.array([len(spikes) for spikes in spike_trains_CA3])
                        # Flatten the per-neuron spike-train lists into two arrays:
                        #   spiking_neurons_CA3 — neuron index of each spike event
                        #   spike_times_CA3     — time of each spike event (ms, offset by tt)
                        spiking_neurons_CA3, spike_times_CA3 = concat_spike_trains(spike_trains_CA3, num_CA3_neurons)
                        spiking_neurons_CA3 = spiking_neurons_CA3.astype(int)
                        spike_times_CA3 = tt + np.round(spike_times_CA3,decimals=(-np.log10(dt*1e-3)).astype(int))*sec

                        if verbose:
                            print("act. CA3: %.4f"%(np.average(CA3_FR)))
                            print("--")

                    # --- CA3 STDP update (recurrent synapses) ---
                    spiking_CA3 = spiking_neurons_CA3[np.where(spike_times_CA3==tt)[0]]
                    ET_CA3[spiking_CA3] += dApre

                    # Step 3: Apply STDP rule.
                    w_CA3_CA3[spiking_CA3,:] += ET_CA3[None, :]
                    w_CA3_CA3[:,spiking_CA3] += ET_CA3[:, None]
                    w_CA3_CA3[~connectivity_CA3_CA3] = 0  # enforce sparse connectivity mask

                    # --- Exponential decay of eligibility and plateau traces ---
                    ET_CA3 -= ET_CA3 * (dt / tpre)

                current_position += actions[action_ID]

            lap_w_incr = w_CA3_CA3 - init_w_CA3_CA3
            w_CA3_CA3 += (lap_incr-1)*lap_w_incr

            # Save end-of-lap snapshot.
            file_out = os.path.join(file_dir,foldername,"lap_%d.npz"%lap)
            np.savez_compressed(file_out,
                                w_CA3_CA3=w_CA3_CA3, connectivity_CA3_CA3=connectivity_CA3_CA3,
                                CA3_FR=CA3_FR, ET_CA3=ET_CA3)
            del file_out
        del current_position

# (target_speed, target_MI) applied to the shock-cue feature only; all other
# features keep neutral speed=1, MI=1.
FACTORIAL_SHOCK_SCENARIOS = [(1, 1), (1, 3), (1, 5), (1, 7), (1, 10),
                             (2, 1), (2, 3), (2, 5), (2, 7), (2, 10),
                             (0.5, 1), (0.5, 3), (0.5, 5), (0.5, 7), (0.5, 10)]

def run_factorial_shock(trial_number, data_root=os.path.join(data_path,"linear_shock"),
                         out_root=os.path.join(base_path,"results/factorial_control"), resume_lap=3, next_lap=4,
                         feat_idx=1, scenarios=FACTORIAL_SHOCK_SCENARIOS):
    """
    Factorial control simulation (linear shock only): resume each trial's
    `resume_lap` checkpoint (exploration phase, shock not yet present) and
    re-run `next_lap` — the lap where the shock cue turns on — under every
    (speed, MI) scaling in `scenarios`, applied to the shock feature
    (`feat_idx`) only.

    """
    from linear_shock_functions import load_PF_starts
    from linear_shock_variables import num_features

    mode = 2
    CA3_place_fields = load_PF_starts(os.path.join(data_root, "PF_peak_data.pkl"))

    for trial in range(trial_number):
        foldername = "trial%d" % trial
        print("Shock scenario — trial %d/%d" % (trial + 1, trial_number))

        sim_info = np.load(os.path.join(data_root, foldername, "simulation_information.npz"))
        connectivity_CA3_CA3 = sim_info["connectivity_CA3_CA3"]
        connectivity_CA3_CA1 = sim_info["connectivity_CA3_CA1"]
        initial_seed = int(sim_info["initial_seed"])

        checkpoint = np.load(os.path.join(data_root, foldername, "lap_%d.npz" % resume_lap))
        state = {k: checkpoint[k] for k in CHECKPOINT_FIELDS}

        out_trial_dir = os.path.join(out_root, foldername)
        os.makedirs(out_trial_dir, exist_ok=True)

        # Shared prefix (identical across all scenarios below): simulate once
        # per trial up to the step right before the shock cue turns on, instead
        # of re-simulating it inside every scenario iteration.
        prefix_out = os.path.join(out_trial_dir, "lap_%d_prefix.npz" % next_lap)
        if os.path.exists(prefix_out):
            prefix = np.load(prefix_out, allow_pickle=True)
        else:
            prefix = run_lap_prefix(mode, next_lap, initial_seed, state,
                                     connectivity_CA3_CA3, connectivity_CA3_CA1,
                                     CA3_place_fields, feat_idx)
            np.savez_compressed(prefix_out, **prefix)

        for target_speed, target_MI in tqdm(scenarios):
            MI_vector = np.ones(num_features); MI_vector[feat_idx] = target_MI
            feature_speed = np.ones(num_features); feature_speed[feat_idx] = target_speed

            result = run_lap_from_prefix(mode, next_lap, prefix,
                                          connectivity_CA3_CA3, connectivity_CA3_CA1,
                                          CA3_place_fields, MI_vector, feature_speed)

            file_out = os.path.join(out_trial_dir, "lap_%d_speed_%g_MI_%g.npz" % (next_lap, target_speed, target_MI))
            np.savez_compressed(file_out, target_speed=target_speed, target_MI=target_MI, **result)