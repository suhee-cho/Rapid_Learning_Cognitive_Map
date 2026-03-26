import os, warnings, sys, copy
import numpy as np
import random as pyrandom
from tqdm import tqdm

warnings.filterwarnings("ignore")

base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-1])
sys.path.insert(0, os.path.join(base_path, "functions"))

from global_variables import *
from common_functions import init_weights, init_layervars, save_place_fields
from common_functions import concat_spike_trains, generate_spike_byInput
from common_functions import ET_update, plateau_update, BTSP_update, PS_update, feat_weight_update

def run_online(mode, simul_trial, save_lap, pause_state=[], verbose=False):

    if mode == 0:
        file_dir = os.path.join(base_path,"linear_reward")
        from linear_reward_variables import actions, num_CA3_neurons, num_CA1_neurons
        from linear_reward_variables import tot_lap, exploration_actions, start, feature_speed, MI_vector, num_features
        from linear_reward_functions import retreive_ID_from_position, generate_place_field, presence_update
        from linear_reward_functions import generate_spike_byPlaceAndInput, load_PF_starts

    elif mode == 1:
        file_dir = os.path.join(base_path,"Tmaze")
        from Tmaze_variables import actions, num_CA3_neurons, num_CA1_neurons
        from Tmaze_variables import tot_lap, exploration_actions, start, feature_speed, MI_vector, num_features
        from Tmaze_functions import retreive_ID_from_position, generate_place_field, presence_update
        from Tmaze_functions import generate_spike_byPlaceAndInput, load_PF_starts

    elif mode == 2:
        file_dir = os.path.join(base_path,"linear_shock")
        from linear_shock_variables import actions, num_CA3_neurons, num_CA1_neurons
        from linear_shock_variables import tot_lap, exploration_actions, start, feature_speed, MI_vector, num_features
        from linear_shock_functions import retreive_ID_from_position, generate_place_field, presence_update
        from linear_shock_functions import generate_spike_byPlaceAndInput, load_PF_starts


    for trial in range(simul_trial):
        print("Running %dth simulation"%(trial+1))
        foldername = "trial"+str(trial)
        os.makedirs(os.path.join(file_dir,"results",foldername), exist_ok=True)

        # Initialize variables
        initial_seed = int(trial*1e6)

        w_CA3_CA3, w_CA3_CA1, w_CA1_feat, connectivity_CA3_CA3, connectivity_CA3_CA1 = init_weights(num_CA3_neurons,num_CA1_neurons,num_features)
        
        pklf_name = os.path.join(file_dir, "PF_peak_data.pkl")
        try: CA3_place_fields = load_PF_starts()
        except: CA3_place_fields, _, _ = generate_place_field(initial_seed,num_CA3_neurons,pklf_name)
        del pklf_name

        init_w_CA3_CA3 = copy.deepcopy(w_CA3_CA3); init_w_CA3_CA1 = copy.deepcopy(w_CA3_CA1)
        w_CA3_CA3 = copy.deepcopy(init_w_CA3_CA3); w_CA3_CA1 = copy.deepcopy(init_w_CA3_CA1)
        
        ET_CA3, PT_CA3, plateau_flag_CA3, plateau_refractory_CA3, CA3_FR = init_layervars(num_CA3_neurons)
        _, PT_CA1, plateau_flag_CA1, plateau_refractory_CA1, CA1_FR = init_layervars(num_CA1_neurons)
        
        step_error = np.zeros((num_features))

        for lap in tqdm(range(1,tot_lap+1)):
            
            seed = int(1e6*lap+initial_seed)

            np.random.seed(seed)
            pyrandom.seed(seed)

            current_position = start
            PS_list = []; error_list = []

            # Simulation loop
            for step in range(exploration_actions.shape[1]):
                
                action_ID = exploration_actions[lap-1,step]
                current_unit_ID, _ = retreive_ID_from_position(current_position + actions[action_ID]/2)

                f_presence = presence_update(current_unit_ID, lap)
                mice_speed = v_mice*feature_speed[np.where(f_presence==1)[0][0]]
                current_T = int(step_time_length*np.linalg.norm(actions[action_ID])/(mice_speed*sec))
                if verbose: print("Moving through state %d for %dms"%(current_unit_ID,current_T))
                
                for tt in range(current_T):
                # Update activity of each cell
                
                    if tt%dA_granularity == 0:
                        # Generate CA3 spike trains
                        spike_trains_CA3 = generate_spike_byPlaceAndInput(
                            np.arange(num_CA3_neurons),
                            CA3_place_fields,
                            current_position+actions[action_ID]*tt/current_T,
                            current_position+actions[action_ID]*(tt+dA_granularity)/current_T,
                            dA_granularity/sec,w_CA3_CA3, CA3_FR,
                            mice_speed=mice_speed,
                            seed=seed)
                        seed += 1
                        CA3_FR = (sec/dA_granularity) * np.array([len(spikes) for spikes in spike_trains_CA3])
                        spiking_neurons_CA3, spike_times_CA3 = concat_spike_trains(spike_trains_CA3, num_CA3_neurons)
                        spiking_neurons_CA3 = spiking_neurons_CA3.astype(int)
                        spike_times_CA3 = tt + np.round(spike_times_CA3,decimals=(-np.log10(dt*1e-3)).astype(int))*sec

                        spike_trains_CA1 = generate_spike_byInput(
                            np.arange(num_CA1_neurons),
                            dA_granularity/sec,w_CA3_CA1,CA3_FR,
                            seed=seed)
                        seed += 1
                        CA1_FR = (sec/dA_granularity) * np.array([len(spikes) for spikes in spike_trains_CA1])
                        spiking_neurons_CA1, spike_times_CA1 = concat_spike_trains(spike_trains_CA1, num_CA1_neurons)
                        spiking_neurons_CA1 = spiking_neurons_CA1.astype(int)
                        spike_times_CA1 = tt + np.round(spike_times_CA1,decimals=(-np.log10(dt*1e-3)).astype(int))*sec
                        
                        mean_error = (step_error/dA_granularity)
                        error_list.append(mean_error)
                        step_error = np.zeros((num_features))

                        mean_perceived_salience = PS_update(f_presence,MI_vector,np.abs(mean_error))
                        PS_list.append(mean_perceived_salience)

                        if verbose:
                            print("act. CA3: %.4f"%(np.average(CA3_FR)))
                            print("act. CA1: %.4f"%(np.average(CA1_FR)))
                            print("mean error:", mean_error)
                            print("Mean perceived salience:", mean_perceived_salience)
                            print("--")
                    
                    # Update CA3 layer, W_CA3
                    ET_CA3 = ET_update(tt, spike_times_CA3, spiking_neurons_CA3, ET_CA3, ET=ET_amp)
                    PT_CA3, plateau_flag_CA3, plateau_refractory_CA3 = plateau_update(CA3_FR, PT_CA3, target_FR_CA3,
                                                                                        plateau_flag_CA3, plateau_refractory_CA3,
                                                                                        base_prob=base_prob_CA3, p_slope=firing_prob_slope_CA3,
                                                                                        seed=seed)
                    seed += 1
                    w_CA3_CA3 = BTSP_update(ET_CA3,PT_CA3,plateau_flag_CA3,w_CA3_CA3,connectivity_CA3_CA3,BTSP_scaling_CA3)

                    # Update W_pred
                    w_CA1_feat, error = feat_weight_update(w_CA1_feat, CA1_FR, f_presence)
                    step_error += np.abs(error)
                    perceived_salience = PS_update(f_presence,MI_vector,np.abs(error))

                    # Update CA1 layer, W_CA1
                    PT_CA1, plateau_flag_CA1, plateau_refractory_CA1 = plateau_update(CA1_FR, PT_CA1, target_FR_CA1, 
                                                                                        plateau_flag_CA1,  
                                                                                        plateau_refractory_CA1, min_prob=min_prob_CA1,
                                                                                        PS=perceived_salience,
                                                                                        seed=seed)
                    seed += 1
                    w_CA3_CA1 = BTSP_update(ET_CA3,PT_CA1,plateau_flag_CA1,w_CA3_CA1,connectivity_CA3_CA1,BTSP_scaling_CA1)

                    # Time decay
                    ET_CA3 -= ET_CA3 * (dt / tpre)
                    PT_CA3 -= PT_CA3 * (dt / tpre)
                    PT_CA1 -= PT_CA1 * (dt / tpost)
                
                if (lap%save_lap == 0)and(current_unit_ID+1 in pause_state):
                    file_out = os.path.join(file_dir,"data",foldername,"lap_%d_pause_%d.npz"%(lap,current_unit_ID+1))
                    np.savez_compressed(file_out,
                        error_list=error_list,PS_list=PS_list,
                        w_CA3_CA3=w_CA3_CA3,w_CA3_CA1=w_CA3_CA1,w_CA1_feat=w_CA1_feat)
                    del file_out
                current_position = current_position + actions[action_ID]

            if lap%save_lap == 0:
                file_out = os.path.join(file_dir,"data",foldername,"lap_%d.npz"%lap)
                np.savez_compressed(file_out,
                    error_list=error_list,PS_list=PS_list,
                    w_CA3_CA3=w_CA3_CA3,w_CA3_CA1=w_CA3_CA1,w_CA1_feat=w_CA1_feat)
                del file_out

def find_place_cells(mode, trial_number):

    if mode == 0:
        file_dir = os.path.join(base_path,"results/linear_reward")
        from linear_reward_variables import num_CA1_neurons, tot_lap
        from linear_reward_functions import sample_spatial_points, get_tuning_curve, load_PF_starts
    
    elif mode == 1:
        file_dir = os.path.join(base_path,"results/Tmaze")
        from Tmaze_variables import num_CA1_neurons, tot_lap
        from Tmaze_functions import sample_spatial_points, get_tuning_curve, load_PF_starts

    elif mode == 2:
        file_dir = os.path.join(base_path,"results/linear_shock")
        from linear_shock_variables import num_CA1_neurons, tot_lap
        from linear_shock_functions import sample_spatial_points, get_tuning_curve, load_PF_starts

    
    print("Identify place cells...")
    place_peak_position = sample_spatial_points(unit_gran)

    for seed in tqdm(range(trial_number)):
        foldername = "trial" + str(seed)
        CA3_place_fields = load_PF_starts(os.path.join(file_dir, "PF_peak_data.pkl"))

        for load_episode in range(1,tot_lap+1):
            w_CA3_CA1 = np.load(os.path.join(base_path,"files",foldername,"simul_trial_%d_exp.npz"%(load_episode)))["w_CA3_CA1"]
            CA1_activity = np.zeros((place_peak_position.shape[0],num_CA1_neurons))

            for ll in range(place_peak_position.shape[0]):
                position = place_peak_position[ll]
                CA3_FR = get_tuning_curve(position, list(CA3_place_fields.values()))*infield_rate*theta_mod_factor
                CA1_activity[ll,:] = input_driven_rate(np.arange(num_CA1_neurons), CA3_FR, w_CA3_CA1, rate_shift=5)

            np.savez_compressed(os.path.join(base_path, "files", foldername, "activity/CA1_activity_lap_"+str(load_episode)+".npz"),CA1_activity=CA1_activity, place_thr_FR=place_thr_FR, unit_gran=unit_gran)

            place_cell_idx = np.where((np.max(CA1_activity,axis=0)>=place_thr_FR))[0]
            place_field_sorted_idx = place_cell_idx[np.argsort(np.argmax(CA1_activity[:,place_cell_idx],axis=0))]
            place_peak_idx = find_PF_peak(CA1_activity)
            CA1_place_fields = {neuron_id:np.array(place_peak_position[place_peak_idx[neuron_id]]) for neuron_id in place_field_sorted_idx}
            print("%d place cell identified after lap %d!"%(len(place_cell_idx),load_episode))

            pklf_name = os.path.join(file_dir, "data", foldername, "detected_PC/CA1_PF_lap_"+str(load_episode)+".pkl")
            save_place_fields(CA1_place_fields, pklf_name)