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

def generate_cue_spikes(spike_rates, duration, round_dec=3):
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

def simulate_offline_spikes(mode, target_lap, trial_number, pause_state=None, use_example=False):
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
        foldername = "example"+str(trial) if use_example else "trial"+str(trial)
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

def find_optimal_parameters(mode, target_trial, target_lap, pause_state=-1, offspring_size=50, max_ngen=10, use_example=False):
    
    if mode == 0:
        from linear_reward_functions import load_PF_starts
        file_dir = os.path.join(base_path,"results/linear_reward")
    elif mode == 1:
        from Tmaze_functions import load_PF_starts
        file_dir = os.path.join(base_path,"results/Tmaze")
    elif mode == 2:
        from linear_shock_functions import load_PF_starts
        file_dir = os.path.join(base_path,"results/linear_shock")

    foldername = "example%d"%target_trial if use_example else "trial%d"%target_trial
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
    _, hof, _, _ = opt.run(max_ngen=max_ngen, cp_filename=cp_f_name)
    del pool, opt

    best = hof[0]
    for pname, value in zip(pnames, best):
        print("%s = %.3f" % (pname, value))
    _ = evaluator.evaluate_with_lists(best)
    np.savez(params_fname, **dict(zip(pnames, best)))
    return best

def detect_replay_type(mode, target_lap, trial_number, use_example=False):

    if mode == 1:
        from Tmaze_functions import analyse_replay_type
        file_dir = os.path.join(base_path,"results/Tmaze")
    else: raise ValueError("This function is only supported to analyze T-maze learning scenario!")

    for trial in tqdm(range(trial_number),desc="Trial: "):
        foldername = "example%d"%trial if use_example else "trial%d"%trial
        save_path = os.path.join(file_dir,foldername)
        ff = np.load(os.path.join(save_path,"CA3_replay_lap_%d_pause_4.npz"%(target_lap)))
        spike_times = ff['spike_times_CA3_PC']; spiking_neurons = ff['spiking_neurons_CA3_PC']; rate = ff['rate_CA3_PC']
        del ff
        
        analyse_replay_type(spike_times, spiking_neurons, rate, save_path=save_path)

def offline_simulation(mode, wmx_CA3, wmx_CA1, pause_state, PF, w_PC_IN_CA3, w_IN_PC_CA3, w_IN_IN_CA3, wmx_mult_CA3, w_PC_MF_CA3, rate_MF, w_PC_IN_CA1, w_IN_PC_CA1, w_IN_IN_CA1, wmx_mult_CA1, verbose=False):

    if mode == 0:
        from linear_reward_variables import rest_time, num_CA3_neurons, num_CA1_neurons, state_position
        from linear_reward_functions import generate_place_cell_ID_list, get_tuning_curve
    elif mode == 1:
        from Tmaze_variables import rest_time, num_CA3_neurons, num_CA1_neurons, state_position
        from Tmaze_functions import generate_place_cell_ID_list, get_tuning_curve
    elif mode == 2:
        from linear_shock_variables import rest_time, num_CA3_neurons, num_CA1_neurons, state_position
        from linear_shock_functions import generate_place_cell_ID_list, get_tuning_curve

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

class NetworkEvaluator(bpop.evaluators.Evaluator):
    """Evaluator class required by BluePyOpt"""

    def __init__(self, mode, W_CA3, W_CA1, cue, PF, params):
        """
        :param W_CA3: weight matrix (passing Wee with cPickle to the slaves (as BluPyOpt does) is still the fastest solution)
        :param params: list of parameters to fit - every entry must be a tuple: (name, lower bound, upper bound)
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

def analyze_related_params(RM_PC, RM_BC, len_sim=5000):
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

def train_network(mode, target_lap, trial_number, pause_state=None, replay_type=False, use_example=False):
    
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
        foldername = "example%d"%trial if use_example else "trial%d"%trial
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

def update_STDP(init_w, spk_neuron_pre, spk_neuron_post, spk_time_pre, spk_time_post, connectivity, timepoint_interest, tau=30, eta=1e-3):
    from global_variables import dt, wmax

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
    for tt in range(timepoint_interest.max()):
        spiking_CA3 = target_neurons_pre[np.where(target_times_pre==tt)[0]]
        spiking_CA1 = target_neurons_post[np.where(target_times_post==tt)[0]]

        ET_pre[spiking_CA3] += dApre
        ET_post[spiking_CA1] += dApost

        w[spiking_CA3,:] += ET_pre[None, :]
        w[:,spiking_CA1] += ET_post[:, None]
        w = np.clip(w, 0, wmax)

        ET_pre -= ET_pre * (dt / taupre)
        ET_post -= ET_post * (dt / taupost)

        ET_pre = np.maximum(ET_pre,0)
        ET_post = np.maximum(ET_post,0)

        if np.isin(tt+1, timepoint_interest): w[connectivity == False] = 0; w_array[count] = w; count+=1
    
    return w_array
