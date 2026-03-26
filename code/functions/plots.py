import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from global_variables import *
from matplotlib.colors import LogNorm

def create_gradient(color1, color2, num_steps):
    """Generates a list of RGB colors for a linear gradient."""
    r_vals = np.linspace(color1[0], color2[0], num_steps)
    g_vals = np.linspace(color1[1], color2[1], num_steps)
    b_vals = np.linspace(color1[2], color2[2], num_steps)
    return [[r,g,b] for r, g, b in zip(r_vals, g_vals, b_vals)]


def average_weight(w, x_dim, y_dim):
    
    pop_size_x = int(w.shape[1] / x_dim)
    pop_size_y = int(w.shape[0] / y_dim)

    mean_wmx = np.zeros((y_dim, x_dim))
    for i in range(y_dim):
        for j in range(x_dim):
            try: tmp = w[int(i*pop_size_y):int((i+1)*pop_size_y), int(j*pop_size_x):int((j+1)*pop_size_x)]
            except: tmp = w[int(i*pop_size_y):, int(j*pop_size_x):]
            mean_wmx[i, j] = np.mean(tmp)

    return mean_wmx

def plot_heatmap(data, fig, ax, title, cmap='inferno', vmin=None, vmax=None):

    x = np.arange(data.shape[1] + 1)
    y = np.arange(data.shape[0] + 1)
    
    i = ax.pcolormesh(x, y, data, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(i)
    ax.set_title(title)

    return fig, ax

def extract_aligned_activity(reference_array, arr_peak_idx, target_window):
    
    # Extend activity to avoid edge slicing
    extended_act = np.vstack([reference_array] * 3)
    # print(extended_act.shape)

    target_act = np.zeros((reference_array.shape[1], 2 * target_window + 1))
    for ll in range(reference_array.shape[1]):
        center = arr_peak_idx[ll] + reference_array.shape[0]
        # print(center - target_window,center + target_window + 1)
        target_act[ll, :] = extended_act[center - target_window:center + target_window + 1, ll]
    
    return target_act

def plot_shaded(xx,mean_val,error_val,ax,color='black'):
    ax.fill_between(xx, mean_val-error_val, mean_val+error_val, alpha=0.2, color=color, linewidth=0)
    ax.plot(xx, mean_val, color=color)

    return ax

def plot_box(data1, data2, ax, colors=['lightblue', 'lightgreen']):
    bp = ax.boxplot([data1, data2], positions=[1, 2], patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticks([1, 2])

    return ax

def create_annot_nonzero(data):
    annot_data = np.full(data.shape, '', dtype=object) # Initialize with empty strings
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] != 0:
                annot_data[i, j] = str("%.1f"%data[i, j])

    return annot_data

def plot_spike(spike_times, spiking_neurons, target_idx, cue_idx=np.array([]), zoom_from=0, zoom_to=None, color_='blue', dotsize=1.5, ax=None):
    
    num_neuron = len(target_idx); order_arr = np.zeros(target_idx.max()+1)
    idx = np.where((spike_times > zoom_from)&(spike_times < zoom_to)&np.isin(spiking_neurons, target_idx))
    spike_times = spike_times[idx]; spiking_neurons = spiking_neurons[idx]
    idx = np.isin(spiking_neurons, target_idx)
    for ll in range(num_neuron): order_arr[target_idx[ll]]=ll
    reordered_spiking_neurons = [order_arr[i] for i in spiking_neurons]
    
    if len(cue_idx) > 0:
        reordered_cue_idx = [order_arr[i] for i in cue_idx]
    del ll, order_arr

    if len(cue_idx) > 0:
        cue_cell_targ = np.where(np.isin(spiking_neurons,reordered_cue_idx))[0]
        ax.scatter(np.array(spike_times)[cue_cell_targ], np.array(reordered_spiking_neurons)[cue_cell_targ], c="red", marker="o", s=dotsize)

        non_cue_targ = np.where(~np.isin(spiking_neurons,reordered_cue_idx))[0]
        ax.scatter(np.array(spike_times)[non_cue_targ], np.array(reordered_spiking_neurons)[non_cue_targ], c=color_, marker="o", s=dotsize)

    else:
        ax.scatter(np.array(spike_times), np.array(reordered_spiking_neurons), c=color_, marker="o", s=dotsize)

    if zoom_to == None: zoom_to = spike_times.max() + 10

    ax.set_xlim([zoom_from, zoom_to])
    ax.set_ylabel("Neuron ID")
    ax.set_xlabel("Time (ms)")
    
    return ax