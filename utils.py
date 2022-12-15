from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chisquare, chi2_contingency, wasserstein_distance, tstd
from scipy.spatial import distance
import numpy as np
import pandas as pd
import random

significant_digits = 3

def compute_optimal_histogram_bin_edges(x_array,y_array):
    x_bin_edges=np.histogram_bin_edges(x_array, bins='auto')
    y_bin_edges=np.histogram_bin_edges(y_array, bins='auto')
    
    #select the smaller amount of bins to ensure both x_array and y_array have well populated histograms 
    min_number_of_bins = min(len(x_bin_edges), len(y_bin_edges))

    #create bin edges appropriate for both distributions
    min_value = min(np.amin(x_array), np.amin(y_array))
    max_value = min(np.amax(x_array), np.amax(y_array))

    bin_edges=np.linspace(min_value, max_value, min_number_of_bins)
    return bin_edges


def numerical_data_distribution_plot(x_array,y_array):

    bin_edges = compute_optimal_histogram_bin_edges(x_array, y_array)

    fig = plt.figure(figsize=(10, 5))

    xlabel = 'Value'
    title = ( 
        f'JSD = {jensen_shannon_numerical(x_array, y_array):.3f} \n'
        f'NWD = {normed_wasserstein_numerical(x_array, y_array):.3f} \n'
        )

    sns.histplot(data=x_array, bins=bin_edges, legend=False, stat='density', color='blue').set(title=title,xlabel=xlabel)
    sns.histplot(data=y_array, bins=bin_edges, legend=False, stat='density', color='orange')
    
    return fig


def categorical_grouped_bar_plot(x_list, y_list, title="", normalize = True):
    xcounts = pd.Series(x_list).value_counts(normalize = normalize).to_frame()
    xcounts['source'] = 'Baseline'
    
    ycounts = pd.Series(y_list).value_counts(normalize = normalize).to_frame()
    ycounts['source'] = 'Production'

    combined_counts = pd.concat([xcounts,ycounts])
    combined_counts.reset_index(inplace=True)
    combined_counts.columns = ['Category','Density','Source']

    fig = plt.figure(figsize=(10, 5))

    title= f'JSD = {jensen_shannon_categorical(x_list, y_list):.3f}'

    sns.barplot(x='Category', y='Density', hue='Source', data=combined_counts).set_title(title)
    plt.legend(loc='lower right')
    
    return fig


def jensen_shannon_numerical(x_array, y_array):
    bin_edges = compute_optimal_histogram_bin_edges(x_array, y_array)

    x_hist, x_bin_edges = np.histogram(x_array, density = True, bins=bin_edges)
    y_hist, y_bin_edges = np.histogram(y_array, density = True, bins=bin_edges)

    x_hist_list = x_hist.tolist()
    y_hist_list = y_hist.tolist()

    return round(distance.jensenshannon(x_hist_list, y_hist_list), significant_digits)


def normed_wasserstein_numerical(x_array, y_array):

    #x_std = tstd(x_array)
    #y_std = tstd(y_array)
    
    #watch out! both are normalized by x_std
    #normed_x_array = x_array / x_std
    #normed_y_array = y_array / x_std

    bin_edges = compute_optimal_histogram_bin_edges(x_array, y_array)

    x_hist, x_bin_edges = np.histogram(x_array, density = True, bins=bin_edges)
    y_hist, y_bin_edges = np.histogram(y_array, density = True, bins=bin_edges)

    x_hist_list = x_hist.tolist()
    y_hist_list = y_hist.tolist()

    norm = tstd(x_hist_list)

    return round(wasserstein_distance(x_hist_list, y_hist_list) / norm, significant_digits) 


def jensen_shannon_categorical(x_list, y_list):
    values = set(x_list + y_list)
        
    x_freqs = np.array([x_list.count(value) for value in values])
    y_freqs = np.array([y_list.count(value) for value in values])
    
    x_ratios = x_freqs / np.sum(x_freqs)  #Optional as JS-D normalizes probability vectors
    y_ratios = y_freqs / np.sum(y_freqs)

    return round(distance.jensenshannon(x_ratios, y_ratios), significant_digits)