from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chisquare, chi2_contingency, wasserstein_distance, tstd
from scipy.spatial import distance
import numpy as np
import pandas as pd
import random
from yaml import safe_load


config = safe_load(open('config.yml','rb'))


# --- BINNING FUNCTIONS ---


def compute_optimal_histogram_bin_edges(x_array,y_array):
    
    #find overall min and max values for both distributions
    min_value = min(np.amin(x_array), np.amin(y_array))
    max_value = min(np.amax(x_array), np.amax(y_array))
    
    #add min and max value to both distributions in order estimate the appropriate number of bins
    #for both samples that span the entire range of data.
    if min_value not in x_array:
        x_array = np.append(x_array,min_value)
    if min_value not in y_array:
        y_array = np.append(y_array,min_value)    
    if max_value not in x_array:
        x_array = np.append(x_array,max_value)
    if max_value not in y_array:
        y_array = np.append(y_array,max_value)    

    #compute optimal bin edges for both distributions after having been extended to [min_value,max_value]
    x_bin_edges=np.histogram_bin_edges(x_array, bins='auto')
    y_bin_edges=np.histogram_bin_edges(y_array, bins='auto')
    
    #select the smaller amount of bins to ensure both x_array and y_array have well populated histograms 
    min_number_of_bins = min(len(x_bin_edges), len(y_bin_edges))

    bin_edges=np.linspace(min_value, max_value, min_number_of_bins)
    return bin_edges


# --- HISTOGRAM PLOTTING FUNCTIONS ---

def numerical_data_distribution_plot(x_array,y_array):

    bin_edges = compute_optimal_histogram_bin_edges(x_array, y_array)

    fig = plt.figure(figsize=(10, 5))

    xlabel = 'Value'
    title = ( 
        f'{drift_evaluator(x_array,y_array,"jensen_shannon_distance_numerical")} \n'
        f'{drift_evaluator(x_array,y_array,"normed_wasserstein_distance_numerical")} \n'
        f'{drift_evaluator(x_array,y_array,"two_sample_ks_test_numerical")} \n'
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

    title= (
        f'{drift_evaluator(x_list,y_list,"jensen_shannon_distance_categorical")} \n'
        f'{drift_evaluator(x_list,y_list,"chi_squared_test_categorical")} \n'
        )
    
    sns.barplot(x='Category', y='Density', hue='Source', data=combined_counts).set_title(title)
    plt.legend(loc='lower right')
    
    return fig



# --- STATISTICAL TESTS FOR NUMERICAL FEATURES ---

def two_sample_ks_test_numerical(x_array, y_array):
    return ks_2samp(x_array, y_array).pvalue


# --- STATISTICAL DISTANCE FUNCTIONS FOR NUMERICAL FEATURES ---


def jensen_shannon_distance_numerical(x_array, y_array):
    bin_edges = compute_optimal_histogram_bin_edges(x_array, y_array)

    x_hist, x_bin_edges = np.histogram(x_array, density = True, bins=bin_edges)
    y_hist, y_bin_edges = np.histogram(y_array, density = True, bins=bin_edges)

    x_hist_list = x_hist.tolist()
    y_hist_list = y_hist.tolist()

    return distance.jensenshannon(x_hist_list, y_hist_list)


def normed_wasserstein_distance_numerical(x_array, y_array):

    bin_edges = compute_optimal_histogram_bin_edges(x_array, y_array)

    x_hist, x_bin_edges = np.histogram(x_array, density = True, bins=bin_edges)
    y_hist, y_bin_edges = np.histogram(y_array, density = True, bins=bin_edges)

    x_hist_list = x_hist.tolist()
    y_hist_list = y_hist.tolist()

    norm = tstd(x_hist_list)

    return wasserstein_distance(x_hist_list, y_hist_list) / norm

# --- STATISTICAL DISTANCE FUNCTIONS FOR CATEGORICAL FEATURES ---


def jensen_shannon_distance_categorical(x_list, y_list):
    
    # unique values observed in x and y
    values = set(x_list + y_list)
        
    x_freqs = np.array([x_list.count(value) for value in values])
    y_freqs = np.array([y_list.count(value) for value in values])
    
    x_ratios = x_freqs / np.sum(x_freqs)  #Optional as JS-D normalizes probability vectors
    y_ratios = y_freqs / np.sum(y_freqs)

    return distance.jensenshannon(x_ratios, y_ratios)

# --- STATISTICAL TESTS FOR NUMERICAL FEATURES ---

def chi_squared_test_categorical(x_list, y_list):
    values = set(x_list + y_list)
        
    x_counts = np.array([x_list.count(value) for value in values])
    y_counts = np.array([y_list.count(value) for value in values])

    y_ratios = y_counts / np.sum(y_counts)
    expected_y_counts = y_ratios * len(x_list)

    return chisquare(x_counts, expected_y_counts).pvalue



# --- DRIFT EVALUATOR ---

def drift_evaluator(x,y,drift_test):

    drift_status = ""
    acro = { 'jensen_shannon_distance_numerical':'JS',
            'normed_wasserstein_distance_numerical':'Wasserstein',
            'jensen_shannon_distance_categorical':'JS',
            'two_sample_ks_test_numerical':'KS Test',
            'chi_squared_test_categorical':'Chi^2 Test'
    }

    if drift_test in ['jensen_shannon_distance_numerical','normed_wasserstein_distance_numerical', 'jensen_shannon_distance_categorical']:
        distance_measure = globals()[drift_test] # Select the appropriate function based on string
        dist = distance_measure(x,y)
        if dist > float(config['thresholds'][drift_test]):
            drift_status = "drift detected"
        else:
            drift_status = "no drift"
        return f"{acro[drift_test]} Distance: {round(dist, config['significant_digits'])}; max:{config['thresholds'][drift_test]}; status:{drift_status}"

    if drift_test in ['two_sample_ks_test_numerical','chi_squared_test_categorical']:
        stat_test = globals()[drift_test] # Select the appropriate function based on string
        pvalue = stat_test(x,y)
        if pvalue < float(config['thresholds'][drift_test]):
            drift_status = "drift detected"
        else:
            drift_status = "no drift"
        return f"{acro[drift_test]} p-value:{round(pvalue, config['significant_digits'])}; min:{config['thresholds'][drift_test]}; status:{drift_status}"