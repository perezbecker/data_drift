from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chisquare, chi2_contingency, wasserstein_distance, tstd
from scipy.spatial import distance
import numpy as np
import pandas as pd
import random
from evidently.calculations.stattests import jensenshannon_stat_test, wasserstein_stat_test, psi_stat_test
from yaml import safe_load


config = safe_load(open('config.yml','rb'))


# --- BINNING FUNCTIONS ---


def compute_optimal_histogram_bin_edges(x_array,y_array, bin_strategy="min"):
    
    #find overall min and max values for both distributions
    min_value = min(np.amin(x_array), np.amin(y_array))
    max_value = max(np.amax(x_array), np.amax(y_array))
    
    #add overall min and max value to both distributions in order estimate the appropriate number of bins
    #for both samples that span the entire range of data.
    x_array_extended = np.append(x_array,[min_value, max_value])
    y_array_extended = np.append(y_array,[min_value, max_value])    

    #compute optimal bin edges for both distributions after having been extended to [min_value,max_value]
    x_bin_edges=np.histogram_bin_edges(x_array_extended, bins='auto')
    y_bin_edges=np.histogram_bin_edges(y_array_extended, bins='auto')
    
    #select the amount of bins used. The smaller amount of bins is recommended to ensure both x_array and y_array have well populated histograms 
    if bin_strategy == "min":
        number_of_bins = min(len(x_bin_edges), len(y_bin_edges))
        bin_edges=np.linspace(min_value, max_value, number_of_bins)
    
    elif bin_strategy == "max":
        number_of_bins = max(len(x_bin_edges), len(y_bin_edges))
        bin_edges=np.linspace(min_value, max_value, number_of_bins)

    elif bin_strategy == "evi":
        bin_edges = np.histogram_bin_edges(list(x_array)+list(y_array), bins='sturges')
    
    else:
        raise ValueError("bin_strategy must be either 'min', 'max', or 'evi'")

    return bin_edges


# --- HISTOGRAM PLOTTING FUNCTIONS ---

def numerical_data_distribution_plot(x_array,y_array,bin_strategy="min"):

    bin_edges = compute_optimal_histogram_bin_edges(x_array, y_array, bin_strategy=bin_strategy)

    fig = plt.figure(figsize=(10, 5))

    xlabel = 'Value'
    title = (
        f'Binning Strategy: {bin_strategy} \n' 
        f'{drift_evaluator(x_array,y_array,"jensen_shannon_distance_numerical", bin_strategy=bin_strategy)} \n'
        f'JS Distance Evidently: {evaluate_evidently(jensenshannon_stat_test, x_array, y_array, feature_type="num")} \n'
        #f'{drift_evaluator(x_array,y_array,"jensen_shannon_distance_aml")} \n'
        f'{drift_evaluator(x_array,y_array,"normed_wasserstein_distance_numerical")} \n'
        f'{drift_evaluator(x_array,y_array,"wasserstein_distance_aml")} \n'
        f'Normed Wasserstein Distance Evidently: {evaluate_evidently(wasserstein_stat_test,x_array,y_array, feature_type="num")} \n'
        f'{drift_evaluator(x_array,y_array,"psi_numerical", bin_strategy=bin_strategy)} \n'
        f'PSI Evidently: {evaluate_evidently(psi_stat_test,x_array,y_array, feature_type="num")} \n'
        f'{drift_evaluator(x_array,y_array,"d_inf_numerical", bin_strategy=bin_strategy)} \n'
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
        f'JS Distance Evidently: {evaluate_evidently(jensenshannon_stat_test, x_list, y_list, feature_type="cat")} \n'
        f'{drift_evaluator(x_list,y_list,"psi_categorical")} \n'
        f'PSI Evidently: {evaluate_evidently(psi_stat_test, x_list, y_list, feature_type="cat")} \n'
        f'{drift_evaluator(x_list,y_list,"d_inf_categorical")} \n'
        f'{drift_evaluator(x_list,y_list,"chi_squared_test_categorical")} \n'
        )
    
    sns.barplot(x='Category', y='Density', hue='Source', data=combined_counts).set_title(title)
    plt.legend(loc='lower right')
    
    return fig



# --- STATISTICAL TESTS FOR NUMERICAL FEATURES ---

def two_sample_ks_test_numerical(x_array, y_array):
    return ks_2samp(x_array, y_array).pvalue


# --- STATISTICAL DISTANCE FUNCTIONS FOR NUMERICAL FEATURES ---


def jensen_shannon_distance_numerical(x_array, y_array, bin_strategy='min'):
    bin_edges = compute_optimal_histogram_bin_edges(x_array, y_array,bin_strategy=bin_strategy)
    x_percent = np.histogram(x_array, bins=bin_edges)[0] / len(x_array)
    y_percent = np.histogram(y_array, bins=bin_edges)[0] / len(y_array)

    return distance.jensenshannon(x_percent, y_percent)


def jensen_shannon_distance_aml(x_array, y_array):
    values_per_column = list(set(np.unique(x_array)) | set(np.unique(y_array)))
    current_frequencies = np.array([list(x_array).count(value) for value in values_per_column])
    reference_frequencies = np.array([list(y_array).count(value) for value in values_per_column])
    current_ratios = current_frequencies / np.sum(current_frequencies)
    reference_ratios = reference_frequencies / np.sum(reference_frequencies)
    return distance.jensenshannon(current_ratios, reference_ratios)


def normed_wasserstein_distance_numerical(x_array, y_array):
    norm = np.std(x_array)
    return wasserstein_distance(x_array, y_array) / norm

def wasserstein_distance_aml(x_array, y_array):
    return wasserstein_distance(x_array, y_array)


def psi_numerical(x_array, y_array, bin_strategy='min', smoothing='evi'):
    bin_edges = compute_optimal_histogram_bin_edges(x_array, y_array,bin_strategy=bin_strategy)
    
    # two smoothing options to avoid zero values in bins and have the SPI value be finite.
    # laplace smoothing by incrementing the count of each bin by 1 (see https://docs.fiddler.ai/docs/data-drift)
    # an alternative is to increment percentage bins by epsilon ~ 0.0001 as done by the feel_zeroes flag in Evidently:
    # https://github.com/evidentlyai/evidently/blob/main/src/evidently/calculations/stattests/utils.py 


    if smoothing == 'fid':
        x_count = np.histogram(x_array, bins=bin_edges)[0]
        y_count = np.histogram(y_array, bins=bin_edges)[0] 

        x_count = x_count + 1
        y_count = y_count + 1 

        x_percent = x_count / len(x_array) 
        y_percent = y_count / len(y_array)   

        psi = 0.0
        for i in range(len(x_percent)):
            psi += (y_percent[i] - x_percent[i]) * np.log(y_percent[i] / x_percent[i])
    
    elif smoothing == 'evi':
        x_percent = np.histogram(x_array, bins=bin_edges)[0] / len(x_array)
        y_percent = np.histogram(y_array, bins=bin_edges)[0] / len(y_array)

        np.place(x_percent, x_percent == 0, config['laplace_smoothing_epsilon'])
        np.place(y_percent, y_percent == 0, config['laplace_smoothing_epsilon'])

        psi = 0.0
        for i in range(len(x_percent)):
            psi += (y_percent[i] - x_percent[i]) * np.log(y_percent[i] / x_percent[i])    
    else:
        raise ValueError("smoothing must be either 'fid' or 'evi'")
    
    return psi

def d_inf_numerical(x_array, y_array, bin_strategy='min'):
    bin_edges = compute_optimal_histogram_bin_edges(x_array, y_array,bin_strategy=bin_strategy)
    x_percent = np.histogram(x_array, bins=bin_edges)[0] / len(x_array)
    y_percent = np.histogram(y_array, bins=bin_edges)[0] / len(y_array)
    return distance.chebyshev(x_percent, y_percent) 

# --- STATISTICAL DISTANCE FUNCTIONS FOR CATEGORICAL FEATURES ---

def jensen_shannon_distance_categorical(x_list, y_list):
    
    # unique values observed in x and y
    values = set(x_list + y_list)
        
    x_counts = np.array([x_list.count(value) for value in values])
    y_counts = np.array([y_list.count(value) for value in values])
    
    x_ratios = x_counts / np.sum(x_counts)  #Optional as JS-D normalizes probability vectors
    y_ratios = y_counts / np.sum(y_counts)

    return distance.jensenshannon(x_ratios, y_ratios)

def psi_categorical(x_list, y_list, smoothing = 'evi'):
        
    # unique values observed in x and y
    values = set(x_list + y_list)
        
    x_counts = np.array([x_list.count(value) for value in values])
    y_counts = np.array([y_list.count(value) for value in values])


    if smoothing == 'fid':
    
        x_counts = x_counts + 1
        y_counts = y_counts + 1

        x_ratios = x_counts / np.sum(x_counts)
        y_ratios = y_counts / np.sum(y_counts)

        psi = 0
        for i in range(len(x_ratios)):
            psi += (y_ratios[i] - x_ratios[i]) * np.log(y_ratios[i] / x_ratios[i])
    
    elif smoothing == 'evi':

        x_ratios = x_counts / np.sum(x_counts)
        y_ratios = y_counts / np.sum(y_counts)

        psi = 0
        for i in range(len(x_ratios)):
            if x_ratios[i] == 0:
                x_ratios[i] = config['laplace_smoothing_epsilon']
            if y_ratios[i] == 0:
                y_ratios[i] = config['laplace_smoothing_epsilon']
            
            psi += (y_ratios[i] - x_ratios[i]) * np.log(y_ratios[i] / x_ratios[i])    
    else:
        raise ValueError("smoothing must be either 'fid' or 'evi'")
    
    return psi

def d_inf_categorical(x_list, y_list):

    # unique values observed in x and y
    values = set(x_list + y_list)
        
    x_counts = np.array([x_list.count(value) for value in values])
    y_counts = np.array([y_list.count(value) for value in values])

    x_ratios = x_counts / np.sum(x_counts)
    y_ratios = y_counts / np.sum(y_counts)

    return distance.chebyshev(x_ratios, y_ratios)

# --- STATISTICAL TESTS FOR NUMERICAL FEATURES ---

def chi_squared_test_categorical(x_list, y_list):
    values = set(x_list + y_list)
        
    x_counts = np.array([x_list.count(value) for value in values])
    y_counts = np.array([y_list.count(value) for value in values])

    y_ratios = y_counts / np.sum(y_counts)
    expected_y_counts = y_ratios * len(x_list)

    return chisquare(x_counts, expected_y_counts).pvalue



# --- DRIFT EVALUATOR ---

def drift_evaluator(x,y,drift_test,bin_strategy='min'):

    drift_status = ""
    acro = { 'jensen_shannon_distance_numerical':'JS',
            'jensen_shannon_distance_aml':'JS_AML',
            'normed_wasserstein_distance_numerical':'Wasserstein',
            'wasserstein_distance_aml':'Wasserstein_AML',
            'psi_numerical':'PSI',
            'd_inf_numerical':'d_inf',
            'jensen_shannon_distance_categorical':'JS',
            'psi_categorical':'PSI',
            'd_inf_categorical':'d_inf',
            'two_sample_ks_test_numerical':'KS Test',
            'chi_squared_test_categorical':'Chi^2 Test'
    }

    if drift_test in ['jensen_shannon_distance_numerical','psi_numerical', 'd_inf_numerical']:
        distance_measure = globals()[drift_test] # Select the appropriate function based on string
        dist = distance_measure(x,y,bin_strategy=bin_strategy)
        if dist > float(config['thresholds'][drift_test]):
            drift_status = "drift detected"
        else:
            drift_status = "no drift"
        #return f"{acro[drift_test]} Distance: {round(dist, config['significant_digits'])}; max:{config['thresholds'][drift_test]}; status:{drift_status}"
        return f"{acro[drift_test]}: {round(dist, config['significant_digits'])}"
    
    if drift_test in ['jensen_shannon_distance_aml','jensen_shannon_distance_categorical', 'normed_wasserstein_distance_numerical', 'wasserstein_distance_aml', 'psi_categorical','d_inf_categorical']:
        distance_measure = globals()[drift_test] # Select the appropriate function based on string
        dist = distance_measure(x,y)
        if dist > float(config['thresholds'][drift_test]):
            drift_status = "drift detected"
        else:
            drift_status = "no drift"
        #return f"{acro[drift_test]} Distance: {round(dist, config['significant_digits'])}; max:{config['thresholds'][drift_test]}; status:{drift_status}"
        return f"{acro[drift_test]}: {round(dist, config['significant_digits'])}"

    if drift_test in ['two_sample_ks_test_numerical','chi_squared_test_categorical']:
        stat_test = globals()[drift_test] # Select the appropriate function based on string
        pvalue = stat_test(x,y)
        if pvalue < float(config['thresholds'][drift_test]):
            drift_status = "drift detected"
        else:
            drift_status = "no drift"
        return f"{acro[drift_test]} p-value:{round(pvalue, config['significant_digits'])}; min:{config['thresholds'][drift_test]}; status:{drift_status}"

# --- EVALUATE EVIDENTLY FUNCTIONS ---

def evaluate_evidently(evidently_distance_function, x_array, y_array, feature_type='num'):
    x_series = pd.Series(x_array) 
    y_series = pd.Series(y_array) 
    drift_score = evidently_distance_function(x_series,y_series, feature_type=feature_type, threshold=0.1).drift_score
    return round(drift_score, config['significant_digits'])