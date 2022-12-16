import pytest
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from utils import ( 
                    numerical_data_distribution_plot, 
                    categorical_grouped_bar_plot, 
                    jensen_shannon_distance_numerical, 
                    jensen_shannon_distance_categorical, 
                    normed_wasserstein_distance_numerical,
                    two_sample_ks_test_numerical,
                    chi_squared_test_categorical,
             )
from yaml import safe_load
config = safe_load(open('config.yml','rb'))

epsilon = config['test_epsilon']
alpha_ks = config['thresholds']['two_sample_ks_test_numerical']
alpha_chi2 = config['thresholds']['chi_squared_test_categorical']



# ------------------------------------------------------------
# NUMERICAL FEATURE TESTS
# ------------------------------------------------------------

# Generate test data
sample_size = 100_000
x1 = np.random.normal(50, 15, sample_size) # reference
x3 = np.random.normal(50, 15, int(sample_size/3)) # reference
x10 = np.random.normal(50, 15, int(sample_size/10)) # reference

y1 = np.random.normal(62, 20, sample_size) # major drift 
y3 = np.random.normal(62, 20, int(sample_size/3)) # major drift 
y10 = np.random.normal(62, 20, int(sample_size/10)) # major drift 



# ------------------------------------------------------------ 

# TEST: Distances should be close to 0 for samples of different sizes drawn from the same distribution

# Applied to Jensen Shannon Distance
def test_jensen_shannon_distances_close_to_0_for_samples_of_different_sizes_drawn_from_the_same_distribution():    
    assert jensen_shannon_distance_numerical(x1, x1) < epsilon
    assert jensen_shannon_distance_numerical(x1, x3) < epsilon
    assert jensen_shannon_distance_numerical(x1, x10) < epsilon

# Applied to Normed_Wasserstein Distance
def test_normed_wasserstein_distances_close_to_0_for_samples_of_different_sizes_drawn_from_the_same_distribution():    
    assert normed_wasserstein_distance_numerical(x1, x1) < epsilon
    assert normed_wasserstein_distance_numerical(x1, x3) < epsilon
    assert normed_wasserstein_distance_numerical(x1, x10) < epsilon

# ------------------------------------------------------------ 


# TEST: Distances should remain roughly constant for samples of different sizes drawn from the same drifted distribution

# Applied to Jensen Shannon Distance
def test_jensen_shannon_distance_remain_roughly_constant_for_samples_of_different_sizes_drawn_from_the_same_drifted_distribution():
    assert abs(jensen_shannon_distance_numerical(x1, y1) - jensen_shannon_distance_numerical(x1, y3)) < epsilon
    assert abs(jensen_shannon_distance_numerical(x1, y1) - jensen_shannon_distance_numerical(x1, y10)) < epsilon


#Applied to Normed Wasserstein Distance 
def test_normed_wasserstein_distance_remain_roughly_constant_for_samples_of_different_sizes_drawn_from_the_same_drifted_distribution():
    assert abs(normed_wasserstein_distance_numerical(x1, y1) - normed_wasserstein_distance_numerical(x1, y3)) < epsilon
    assert abs(normed_wasserstein_distance_numerical(x1, y1) - normed_wasserstein_distance_numerical(x1, y10)) < epsilon

# ------------------------------------------------------------ 


# TEST: Statistical tests should fail reject the null hypothesis (p-value > alpha) for drifted distributions with different sample sizes

# Applied to Kolmogorov-Smirnov test
def test_ks_test_fail_to_reject_null_hypothesis_for_identical_distributions_with_different_sample_sizes():
    assert two_sample_ks_test_numerical(x1, x1) > alpha_ks
    assert two_sample_ks_test_numerical(x1, x3) > alpha_ks
    assert two_sample_ks_test_numerical(x1, x10) > alpha_ks

# ------------------------------------------------------------ 


# TEST: Statistical tests should reject the null hypothesis (p-value < alpha) for drifted distributions with different sample sizes

# Applied to Kolmogorov-Smirnov test
def test_ks_test_reject_null_hypothesis_drifted_distributions_with_different_sample_sizes():
    assert two_sample_ks_test_numerical(x1, y1) < alpha_ks
    assert two_sample_ks_test_numerical(x1, y3) < alpha_ks
    assert two_sample_ks_test_numerical(x1, y10) < alpha_ks
    
# ------------------------------------------------------------ 



# ------------------------------------------------------------
# CATEGORICAL FEATURE TESTS
# ------------------------------------------------------------

#Generate test data
a1=('a'* 1000 + 
    'b'* 1000 + 
    'c'* 1000 + 
    'd'* 1000 + 
    'e'* 1000 + 
    'f'* 1000 + 
    'g'* 1000 + 
    'h'* 1000)

a3=('a'* 333 + 
    'b'* 333 + 
    'c'* 333 + 
    'd'* 333 + 
    'e'* 333 + 
    'f'* 333 + 
    'g'* 333 + 
    'h'* 333)

a10=('a'* 100 + 
     'b'* 100 + 
     'c'* 100 + 
     'd'* 100 + 
     'e'* 100 + 
     'f'* 100 + 
     'g'* 100 + 
     'h'* 100)


b1=('a'* 1200 + 
    'b'* 1500 + 
    'c'* 1000 + 
    'd'* 900 + 
    'e'* 900 + 
    'f'* 600 + 
    'g'* 1100 + 
    'h'* 1100)

b3=('a'* 360 + 
    'b'* 450 + 
    'c'* 300 + 
    'd'* 270 + 
    'e'* 270 + 
    'f'* 180 + 
    'g'* 330 + 
    'h'* 330)

b10=('a'* 120 + 
     'b'* 150 + 
     'c'* 100 + 
     'd'* 90 + 
     'e'* 90 + 
     'f'* 60 + 
     'g'* 110 + 
     'h'* 110)

# convert to lists
a1_list = [letter for letter in a1]
a3_list = [letter for letter in a3]
a10_list = [letter for letter in a10]
b1_list = [letter for letter in b1]
b3_list = [letter for letter in b3]
b10_list = [letter for letter in b10]

# shuffle lists
np.random.shuffle(a1_list)
np.random.shuffle(a3_list)
np.random.shuffle(a10_list)
np.random.shuffle(b1_list)
np.random.shuffle(b3_list)
np.random.shuffle(b10_list)

# ------------------------------------------------------------ 

# TEST: Distances should be 0 for samples of different sizes drawn from the same distribution

# Applied to Jensen Shannon Distance
def test_jensen_shannon_categorical_distance_equal_0_for_samples_of_different_sizes_drawn_from_the_same_distribution():    
    assert jensen_shannon_distance_categorical(a1_list, a1_list) == 0
    assert jensen_shannon_distance_categorical(a1_list, a3_list) == 0
    assert jensen_shannon_distance_categorical(a1_list, a10_list) == 0

# ------------------------------------------------------------ 

# TEST: Distances should remain roughly constant for samples of different sizes drawn from the same drifted distribution

# Applied to Jensen Shannon Distance
def test_jensen_shannon_categorical_distance_remain_roughly_constant_for_samples_of_different_sizes_drawn_from_the_same_drifted_distribution():
    assert abs(jensen_shannon_distance_categorical(a1_list, b1_list) - jensen_shannon_distance_categorical(a1_list, b3_list)) == 0
    assert abs(jensen_shannon_distance_categorical(a1_list, b1_list) - jensen_shannon_distance_categorical(a1_list, b10_list)) == 0


# TEST: Statistical tests should fail reject the null hypothesis (p-value > alpha) for drifted distributions with different sample sizes

# Applied to Chi-Squared test
def test_chi2_test_fail_to_reject_null_hypothesis_for_identical_distributions_with_different_sample_sizes():
    assert chi_squared_test_categorical(a1_list, a1_list) > alpha_chi2
    assert chi_squared_test_categorical(a1_list, a3_list) > alpha_chi2
    assert chi_squared_test_categorical(a1_list, a10_list) > alpha_chi2

# ------------------------------------------------------------ 


# TEST: Statistical tests should reject the null hypothesis (p-value < alpha) for drifted distributions with different sample sizes

# Applied to Chi-Squared test
def test_chi2_test_reject_null_hypothesis_for_drifted_distributions_with_different_sample_sizes():
    assert chi_squared_test_categorical(a1_list, b1_list) < alpha_chi2
    assert chi_squared_test_categorical(a1_list, b3_list) < alpha_chi2
    assert chi_squared_test_categorical(a1_list, b10_list) < alpha_chi2
    
# ------------------------------------------------------------ 