# data_drift

This repo contains code to implement a series of data drift tests and distances, as well as the visualization of their underlying distributions.

## Demo Notebook
The demo notebook is `data_drift_demo.ipynb`. It contains a demo of the data drift tests and distances, including visualizations of the underlying distributions. It implements the following test and distances:

**Tests**
- **Two-sided Kolmogorov-Smirnov (KS) Test** - Numerical Features
- **Chi-Squared (Chi^2) Test** - Categorical Features

**Distances**
- **Jensen-Shannon (JS) Distance** - Numerical and Categorical Features
- **Normed Wasserstein Distance** - Numerical Features
- **PSI (Population Stability Index)** - Numerical Features and Categorical Features
- **d_inf (Chebyshev distance on discrete probability distributions)** - Numerical Features and Categorical Features


## Features 
- Automatic bin size selection for continous features based on sample sizes of baseline and test datasets.


## Unit Tests
The unit tests are in `tests.py`. They test the following:
- Assert that drift measures are independent of sample size
    - Distances should be close to 0 for samples of different sizes drawn from the same distribution.
    - Distances should remain roughly constant for samples of different sizes drawn from the same drifted distribution.
- Statistical tests should fail reject the null hypothesis (p-value > alpha) samples of different sizes drawn from the same distribution
- Statistical tests should reject the null hypothesis (p-value < alpha) for samples of different sizes drawn from a drifted distribution
- To be implemented: Assert that same threholds are appropriate regardles of feature value range 
- To be implemented: Assert that PSI metric does not blow up to infinity when a bin has a 0 count.


## Files
 - `config.yml` contains the configuration for data drift threholds.
 - `utils.py` contains functions implementing the tests, distances, and visualizations.
 - `tests.py` contains unit tests for the data drift test/measures.
 - `data_drift_demo.ipynb` contains a demo of the data drift test/measures, including visulizations.

## Enviroment:
- Create the conda environment: `conda env create -f conda_env.yml`
- Activate the environment: `conda activate data_drift`
