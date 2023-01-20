from evidently.calculations.stattests import jensenshannon_stat_test
import numpy as np
import pandas as pd

# Create two distributions
x = pd.Series(np.random.normal(100, 10, 100_000))
y = pd.Series(np.random.normal(10_000, 10, 100_000))


# Calculate the Jensen-Shannon divergence
print(jensenshannon_stat_test(x, y, feature_type='num',threshold=0.1))