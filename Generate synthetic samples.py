import pandas as pd
from pyDOE import *
from scipy.stats.distributions import norm

# Read Date
allcoo = pd.read_excel('./your coordinates.xlsx', sheet_name=0)
allcoo = allcoo.values

# Defining parameters
mean = 8
sd = 2
sofx = 50
sofy = 50

# Cholieusky decomposition
correlation_matrix = np.zeros(shape=(len(allcoo), len(allcoo)))
for i in range(0, len(allcoo)):
    d_x = abs(allcoo[i, 1] - allcoo[:, 1])
    d_y = abs(allcoo[i, 2] - allcoo[:, 2])
    correlation_matrix[i, :] = np.exp(-((2*d_x/sofx)+(2*d_y/sofy)))
L = np.array(np.linalg.cholesky(np.mat(correlation_matrix)))
lhd = lhs(1, samples=len(allcoo))
lhd = norm(loc=0, scale=1).ppf(lhd)
lhd = np.array(lhd).reshape(-1, 1)
standard_matrix = np.dot(L, lhd)
synthetic_sample = mean + sd * standard_matrix
