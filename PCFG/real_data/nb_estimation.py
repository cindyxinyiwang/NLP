import numpy as np
from scipy import stats
import statsmodels.api as sm

datafile = "sec02-21.words"
len_list = []
with open(datafile, 'r') as myfile:
	for sent in myfile:
		words = sent.split(" ")
		l = 0
		for w in words:
			if w.isalnum():
				l += 1
		#l = len(sent.split(" "))
		if l == 0 or l > 100:
			continue
		len_list.append(l)


# generate some data to check
"""
nobs = 1000
n, p = 50, 0.25
dist0 = stats.nbinom(n, p)
y = dist0.rvs(size=nobs)
x = np.ones(nobs)
"""
y = len_list
x = np.ones(len(len_list))

loglike_method = 'nb1'  # or use 'nb2'
res = sm.NegativeBinomial(y, x, loglike_method=loglike_method).fit(start_params=[0.1, 0.1])

#print dist0.mean()
print res.params

mu = res.predict()   # use this for mean if not constant
mu = np.exp(res.params[0])   # shortcut, we just regress on a constant
alpha = res.params[1]

if loglike_method == 'nb1':
    Q = 1
elif loglike_method == 'nb2':    
    Q = 0

size = 1. / alpha * mu**Q
prob = size / (size + mu)

#print 'data generating parameters', n, p
print 'estimated params          ', size, prob

#estimated distribution
dist_est = stats.nbinom(size, prob)