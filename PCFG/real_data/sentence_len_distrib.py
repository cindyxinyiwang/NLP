import nltk
from nltk.corpus import brown, gutenberg
from nltk import sent_tokenize, word_tokenize

import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm, entropy

datafile = "sec02-21.words"
data = ""
total = 0
count = {}
len_list = []

with open(datafile, 'r') as myfile:
	for sent in myfile:
		total += 1
		l = len(sent.split(" "))
		len_list.append(l)
		if l in count:
			count[l] += 1
		else:
			count[l] = 1



"""
count = {}
total = 0
fileids = 'carroll-alice.txt'
# l == 2 has the highest probability because of Chapter nubmers!!!!!
for fileids in gutenberg.fileids():
	for sent in gutenberg.sents(fileids):
		l = len(sent)
		#if l > 200:
		#	print " ".join(sent)
		#	print sent
		total += 1
		if l in count:
			count[l] += 1
		else:
			count[l] = 1
"""
"""
max_count = 0
max_len = 0
for k in count:
	if count[k] > max_count:
		max_count = count[k]
		max_len = k
# the following command gives us the result
# max len:  2  max count:  219
#print "max len: ", max_len, " max count: ", max_count
"""
probs = []

for i in range(100):
	if i in count.keys():
		probs.append(count[i]/(total+0.0))
	else:
		probs.append(0)
#plt.bar(count.keys(), [x/(total+0.0) for x in count.values()])
plt.bar(range(100), probs)
plt.xlim([0, 100])
#plt.savefig("word_len.png")


#plot log normal distribution
#mu, sigma = 3.25, 0.45 # mean and standard deviation
mu, sigma = norm.fit(np.log(len_list))
print mu, sigma
x = np.linspace(1, 100, 100)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='r')
fit_kl = entropy(probs, pdf)
print fit_kl

mu1, sigma1 = 3.25, 0.45 # mean and standard deviation
#mu, sigma = norm.fit(np.log(len_list))
print mu1, sigma1
x = np.linspace(1, 100, 100)
pdf1 = (np.exp(-(np.log(x) - mu1)**2 / (2 * sigma1**2)) / (x * sigma1 * np.sqrt(2 * np.pi)))
plt.plot(x, pdf1, linewidth=2, color='b')
pick_kl = entropy(probs, pdf1)
print pick_kl

plt.show()