import nltk
from nltk.corpus import brown, gutenberg
from nltk import sent_tokenize, word_tokenize

import matplotlib.pyplot as plt
import numpy as np

datafile = "sec02-21.words"
data = ""
total = 0
count = {}

with open(datafile, 'r') as myfile:
	for sent in myfile:
		total += 1
		l = len(sent.split(" "))
		if l in count:
			count[l] += 1
		else:
			count[l] = 1
	data = myfile.read().replace('\n', '')


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

plt.bar(count.keys(), [x/(total+0.0) for x in count.values()])
plt.xlim([0, 500])
#plt.savefig("word_len.png")


#plot log normal distribution
mu, sigma = 3.25, 0.45 # mean and standard deviation
x = np.linspace(1, 200, 10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='r')

plt.show()
