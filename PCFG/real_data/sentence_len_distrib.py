import nltk
from nltk.corpus import brown, gutenberg
from nltk import sent_tokenize, word_tokenize

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

from scipy.stats import poisson, nbinom

datafile = "sec02-21.words"
data = ""
total = 0
count = {}
len_list = []
probs = []

"""
with open(datafile, 'r') as myfile:
	l1 = myfile.readline()
	l1.strip()
	probs = l1.split(",")
	probs = [i.strip() for i in probs]
	probs[0] = probs[0][1:]
	probs[-1] = probs[-1][:-1]

	l2 = myfile.readline()
	l2.strip()
	len_list = l2.split(",")
	len_list = [i.strip for i in len_list]
	len_list[0] = len_list[0][1:]
	len_list[-1] = len_list[-1][:-1]

	probs = [float(i) for i in probs]
	len_list = [int(i) for i in len_list]
"""
with open(datafile, 'r') as myfile:
	for sent in myfile:
		total += 1
		l = len(sent.split(" "))
		len_list.append(l)
		if l in count:
			count[l] += 1
		else:
			count[l] = 1
	data = myfile.read().replace('\n', '')


#probs = [x/(total+0.0) for x in count.values()]
probs = []
for i in range(100):
	if i in count.keys():
		probs.append(count[i]/(total+0.0))
	else:
		probs.append(0)

#real_plt, = plt.plot(count.keys(), probs, linewidth=3, color='g')
real_plt, = plt.plot([i for i in range(100)], probs, linewidth=3, color='g')
plt.xlim([0, 100])
#plt.savefig("word_len.png")
m = np.mean(len_list)
v = np.var(len_list)
llength = np.log(len_list)
#m_log = np.log( m / np.sqrt(1+v/(m**2)) )
#s_log = np.sqrt( np.log(1+v/(m**2)) )
m_log = np.mean(llength)
s_log = np.std(llength)

#plot log normal distribution
#mu, sigma = 3.25, 0.45 # mean and standard deviation
mu, sigma = m_log, s_log
print m_log, s_log, m, v
x = np.linspace(1, 100, 10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
line_lognorm, = plt.plot(x, pdf, linewidth=2, color='r')

#plot normal distribution
line_norm, = plt.plot(x, mlab.normpdf(x, m, np.sqrt(v)), linewidth=2, color='k')

#plot poisson distribution
mu = m
dist = poisson(mu)
x = np.arange(0, 100)
line_poission, = plt.plot(x, dist.pmf(x), ls="--", linewidth=2)

#plot negative poisson
p = 1- m/v
n = m*(1-p)/p
print n, p
line_neg_poisson, = plt.plot(x, nbinom.pmf(x, n, p), ls=":", linewidth=2)

# plot probabilites obtained by berkerly parser
b_prob = [0, 0.0035237700708999546, 0.004540253071801108, 0.005670779362664136, 0.006509449132521744, 0.010003770166938228, 0.015175545661952762, 0.02023282514383079, 0.02492212660337487, 0.02858613120202318, 0.03119180600983595, 0.032644580100564896, 0.03340794865198185, 0.03352666309239694, 0.03311452745438423, 0.03230941022989179, 0.031217276069111545, 0.029932988641918314, 0.028514453053581298, 0.027028120093320965, 0.025517080286036765, 0.024009741301823807, 0.022531131184157702, 0.021097092585539507, 0.019719557341012238, 0.018406124493552837, 0.01716127125772899, 0.015987137935393834, 0.014883736550738491, 0.013849950589958953, 0.012883677868498767, 0.011982217789347336, 0.011142482226450413, 0.010361141026452053, 0.00963478652936984, 0.008959998141594594, 0.008333425213555692, 0.007751825959759251, 0.007212095340986956, 0.006711285482842196, 0.006246612472869113, 0.0058154618524964615, 0.0054153868105329605, 0.005044104503104311, 0.004699490259410519, 0.004379570123118744, 0.004082513087198912, 0.0038066226281486567, 0.0035503283073239278, 0.003312177410313341, 0.003090826806298775, 0.0028850351653677236, 0.002693655523274581, 0.0025156282786914157, 0.0023499746066763315, 0.0021957903013826827, 0.0020522400410035147, 0.0019185520568274555, 0.001794013193884219, 0.0016779643396213032, 0.001569796200946929, 0.0014689454072910348, 0.0013748909176517015, 0.001287150710693388, 0.0012052787369661367, 0.0011288621138335854, 0.0010575185444567257, 0.0009908939433953764, 0.0009286602525418495, 0.000870513432166057, 0.0008161716129993718, 0.0007653733962987129, 0.0007178762898467142, 0.0006734552687829124, 0.0006319014510406163, 0.0005930208779929403, 0.0005566333916734926, 0.0005225716006487069, 0.0004906799272734553, 0.00046081372966572023, 0.00043283849229246494, 0.0004066290795693807, 0.000382069047346403, 0.0003590500075807051, 0.00033747104189291215, 0.000317238160063031, 0.0002982637998527279, 0.00028046636484271734, 0.0002637697972503061, 0.0002481031829448523, 0.00023340038610999414, 0.00021959971121278724, 0.00020664359013314562, 0.0001944782924836829, 0.00018305365731168644, 0.00017232284452280218, 0.00016224210450129368, 0.00015277056452552218, 0.00014387003069061963, 0.00013550480415407232, 0.00012764151061494573, 0.00012024894202453622]
line_gram, = plt.plot([ i for i in range(len(b_prob))], b_prob, linewidth=2,color='b')

plt.legend([line_lognorm, line_norm, line_gram, real_plt, line_poission, line_neg_poisson], [ 'lognormal', 'normal', 'grammar', 'Treebank data', 'poisson', 'negative'])
plt.show()