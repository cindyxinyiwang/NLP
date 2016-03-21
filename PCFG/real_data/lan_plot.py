import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

from scipy.stats import poisson, nbinom, lognorm, norm, entropy
from scipy import stats
import statsmodels.api as sm

class lan_plot():
	def __init__(self, filename=None, probs=None, d_list=None):
		self.probs = []
		self.b_prob = [0, 0.0035237700708999546, 0.004540253071801108, 0.005670779362664136, 0.006509449132521744, 0.010003770166938228, 0.015175545661952762, 0.02023282514383079, 0.02492212660337487, 0.02858613120202318, 0.03119180600983595, 0.032644580100564896, 0.03340794865198185, 0.03352666309239694, 0.03311452745438423, 0.03230941022989179, 0.031217276069111545, 0.029932988641918314, 0.028514453053581298, 0.027028120093320965, 0.025517080286036765, 0.024009741301823807, 0.022531131184157702, 0.021097092585539507, 0.019719557341012238, 0.018406124493552837, 0.01716127125772899, 0.015987137935393834, 0.014883736550738491, 0.013849950589958953, 0.012883677868498767, 0.011982217789347336, 0.011142482226450413, 0.010361141026452053, 0.00963478652936984, 0.008959998141594594, 0.008333425213555692, 0.007751825959759251, 0.007212095340986956, 0.006711285482842196, 0.006246612472869113, 0.0058154618524964615, 0.0054153868105329605, 0.005044104503104311, 0.004699490259410519, 0.004379570123118744, 0.004082513087198912, 0.0038066226281486567, 0.0035503283073239278, 0.003312177410313341, 0.003090826806298775, 0.0028850351653677236, 0.002693655523274581, 0.0025156282786914157, 0.0023499746066763315, 0.0021957903013826827, 0.0020522400410035147, 0.0019185520568274555, 0.001794013193884219, 0.0016779643396213032, 0.001569796200946929, 0.0014689454072910348, 0.0013748909176517015, 0.001287150710693388, 0.0012052787369661367, 0.0011288621138335854, 0.0010575185444567257, 0.0009908939433953764, 0.0009286602525418495, 0.000870513432166057, 0.0008161716129993718, 0.0007653733962987129, 0.0007178762898467142, 0.0006734552687829124, 0.0006319014510406163, 0.0005930208779929403, 0.0005566333916734926, 0.0005225716006487069, 0.0004906799272734553, 0.00046081372966572023, 0.00043283849229246494, 0.0004066290795693807, 0.000382069047346403, 0.0003590500075807051, 0.00033747104189291215, 0.000317238160063031, 0.0002982637998527279, 0.00028046636484271734, 0.0002637697972503061, 0.0002481031829448523, 0.00023340038610999414, 0.00021959971121278724, 0.00020664359013314562, 0.0001944782924836829, 0.00018305365731168644, 0.00017232284452280218, 0.00016224210450129368, 0.00015277056452552218, 0.00014387003069061963, 0.00013550480415407232, 0.00012764151061494573, 0.00012024894202453622]

		if filename:
			self.parse_lan_file(filename)
			self.m = np.mean(self.len_list)
			self.v = np.var(self.len_list)

			self.llength = np.log(self.len_list)
			self.m_log = np.mean(self.llength)
			self.s_log = np.std(self.llength)
		elif probs:
			try:
				self.probs = probs
				self.m = d_list[0]
				self.v = d_list[1]
				self.m_log = d_list[2]
				self.s_log = d_list[3]
			except Exception as ex:
				print "error: " + str(ex)
				exit(1)
		else:
			print "Error: no data specified"
			exit(1)

	def parse_lan_file(self, filename):
		self.len_list = []
		count = {}
		with open(filename, 'r') as myfile:
			for sent in myfile:
				words = sent.split()
				l = 0
				for w in words:
					if w.isalnum():
						l += 1
				if l == 0 or l > 100:
					continue
				self.len_list.append(l)
				if l in count:
					count[l] += 1
				else:
					count[l] = 1
		total = len(self.len_list)
		for i in range(100):
			if i in count.keys():
				self.probs.append(count[i]/(total+0.0))
			else:
				self.probs.append(0)

	def make_loglen_plot(self):
		"""
		hist, bin_edges = np.histogram(self.llength, bins = range(10))
		plt.bar(bin_edges[:-1], hist, width = 1)
		plt.xlim(min(bin_edges), max(bin_edges))
		"""

		plt.show()

	def make_real_plot(self):
		self.real_plt, = plt.plot([i for i in range(100)], self.probs, linewidth=3, color='g')
		plt.xlim([0, 100])

	def make_parser_plot(self):
		self.line_gram, = plt.plot([ i for i in range(len(self.b_prob))], self.b_prob, linewidth=2,color='b')

	def make_lognorm_plot(self):
		#mu, sigma = self.m_log, self.s_log
		#sigma, _, mu  = lognorm.fit(self.len_list, floc=0)
		mu, sigma = norm.fit(np.log(self.len_list))
		#print mu, sigma
		x = np.linspace(1, 100, 100)
		pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
		self.line_lognorm, = plt.plot(x, pdf, linewidth=2, color='r')

		self.lognorm_real_kl = entropy(self.probs, pdf)
		self.lognorm_grammar_kl = entropy(self.b_prob[:100], pdf)

	def make_poisson_plot(self):
		mu = self.m
		dist = poisson(mu)
		x = np.arange(0, 100)
		pmf = dist.pmf(x)
		self.line_poission, = plt.plot(x, pmf, ls="--", linewidth=2)

		self.poisson_real_kl = entropy(self.probs, pmf)

	def make_nb_plot(self):
		self._get_nb_estimate()
		p = self.nb_prob
		n = self.nb_size
		x = np.arange(0, 100)
		pmf = nbinom.pmf(x, n, p)
		self.line_neg_binomial, = plt.plot(x, pmf, ls=":", linewidth=2)

		self.nb_real_kl = entropy(self.probs, pmf)
		self.nb_grammar_kl = entropy(self.b_prob[:100], pmf)

	def _get_nb_estimate(self):
		y = self.len_list
		x = np.ones(len(self.len_list))

		loglike_method = 'nb1'  # or use 'nb2'
		res = sm.NegativeBinomial(y, x, loglike_method=loglike_method).fit(start_params=[0.1, 0.1])
		mu = res.predict()   # use this for mean if not constant

		mu = np.exp(res.params[0])   # shortcut, we just regress on a constant
		alpha = res.params[1]
		
		if loglike_method == 'nb1':
		    Q = 1
		elif loglike_method == 'nb2':    
		    Q = 0
		
		self.nb_size = 1. / alpha * mu**Q
		self.nb_prob = self.nb_size / (self.nb_size + mu)

	def plot(self):
		self.make_real_plot()
		self.make_parser_plot()
		self.make_lognorm_plot()
		print "lognorm to real data: ", self.lognorm_real_kl
		print "lognorm to grammar: ", self.lognorm_grammar_kl
		self.make_nb_plot()
		print "nb to real data: ", self.nb_real_kl
		print "nb to grammar: ", self.nb_grammar_kl

		self.grammar_real_kl = entropy(self.b_prob[:100], self.probs)
		print "grammar to real: ", self.grammar_real_kl
		plt.legend([self.line_lognorm, self.line_gram, self.real_plt, self.line_neg_binomial], [ 'lognormal',  'grammar', 'Treebank data', 'Negative Binomial'])
		plt.show()

if __name__=="__main__":
	p = lan_plot(filename="sec02-21.words")
	p.plot()