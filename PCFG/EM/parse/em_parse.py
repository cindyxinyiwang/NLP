import numpy as np
import matplotlib.pyplot as plt
import math
import sys

from scipy.stats import nbinom

class GrammarGen():
	def __init__(self, alp):
		self.alphabet = alp
		self.rule_dict = {}
		self.start = alp[0]

	def getAllRules(self):
		"""
		output: all possible rules in chomsky normal form
		"""
		# get a combination of all rules
		rules = []
		for n in self.alphabet:
			for i in self.alphabet:
				rules.append(n + " " + i)

		for n in self.alphabet:
			if n == self.start:
				self.rule_dict[n] = self._get_rules(rules, nonterm=False)
			else:
				self.rule_dict[n] = self._get_rules(rules, nonterm=False)

	def printAllRules(self):
		for s in self.rule_dict:
			for r in self.rule_dict[s]:
				print s + "->" + r + " " + str(self.rule_dict[s][r])

	def _get_rules(self, rules, nonterm=True):
		"""
		input: list of all rules
		output: dictionary of rules and random probs
		"""
		# append a terminal rule
		if not nonterm:
			#rules.append("a")
			rules.extend(["#","$","''",",","-LRB-","-RRB-",".",":","CC","CD","DT","EX","FW","IN","JJ","JJR","JJS","LS","MD","NN","NNP","NNPS","NNS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB","``"])
		n = len(rules)
		a = np.random.random(n)
		a /= a.sum()
		d = {}
		count = 0
		for i in rules:
			d[i] = a[count]
			count += 1
		return d



class EM():
	def __init__(self, gram):
		"""
		input: a GrammarGen object
		"""
		self.grammar = gram
		self.len_distrib = {}
		self.distrib_len = 200
		#self.emp_count = {}
		#self._get_emp_count("../real_data/sec02-21.words")
		self.loglikelihood = []

		self.len = 200
		self.alpha = {}
		self.beta = {}
		self.expect = {}
		self.maximize = {}

		#self.initLenDistrib(normalize=False)
		#self.getLogLikelihood()

	def get_nb(self):
		p = 0.200086480861
		n = 4.88137405883
		x = np.arange(0, self.distrib_len)
		self.nb_pmf = nbinom.pmf(x, n, p)

	def getNBLogLikelihood(self):
		log_result = 0
		for i in range(self.distrib_len):
			i += 1	# start from length 1
			if i in self.emp_count:
				log_result += self.emp_count[i] * np.log(self.nb_pmf[i])
		self.nb_logLikelihood = log_result


	def _get_emp_count(self, filename):
		with open(filename) as myfile:
			for l in myfile:
				c = len(l.split())
				if c in self.emp_count:
					self.emp_count[c] += 1
				else:
					self.emp_count[c] = 1


	def printLenDistrib(self):
		print self.len_distrib

	def initLenDistrib(self, normalize=True):
		# get terminal symbol probabilities
		self.len_distrib = {}
		for s in self.grammar.rule_dict:
			rules = self.grammar.rule_dict[s]
			self.len_distrib[s] = [0, 0]
			for r in rules:
				l = r.split()
				if len(l) == 1:
					if len(self.len_distrib[s]) == 2:
						self.len_distrib[s][1] += rules[r]
					else:
						self.len_distrib[s].append(rules[r])

		for l in range(self.distrib_len):
			l += 2
			for k in self.len_distrib:
				rule_dict = self.grammar.rule_dict[k]
				sum_prob = 0
				for rule in rule_dict:
					prob  = rule_dict[rule]
					rules = rule.split()
					if len(rules) == 2:
						tmp = 0
						for i in range(l-1):
							i += 1
							tmp += self.len_distrib[rules[0]][i] * self.len_distrib[rules[1]][l-i]
						sum_prob += prob * tmp
				self.len_distrib[k].append(sum_prob)
		if (normalize):
			print "normalize"
			for r in self.len_distrib:
				rules = self.len_distrib[r]
				sum_r = 0
				for p in rules:
					sum_r += p
				for i in range(len(rules)):
					self.len_distrib[r][i] = self.len_distrib[r][i]/sum_r

	def getLogLikelihood(self):
		log_result = 0
		for i in range(self.distrib_len):
			i += 1	# start from length 1
			if i in self.emp_count:
				log_result += self.emp_count[i] * np.log(self.len_distrib[self.grammar.start][i])
		self.loglikelihood.append(log_result)

	def getParam(self, sent):
		self.alpha = {}
		self.beta = {}
		self.getAlpha(sent)
		self.getBeta(sent)

		length = len(sent)
		self.Z = self.alpha[self.grammar.start][0][length-1]
		self.mu_A = {}
		for s in self.grammar.alphabet:
			self.mu_A[s] = {}
			for i in range(length):
				self.mu_A[s][i] = {}
				for j in range(length - i):
					j += i
					#self.mu_A[s][i][j] = self.alpha[s][i][j] * self.beta[s][i][j]
					self.mu_A[s][i][j] = self.alpha[s][i][j] + self.beta[s][i][j]

		self.mu_A_rules = {}
		self.count_rules = {}
		for s in self.grammar.alphabet:
			self.mu_A_rules[s] = {}
			self.count_rules[s] = {}
			rules = self.grammar.rule_dict[s]
			for r in rules:
				rule_l = r.split()
				prob = rules[r]
				if len(rule_l) == 2:
					#self.mu_A_rules[s][r] = 0
					self.mu_A_rules[s][r] = -np.inf
					for l in range(length):
						for i in range(length - l):
							j = i + l
							for k in range(l-1):
								k += i
								add = self.beta[s][i][j] + np.log(prob) + self.alpha[rule_l[0]][i][k] + self.alpha[rule_l[1]][k+1][j]
								self.mu_A_rules[s][r] = np.logaddexp(add, self.mu_A_rules[s][r])
					#self.count_rules[s][r] = self.mu_A_rules[s][r]/self.Z
					self.count_rules[s][r] = self.mu_A_rules[s][r] - self.Z
				else:
					#self.count_rules[s][r] = 0
					#self.mu_A_rules[s][r] = 0
					self.count_rules[s][r] = -np.inf
					self.mu_A_rules[s][r] = -np.inf
					for i in range(length):
						if sent[i] == r:
							#self.mu_A_rules[s][r] += self.mu_A[s][i][i]
							self.mu_A_rules[s][r] = np.logaddexp(self.mu_A_rules[s][r], self.mu_A[s][i][i]) 
					#self.count_rules[s][r] = self.mu_A_rules[s][r]/self.Z
					self.count_rules[s][r] = self.mu_A_rules[s][r] - self.Z

	def linear_order_rules(self):
		# eliminate rules to get linearly recursive grammar
		for s in self.grammar.rule_dict:
			rules = self.grammar.rule_dict[s]
			for r in rules:
				rule_l = r.split()
				if len(rule_l) == 2:
					k = self.grammar.alphabet.index(s)
					m = self.grammar.alphabet.index(rule_l[0])
					n =  self.grammar.alphabet.index(rule_l[1])
					if k < m or k < n or (k == m and k == n):
						self.grammar.rule_dict[s].delete(r)
		# normalize the elimintated grammar
		for s in self.grammar.rule_dict:
			rules = self.grammar.rule_dict[s]
			sum_s = 0
			for r in rules:
				sum_s += rules[r]
			for r in rules:
				rules[r] = rules[r]/sum_s

	def getAlpha(self, sent):
		length = len(sent)
		for s in self.grammar.rule_dict:
			rules = self.grammar.rule_dict[s]
			self.alpha[s] = {}
			single_rule_dict = {}
			for r in rules:
				l = r.split()
				prob = rules[r]
				if len(l) == 1:
					single_rule_dict[l[0]] = np.log(prob)


			for i in range(length):
				self.alpha[s][i] = {i: single_rule_dict[sent[i]]}
	
		for l in range(length-1):
			l += 1
			for i in range(length-l):			
				j = i + l
				for s in self.grammar.rule_dict:
					if j not in self.alpha[s][i]:
						#self.alpha[s][i][j] = 0
						self.alpha[s][i][j] = -np.inf
					rules = self.grammar.rule_dict[s]
					for r in rules:
						rule_l = r.split()
						prob = rules[r]
						if len(rule_l) == 2:	
							for k in range(j - i):
								k += i
								#self.alpha[s][i][j] += self.alpha[rule_l[0]][i][k] * self.alpha[rule_l[1]][k+1][j] * prob
								add = self.alpha[rule_l[0]][i][k] + self.alpha[rule_l[1]][k+1][j] + np.log(prob)
								self.alpha[s][i][j] = np.logaddexp(self.alpha[s][i][j], add)
		


	def getBeta(self, sent):
		# outside probability
		length = len(sent)
		for s in self.grammar.alphabet:
			self.beta[s] = {}
			for i in range(length):
				self.beta[s][i] = {}
				for j in range(length - i):
					j += i
					#self.beta[s][i][j] = 0
					self.beta[s][i][j] = -np.inf
		#self.beta[self.grammar.start][0][length-1] = 1
		self.beta[self.grammar.start][0][length-1] = np.log(1)
	
		for l in range(length):
			for i in range(length-l):
				j = length - (i + l) - 1
				for x in self.grammar.alphabet:
					r_rules = self.grammar.rule_dict[x]	

					for r in r_rules:
						prob = r_rules[r]
						rule_l = r.split()
						if len(rule_l) == 2:
							y = rule_l[0]
							z = rule_l[1]

							if i not in self.beta[y]:
								self.beta[y][i] = {}
							for k in range(j-i):
								k += (i + 1)
								#self.beta[y][i][k] += self.beta[x][i][j] * self.alpha[z][k][j] * prob
								add = self.beta[x][i][j] + self.alpha[z][k][j] + np.log(prob)
								self.beta[y][i][k] = np.logaddexp(add, self.beta[y][i][k])

								add = self.beta[x][i][j] + self.alpha[y][i][k] + np.log(prob)
								self.beta[z][k][j] = np.logaddexp(add, self.beta[z][k][j])

	def expectation(self):
		self.f_rules = {}
		for s in self.grammar.rule_dict:
			self.f_rules[s] = {}
			for r in self.grammar.rule_dict[s]:
				#self.f_rules[s][r] = 0
				self.f_rules[s][r] = -np.inf
		i = 0
		with open("../clean/sec00.tags") as tags_file:
			for line in tags_file:
				i += 1
				if i > 5:
					return
				sent = line.split()
				self.getParam(sent)
				for s in self.grammar.rule_dict:
					for r in self.grammar.rule_dict[s]:
						#self.f_rules[s][r] += self.count_rules[s][r]
						self.f_rules[s][r] =  np.logaddexp(self.f_rules[s][r], self.count_rules[s][r]) 

	def maximization(self):
		for x in self.grammar.rule_dict:
			r_rules = self.grammar.rule_dict[x]
			r_expects = self.f_rules[x]
			sum_x = 0
			for s in r_expects:
				#sum_x += r_expects[s]
				#sum_x = np.logaddexp(np.log(r_expects[s]), sum_x)
				sum_x += np.exp(r_expects[s])
			t = 0
			for r in r_rules:
				#self.grammar.rule_dict[x][r] = r_expects[r]/sum_x
				#self.grammar.rule_dict[x][r] = np.log(r_expects[r]) - sum_x
				self.grammar.rule_dict[x][r] = np.exp(r_expects[r]) / sum_x
				t += self.grammar.rule_dict[x][r]
			print x, t

		#self.grammar.printAllRules()
		#print self.expect

	def check(self):
		# calculate beta(a, n)
		self.beta_a = {}
		for x in self.grammar.rule_dict:
			r_rules = self.grammar.rule_dict[x]
			for r in r_rules:
				rules = r.split()
				prob = r_rules[r]
				if len(rules) == 1:
					a = rules[0]
					if a not in self.beta_a:
						self.beta_a[a] = [0 for i in range(self.distrib_len)]
					for i in range(self.distrib_len):
						if i == 0:
							continue
						self.beta_a[a][i] += self.beta[x][i-1] * prob
		sum_distrib = 0
		for i in self.len_distrib[self.grammar.start]:
			sum_distrib += i

		print "start prob sum: " + str(sum_distrib)
		print self.len_distrib[self.grammar.start][:10]
		print self.beta_a['a'][:10]
		print self.beta[self.grammar.start][:30]

	def iteration(self, max_iter, bound):
		self.expectation()
		self.maximization()
		print self.f_rules
		self.grammar.getAllRules()
		
				
		"""
		last_log = self.loglikelihood

		for i in range(max_iter):
			last_log = self.loglikelihood[i]
			self.getBeta()
			self.expectation()
			self.maximization()
			self.initLenDistrib(normalize=False)
			self.getLogLikelihood()
			#print self.loglikelihood
			#self.check()
			#self.grammar.printAllRules()
			#print self.expect
			if (math.fabs(self.loglikelihood[i+1] - last_log) < bound):
				break
		"""
if __name__ == "__main__":
	gram = GrammarGen(['A', 'B', 'C'])
	print gram.alphabet
	gram.getAllRules()
	#gram.printAllRules()

	
	em = EM(gram)
	#em.get_nb()
	#em.getNBLogLikelihood()

	#print em.nb_logLikelihood

	em.iteration(10000, 5)

	print em.loglikelihood
	em.grammar.printAllRules()
