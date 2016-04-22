import numpy as np
import matplotlib.pyplot as plt
import math

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
			rules.append("a")
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
		self.emp_count = {}
		self._get_emp_count("../real_data/sec02-21.words")
		self.loglikelihood = []

		self.beta = {}
		self.expect = {}
		self.maximize = {}

		self.initLenDistrib(normalize=False)
		self.getLogLikelihood()

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

	def getBeta(self):
		# outside probability
		self.beta = {}
		for s in self.grammar.alphabet:
				self.beta[s] = [0]
		self.beta[self.grammar.start][0] = 1

		for i in range(self.distrib_len):
			i += 1
			for x in self.grammar.alphabet:
				r_rules = self.grammar.rule_dict[x]
				for r in r_rules:
					prob = self.grammar.rule_dict[x][r]
					rules = r.split()
					if len(rules) == 2:
						z = rules[0]
						y = rules[1]
						if len(self.beta[z]) >= (i+1):
							tmp = 0
							for k in range(i):
								tmp += self.beta[x][k] * self.len_distrib[y][i-k] * prob
							self.beta[z][i] += tmp 
						else:
							tmp = 0
							for k in range(i):
								tmp += self.beta[x][k] * self.len_distrib[y][i-k] * prob
							self.beta[z].append(tmp )

						y = rules[0]
						z = rules[1]
						if len(self.beta[z]) >= (i+1):
							tmp = 0
							for k in range(i):
								tmp += self.beta[x][k] * self.len_distrib[y][i-k] * prob
							self.beta[z][i] += tmp 
						else:
							tmp = 0
							for k in range(i):
								tmp += self.beta[x][k] * self.len_distrib[y][i-k] * prob
							self.beta[z].append(tmp)

	def expectation(self):
		self.expect = {}
		for x in self.grammar.rule_dict:
			r_rules = self.grammar.rule_dict[x]
			for r in r_rules:
				rules = r.split()
				prob = r_rules[r]
				e = 0
				if len(rules) == 2:
					for n in range(self.distrib_len):
						n += 1
						if n not in self.emp_count:
							continue
						tmp = 0
						for k in range(n+1):
							for l in range(n-k):
								m = n - k - l
								tmp += self.beta[x][k]*prob*self.len_distrib[rules[0]][l]*self.len_distrib[rules[1]][m]
						e += tmp * self.emp_count[n] / self.len_distrib[self.grammar.start][n]
				else:
					for n in range(self.distrib_len):
						n += 1
						if n not in self.emp_count:
							continue
						tmp = self.emp_count[n] * self.beta[x][n-1]*prob / self.len_distrib[self.grammar.start][n]
						e += tmp

				if x in self.expect:
					self.expect[x][r] = e
				else:
					self.expect[x] = {}
					self.expect[x][r] = e

	def maximization(self):
		for x in self.grammar.rule_dict:
			r_rules = self.grammar.rule_dict[x]
			r_expects = self.expect[x]
			sum_x = 0
			for s in r_expects:
				sum_x += r_expects[s]
			for r in r_rules:
				self.grammar.rule_dict[x][r] = r_expects[r]/sum_x

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
		last_log = self.loglikelihood

		for i in range(max_iter):
			last_log = self.loglikelihood[i]
			self.getBeta()
			self.expectation()
			self.maximization()
			self.initLenDistrib(normalize=False)
			self.getLogLikelihood()
			print self.loglikelihood
			#self.check()
			#self.grammar.printAllRules()
			#print self.expect
			if (math.fabs(self.loglikelihood[i+1] - last_log) < bound):
				break

if __name__ == "__main__":
	gram = GrammarGen(['A', 'B' ])
	gram.getAllRules()
	#gram.printAllRules()

	
	em = EM(gram)
	em.iteration(50, 0)
