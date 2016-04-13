import numpy as np

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
			self.rule_dict[n] = self._get_rules(rules)

	def printAllRules(self):
		for s in self.rule_dict:
			for r in self.rule_dict[s]:
				print s + "->" + r + " " + str(self.rule_dict[s][r])

	def _get_rules(self, rules):
		"""
		input: list of all rules
		output: dictionary of rules and random probs
		"""
		# append a terminal rule
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
		self.distrib_len = 10
		self.emp_count = {}
		self._get_emp_count("../real_data/sec02-21.words")
		self.loglikelihood = []

		self.beta = {}
		self.expect = {}
		self.maximize = {}

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

	def initLenDistrib(self):
		# get terminal symbol probabilities
		for s in self.grammar.rule_dict:
			rules = self.grammar.rule_dict[s]
			for r in rules:
				l = r.split()
				if len(l) == 1:
					if s in self.len_distrib:
						self.len_distrib[s][0] += rules[r]
					else:
						self.len_distrib[s] = [0, rules[r]]
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
							tmp += self.len_distrib[rules[0]][i+1] * self.len_distrib[rules[1]][l-i-1]
						sum_prob += prob * tmp
				self.len_distrib[k].append(sum_prob)

	def getLogLikelihood(self):
		log_result = 0
		for i in range(self.distrib_len):
			# len_distrib item i is the distrib of string length i+1
			if i in self.emp_count:
				log_result += self.emp_count[i] * np.log(self.len_distrib[self.grammar.start][i])
		self.loglikelihood.append(log_result)

	def getBeta(self):
		for s in self.grammar.alphabet:
			self.beta[s] = [1]
		for i in range(self.distrib_len):
			i += 1
			for x in self.grammar.alphabet:
				r_rules = self.grammar.rule_dict[x]
				for r in r_rules:
					rules = r.split()
					if len(rules) == 2:
						z = rules[0]
						y = rules[1]
						if len(self.beta[z]) == (i+1):
							for k in range(i):
								self.beta[z][i] += self.beta[z][k] * self.len_distrib[y][i-k]
						else:
							for k in range(i):
								self.beta[z].append(self.beta[z][k] * self.len_distrib[y][i-k])

						y = rules[0]
						z = rules[1]
						if len(self.beta[z]) == (i+1):
							for k in range(i):
								self.beta[z][i] += self.beta[z][k] * self.len_distrib[y][i-k]
						else:
							for k in range(i):
								self.beta[z].append(self.beta[z][k] * self.len_distrib[y][i-k])

	def expectation(self):
		for x in self.grammar.rule_dict:
			r_rules = self.grammar.rule_dict[x]
			for r in r_rules:
				rules = r.split()
				prob = r_rules[r]
				e = 0
				if len(rules == 2):
					for n in range(self.distrib_len):
						tmp = 0
						for k in range(n):
							for l in range(k):
								for m in range(l):
									tmp += self.beta[x][k]*prob*self.len_distrib[rules[0]][l]*self.len_distrib[rules[1]][m]
						e += tmp * self.emp_count[n] / self.len_distrib[self.start][n]
				else:
					pass
				if x in self.expect:
					self.expect[x][r_rules] = e
				else:
					self.expect[x] = {r_rules: e}


if __name__ == "__main__":
	gram = GrammarGen(['A', 'B', 'C' ,'D', 'E'])
	gram.getAllRules()
	gram.printAllRules()

	em = EM(gram)
	em.initLenDistrib()
	em.printLenDistrib()
	em.getLogLikelihood()
	print em.loglikelihood
	em.getBeta()
