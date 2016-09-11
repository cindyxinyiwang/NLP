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
		print self.start
		for s in self.rule_dict:
			for r in self.rule_dict[s]:
				print s + " -> " + r + " " + "[" + str(self.rule_dict[s][r]) + "]"

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

		self.loglikelihood = []

		self.alpha = {}
		self.beta = {}
		self.expect = {}
		self.maximize = {}

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
			for i in xrange(length):
				self.mu_A[s][i] = {}
				for j in xrange(i, length):
					self.mu_A[s][i][j] = self.alpha[s][i][j] * self.beta[s][i][j]

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
					self.mu_A_rules[s][r] = 0.0
					for l in xrange(length):
						for i in xrange(length - l):
							j = i + l
							for k in xrange(i, j):
								self.mu_A_rules[s][r] += self.beta[s][i][j] * prob * self.alpha[rule_l[0]][i][k] * self.alpha[rule_l[1]][k+1][j]
					self.count_rules[s][r] = self.mu_A_rules[s][r]/self.Z
				else:
					self.count_rules[s][r] = 0.0
					self.mu_A_rules[s][r] = 0.0
					for i in xrange(length):
						if sent[i] == r:
							self.mu_A_rules[s][r] += self.mu_A[s][i][i]
					self.count_rules[s][r] = self.mu_A_rules[s][r]/self.Z

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
					single_rule_dict[l[0]] = prob


			for i in range(length):
				if sent[i] in single_rule_dict:
					self.alpha[s][i] = {i: single_rule_dict[sent[i]]}
				else:
					self.alpha[s][i] = {i: 0.0}
	
		for l in range(length-1):
			l += 1
			for i in range(length-l):			
				j = i + l
				for s in self.grammar.rule_dict:
					if j not in self.alpha[s][i]:
						self.alpha[s][i][j] = 0.0
					rules = self.grammar.rule_dict[s]
					for r in rules:
						rule_l = r.split()
						prob = rules[r]
						if len(rule_l) == 2:	
							for k in range(l):
								k += i
								self.alpha[s][i][j] += self.alpha[rule_l[0]][i][k] * self.alpha[rule_l[1]][k+1][j] * prob
		
	def getBeta(self, sent):
		# outside probability
		length = len(sent)
		for s in self.grammar.alphabet:
			self.beta[s] = {}
			for i in range(length):
				self.beta[s][i] = {}
				for j in range(length - i):
					j += i
					self.beta[s][i][j] = 0.0
		self.beta[self.grammar.start][0][length-1] = 1.0

		for l in xrange(length-1, -1, -1):
			for i in xrange(length-l):
				j = i + l
				if i == 0 and j == (length-1):
					continue
				for x in self.grammar.alphabet:
					r_rules = self.grammar.rule_dict[x]	
					for r in r_rules:
						prob = r_rules[r]
						rule_l = r.split()
						if len(rule_l) == 2:
							y = rule_l[0]
							z = rule_l[1]
							for k in xrange(j + 1, length):
								self.beta[y][i][j] += self.beta[x][i][k] * self.alpha[z][j+1][k] * prob
							for k in xrange(i):
								self.beta[z][i][j] += self.beta[x][k][j] * self.alpha[y][k][i-1] * prob

	def expectation(self, train_data):
		self.f_rules = {}
		for s in self.grammar.rule_dict:
			self.f_rules[s] = {}
			for r in self.grammar.rule_dict[s]:
				self.f_rules[s][r] = 0.0
		self.loglikelihood = 0
		for sent in train_data:
			self.getParam(sent)
			self.loglikelihood += np.log(self.Z)
			for s in self.grammar.rule_dict:
				for r in self.grammar.rule_dict[s]:
					self.f_rules[s][r] += self.count_rules[s][r]

	def maximization(self):
		#print self.f_rules
		for x in self.grammar.rule_dict:
			r_rules = self.grammar.rule_dict[x]
			r_expects = self.f_rules[x]
			sum_x = 0
			for s in r_expects:
				sum_x += r_expects[s]

			for r in r_rules:
				self.grammar.rule_dict[x][r] = r_expects[r]/sum_x

	def iteration(self, max_iter, bound, train_file):
		train_data = [line.strip().split() for line in open(train_file)]
		last_log = 0
		for i in range(6):
			self.expectation(train_data)
			self.maximization()
			last_log = self.loglikelihood
			print self.loglikelihood
		
if __name__ == "__main__":
	gram = GrammarGen(['A', 'B', 'C'])

	
	print gram.alphabet
	gram.getAllRules()
	#gram.printAllRules()

	
	em = EM(gram)

	em.iteration(5, 5, "clean/sec00.tags")
	
	#gram.getAllRules()
	#gram.printAllRules()

