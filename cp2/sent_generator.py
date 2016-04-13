#from numpy.random import choice
from random import random
from bisect import bisect

class sent_generator:
	def __init__(self, gram_file="eliminated.grammar", lex_file="out.txt.lexicon"):
		self.grammar_dict = self.parse_gram_file(gram_file)
		self.terminal_dict = self.parse_lex_file(lex_file)

	def getWeight(self, weights):
		return [float(i)/sum(weights) for i in weights]

	def parse_gram_file(self, grammar_file):
		grammar_dict = {}
		with open(grammar_file) as myfile:
			for line in myfile:
				gram = line.replace('\n', '').split(" ")
				if len(gram) == 5:
					term_l = gram[0]
					term_r1 = gram[2]
					term_r2 = gram[3]
					prob = float(gram[4])
					#update grammar dict
					if term_l not in grammar_dict:
						grammar_dict[term_l] = { " ".join([term_r1, term_r2]): prob }
					else:
						grammar_dict[term_l][" ".join([term_r1, term_r2])] = prob
				else:
					term_l = gram[0]
					term_r1 = gram[2]
					prob = float(gram[3])
					if term_l not in grammar_dict:
						grammar_dict[term_l] = { term_r1: prob }
					else:
						grammar_dict[term_l][term_r1] = prob
		return grammar_dict

	def parse_lex_file(self, lexicon_file):
		terminal_dict = {}
		with open(lexicon_file) as myfile:
			for line in myfile:
				gram = line.replace('\n', '').split(" ")
				non_term = gram[0]
				word = gram[1]
				prob_list = gram[2:] #each prob still has [ ] and , that needs to be processed
				i = 0
				for prob in prob_list:
					#remove trailing char
					prob = prob[:-1]
					if prob[0] == '[':
						prob = prob[1:]
					cur_non_term = non_term + "_" + str(i)
					if cur_non_term in terminal_dict:
						terminal_dict[cur_non_term][word] = float(prob)
					else:
						terminal_dict[cur_non_term] = {word: float(prob)}
					i += 1
		return terminal_dict

	def choice(self, values, p):
		total = 0
		cum_weights = []
		for w in p:
			total += w
			cum_weights.append(total)
		x = random() * total
		i = bisect(cum_weights, x)
		return values[i]

	def generate(self, root):
		if root in self.grammar_dict:
			rule = self.choice(self.grammar_dict[root].keys(), p = self.getWeight(self.grammar_dict[root].values()))
			rules = rule.split(" ")
			if len(rules) >= 2:
				#print rules
				left = self.generate(rules[0])
				right = self.generate(rules[1])
				return " ".join([left, right])
			elif len(rules) == 1:
				return self.generate(rules[0])
		if root in self.terminal_dict:
				#print root
				return self.choice(self.terminal_dict[root].keys(), p = self.getWeight(self.terminal_dict[root].values()))
		return ""

if __name__=="__main__":
	gen = sent_generator()
	num = 2
	for i in num:
		print gen.generate("ROOT_0")
