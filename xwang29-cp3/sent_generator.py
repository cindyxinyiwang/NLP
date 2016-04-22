#from numpy.random import choice
from random import random
from bisect import bisect
import random as rd
from nltk.tokenize import word_tokenize
import sys

class markov:
	def __init__(self, train_file="", level=2):
		self.model = {}
		self.n = level
		self.text = self.readtraining(train_file)
		self.tokens = word_tokenize(self.text)
		self.build_model_word()
		

	def readtraining(self, filepath):
		with open(filepath, "r") as file:
			return file.read()

	def build_model_word(self):
		n = self.n
		model = dict()
		#tokens = self.text.strip().split(" ")
		if len(self.tokens) < n:
			return model
		for i in range(len(self.tokens) - n):
			gram = tuple(self.tokens[i:i+n])
			next_token = self.tokens[i+n]
			if gram in model:
				model[gram].append(next_token)
			else:
				model[gram] = [next_token]
		final_gram = tuple(self.tokens[len(self.tokens)-n:])
		if final_gram in model:
			model[final_gram].append(None)
		else:
			model[final_gram] = [None]
		self.model = model

	def generate(self,seed=None, max_iterations=20):
		n = self.n
		if seed is None:
			seed = rd.choice(self.model.keys())
		output = list(seed)
		current = tuple(seed)
		for i in range(max_iterations):
			if current in self.model:
				possible_next_tokens = self.model[current]
				next_token = rd.choice(possible_next_tokens)
				if next_token is None: break
				output.append(next_token)
				current = tuple(output[-n:])
			else:
				break
		return ' '.join(output)

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
	mar = markov("train.txt")
	while True:
		choice = raw_input("Please select: q to quit, p for generate with PCFG grammar, m for generate with Markov chain")
		if choice == 'q':
			sys.exit(1)
		if choice == 'p':
			print gen.generate("ROOT_0")
			continue
		if choice == 'm':
			print mar.generate()
		else:
			print "Not recognized!"

