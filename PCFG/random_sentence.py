import random

lexicon_file = "out.txt.lexicon"
grammar_file = "eliminated.grammar"
grammar_dict = {}
terminal_dict = {}

def generate(root):
	if root in grammar_dict:
		choice = random.randint(0, len(grammar_dict[root].keys())-1)
		rule = grammar_dict[root].keys()[choice]
		rules = rule.split(" ")
		if len(rules) >= 2:
			left = generate(rules[0])
			right = generate(rules[1])
			return " ".join([left, right])
		elif len(rules) == 1:
			return generate(rules[0])
	if root in terminal_dict:
			choice = random.randint(0, len(terminal_dict[root].keys())-1)
			return terminal_dict[root].keys()[choice]
	return ""


if __name__=="__main__":
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
	
	
	# generate sentence	
	for i in range(30):
		print generate("ROOT_0")

