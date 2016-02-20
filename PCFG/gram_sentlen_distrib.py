"""

This program calculates the length distribution of strings derived from a certain grammar
In this case we are looking at PCFG from berkerly parser
Input: file.grammar, file.lexicon
Output: length distribution

"""
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

lexicon_file = "out.txt.lexicon"
grammar_file = "eliminated.grammar"
distrib = {} # distribution as a dict, with non terminal as key, array of probabiliteis as value
grammar_dict = {}
unary_grammar_dict = {}
nonterminals_dict = {}

# first get the length 1 probabilities from lexicon file
with open(lexicon_file) as myfile:
	for line in myfile:
		gram = line.replace('\n', '').split(" ")
		non_term = gram[0]
		prob_list = gram[2:] #each prob still has [ ] and , that needs to be processed
		i = 0
		for prob in prob_list:
			#remove trailing char
			prob = prob[:-1]
			if prob[0] == '[':
				prob = prob[1:]
			cur_non_term = non_term + "_" + str(i)
			if cur_non_term in distrib:
				distrib[cur_non_term][0] += float(prob)
			else:
				distrib[cur_non_term] = [float(prob)]
			i += 1
			# update nonterminals_dict
			if cur_non_term not in nonterminals_dict.itervalues():
				nonterminals_dict[len(nonterminals_dict)] = cur_non_term

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
			# T0 -> T1 T2 not already in then T0 sent has more than 2 symbols
			if term_l not in distrib:
				distrib[term_l] = [0]
			if term_r1 not in distrib:
				distrib[term_r1] = [0]
			if term_r2 not in distrib:
				distrib[term_r2] = [0]
			# update nonterminals_dict
			if term_l not in nonterminals_dict.itervalues():
				nonterminals_dict[len(nonterminals_dict)] = term_l
			if term_r1 not in nonterminals_dict.itervalues():
				nonterminals_dict[len(nonterminals_dict)] = term_r1
			if term_r2 not in nonterminals_dict.itervalues():
				nonterminals_dict[len(nonterminals_dict)] = term_r2
		else:
			term_l = gram[0]
			term_r1 = gram[2]
			prob = float(gram[3])
			# construct unary rule dict
			if term_l not in unary_grammar_dict:
				unary_grammar_dict[term_l] = { term_r1: prob}
			else:
				unary_grammar_dict[term_l][term_r1] = prob

			if term_l not in grammar_dict:
				grammar_dict[term_l] = { term_r1: prob }
			else:
				grammar_dict[term_l][term_r1] = prob

			if term_r1 not in distrib:
				distrib[term_r1] = [0]
			if term_l not in distrib:
				#distrib[term_l] = [ 0 + prob*distrib[term_r1][0] ]
				distrib[term_l] = [0]
			# update nonterminals_dict
			if term_l not in nonterminals_dict.itervalues():
				nonterminals_dict[len(nonterminals_dict)] = term_l
			if term_r1 not in nonterminals_dict.itervalues():
				nonterminals_dict[len(nonterminals_dict)] = term_r1

# update distrib by unary rules
#unary_distrib = np.matrix([distrib[t][0] for t in grammar_dict])
temp_mat = []
for i in range(len(nonterminals_dict)):
	t = nonterminals_dict[i]
	temp_mat.append(distrib[t][0])
unary_distrib = np.matrix(temp_mat)

# get unary matrix u for calculation
mat_array = []
for i in range(len(nonterminals_dict)):
	t = nonterminals_dict[i]
	temp_mat_array = []
	for j in range(len(nonterminals_dict)):
		t2 = nonterminals_dict[j]
		if t in unary_grammar_dict and t2 in unary_grammar_dict[t]:
			temp_mat_array.append(unary_grammar_dict[t][t2])
		else:
			temp_mat_array.append(0)
	mat_array.append(temp_mat_array)
unary_matrix = np.matrix(mat_array)

unary_multiplier = np.linalg.inv(np.identity(len(unary_matrix)) - unary_matrix)
unary_distrib_new = np.dot( unary_multiplier, unary_distrib.transpose())
#unary_distrib_new = np.dot( unary_matrix, unary_distrib.transpose())
#np.savetxt("unary_distrib.txt", unary_distrib_new)
#np.savetxt("unary_distrib.txt", unary_distrib.transpose())
# add new distribution back
for i in range(len(nonterminals_dict)):
	t = nonterminals_dict[i]
	distrib[t][0] = unary_distrib_new.item(i, 0)
"""
for k in distrib:
	print k, " ", distrib[k]

for g in grammar_dict:
	print g, " ", grammar_dict[g]
"""

total_len = 100
for l in range(total_len):
	l += 1	# l is the current length of string generated by each non terminal
	for k in distrib:
		if k in grammar_dict:
			rule_dict = grammar_dict[k]
			sum_prob = 0
			for rule in rule_dict:
				prob = rule_dict[rule]
				rules = rule.split(" ")
				if (len(rules) == 2):
					tmp = 0
					for i in range(l):
						#print distrib[rules[0]][i], distrib[rules[1]][l-1-i]
						tmp += distrib[rules[0]][i] * distrib[rules[1]][l-1-i]
					sum_prob += prob * tmp
			distrib[k].append(sum_prob)
		else:
			# there is no k -> T1 T2 rule exist
			distrib[k].append(0)

	# infinite summation of unary rules
	temp_mat = []
	for i in range(len(nonterminals_dict)):
		t = nonterminals_dict[i]
		temp_mat.append(distrib[t][l])

	unary_distrib = np.matrix(temp_mat)
	unary_distrib_new = np.dot( unary_multiplier, unary_distrib.transpose())	
	#unary_distrib_new = np.dot( unary_matrix, unary_distrib.transpose())
	# add new distribution back
	for i in range(len(nonterminals_dict)):
		t = nonterminals_dict[i]
		distrib[t][l] = unary_distrib_new.item(i, 0)

print distrib['ROOT_0']
#for k in distrib:
#	print k, ": ", distrib[k][0], distrib[k][1]

plt.plot([ i+1 for i in range(len(distrib['ROOT_0']))], distrib['ROOT_0'])
plt.show()
