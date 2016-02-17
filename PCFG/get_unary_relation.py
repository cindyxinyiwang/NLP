"""

Output the dependency relationships of the non terminals of the unary rules

"""
import sys

L = []
temp_l = []

def visit(n, grammar_dict):
	if n in temp_l:
		print n
		print temp_l
		print "not a DAG!"
		exit(1)
	if n not in L:
		temp_l.append(n)
		print n
		for m in grammar_dict[n]:
			visit(m, grammar_dict)
		print n	
		temp_l.remove(n)
		L.append(n)

if __name__ == "__main__":
	grammar_file = "emilinate_1.grammar"
	grammar_dict = {}

	with open(grammar_file) as myfile:
		for line in myfile:
			grams = line.split(" ")
			if len(grams) == 4:
				# if it is unary rule
				term_l = grams[0]
				term_r = grams[2]
				if not term_r == term_l:
					if term_l not in grammar_dict:
						grammar_dict[term_l] = [term_r]
					else:
						grammar_dict[term_l].append(term_r)
	unmarked = []
	unmarked.extend(grammar_dict.keys())

	# topological sort
	while len(unmarked) > 0 :
		visit(unmarked[0], grammar_dict)

	print L
