"""

This program checks if the PCFG grammar is linearly recursive

"""
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys

lexicon_file = "out.txt.lexicon"
grammar_file = "eliminated.grammar"
cut_off_prob = float("0.08")

def DFS(n, visited, top_order, graph):
	if n not in graph:
		visited.append(n)
		top_order.append(n)
		return
	children = graph[n]
	visited.append(n)
	for c in children:
		if c not in visited:
			DFS(c, visited, top_order, graph)
	top_order.append(n)
	


if __name__ == "__main__":
	gram_dict = {}
	alphabet = []
	graph = {}

	with open(grammar_file) as myfile:
		for line in myfile:
			gram = line.split()
			if len(gram) == 5:
				term_l = gram[0]
				term_r1 = gram[2]
				term_r2 = gram[3]
				prob = float(gram[4])
				if prob < cut_off_prob:
					continue
				if term_l not in alphabet:
					alphabet.append(term_l)
				if term_r1 not in alphabet:
					alphabet.append(term_r1)
				if term_r2 not in alphabet:
					alphabet.append(term_r2)

				if term_l in graph:
					if term_r1 not in graph[term_l]:
						graph[term_l].append(term_r1)
					if term_r2 not in graph[term_l]:
						graph[term_l].append(term_r2)
				else:
					graph[term_l] = [term_r1]
					if term_r2 not in graph[term_l]:
						graph[term_l].append(term_r2)


				if term_l not in gram_dict:
					gram_dict[term_l] = []
				gram_dict[term_l].append(term_r1 + " " + term_r2)

			else:
				if prob < cut_off_prob:
					continue
				term_l = gram[0]
				term_r1 = gram[2]
				prob = float(gram[3])
				if term_l not in alphabet:
					alphabet.append(term_l)
				if term_r1 not in alphabet:
					alphabet.append(term_r1)

				if term_l in graph:
					if term_r1 not in graph[term_l]:
						graph[term_l].append(term_r1)
				else:
					graph[term_l] = [term_r1]

				if term_l not in gram_dict:
					gram_dict[term_l] = []
				gram_dict[term_l].append(term_r1)
	
	# topological sort graph, ignore cycle
	top_order = []
	visited = []
	for n in alphabet:
		if n not in visited:
			DFS(n, visited, top_order, graph)
	
	top_order.reverse()
	order_dict = {}
	i = 0
	for n in top_order:
		order_dict[n] = i
		i += 1
	print top_order

	for n in top_order:
		if n not in gram_dict:
			continue
		rules = gram_dict[n]
		x_order = order_dict[n]
		for r in rules:
			rule_list = r.split()
			if len(rule_list) == 2:
				y_order = order_dict[rule_list[0]]
				z_order = order_dict[rule_list[1]]
				if x_order >= y_order and x_order >= z_order:
					print "not linearly recursive!"
					print n, "->", r

					for s in gram_dict:
						rules = gram_dict[s]
						print s, "->", rules
					sys.exit(0)
			else:
				y_order = order_dict[rule_list[0]]

	print "grammar is linearly ordered!"



	