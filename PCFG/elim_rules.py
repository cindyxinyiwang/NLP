"""
this script eliminate grammars unreachable from ROOT
"""
from collections import deque

graph = {}
root = "ROOT_0"
all_nonterms = []
visited = []

grammar_in = "out.txt.grammar"
grammar_out = "eliminated.grammar"

if __name__ == "__main__":
	"""
	grammar_file = "out.txt.grammar"
	with open(grammar_file) as myfile:
		for line in myfile:
			gram = line.replace('\n', '').split(" ")
			if len(gram) == 5:
				term_l = gram[0]
				term_r1 = gram[2]
				term_r2 = gram[3]
				prob = float(gram[4])
				#update grammar dict
				if term_l not in graph:
					graph[term_l] = [term_r1, term_r2]
				else:
					graph[term_l].append(term_r1)
					graph[term_l].append(term_r2)
			else:
				term_l = gram[0]
				term_r1 = gram[2]
				prob = float(gram[3])
				# construct unary rule dict
				if term_l not in graph:
					graph[term_l] = [term_r1]
				else:
					graph[term_l].append(term_r1)

	q = deque([])
	q.append(root)
	print graph[root]
	while not len(q) == 0:
		c = q.popleft()
		visited.append(c)
		for i in graph[c]:
			if i not in visited:
				q.append(i)	

	print visited
	"""
	out = open(grammar_out, 'w')
	with open(grammar_in) as myfile:
		for line in myfile:
			gram = line.replace('\n', '').split(" ")
			if len(gram) == 5:
				out.write(line)
			else:
				term_l = gram[0]
				term_r1 = gram[2]
				prob = float(gram[3])
				if term_r1 == term_l and prob == 1:
					None 
				else:
					out.write(line)

	out.close()

