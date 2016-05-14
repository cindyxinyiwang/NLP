import numpy as np
import sys

class Node():
	def __init__(self, my_char, my_prob, left=None, right=None, terminal= None):
		self.my_char = my_char
		self.left = left
		self.right = right

		self.terminal = terminal

		self.my_prob = my_prob
		l_prob = 1
		r_prob = 1
		if left:
			l_prob = left.my_prob
		if right:
			r_prob = right.my_prob

		self.tree_prob = self.my_prob * l_prob * r_prob

	def addBackPointers(self, left, right):
		self.left = left
		self.right = right

	def printTree(self):
		sys.stdout.write("(")
		if self.left:
			self.left.printTree()
		if self.terminal:
			sys.stdout.write(self.terminal)
		if self.right:
			self.right.printTree()
		sys.stdout.write(")")

	def strTree(self):
		left = ""
		right = ""
		center = ""
		if self.left:
			left = self.left.strTree()
		if self.terminal:
			center = self.terminal
		if self.right:
			right = self.right.strTree()
		return "(" + left + center + right + ")"


class Cell():
	def __init__(self):
		self.node_list = []
		self.final_node = None

	def addNode(self, parent, prob, left, right, terminal=None):
		n = Node(parent, prob, left, right, terminal)
		self.node_list.append(n)

	def get_final_node(self, root = False):
		if len(self.node_list) > 0:
			if root:
				c_node = None
				c_prob = -np.inf
				for n in self.node_list:
					if n.my_char == 'A' and n.tree_prob > c_prob:
						c_node = n
						c_prob = n.tree_prob
				self.final_node = c_node
			else:
				c_node = None
				c_prob = -np.inf
				for n in self.node_list:
					if n.tree_prob > c_prob:
						c_node = n
						c_prob = n.tree_prob
				self.final_node = c_node

	def __str__(self):
		if self.final_node:
			return self.final_node.my_char + str(self.final_node.tree_prob)
		else:
			return ""

class Parser():
	def __init__(self, gram_file):
		self.gram_dict = self.parse_gram_file(gram_file)
	
	def parse_gram_file(self, gram_file):
		result_dict = {}
		
		with open(gram_file) as myfile:
			for line in myfile:
				data = line.split()
				if len(data) < 2:
					continue
				if len(data) == 5:
					lhs = data[0]
					r1 = data[2]
					r2 = data[3]
					prob = data[4]

					prob = float(prob[1:-1])

					if lhs in result_dict:
						result_dict[lhs][r1 + " " + r2] = prob
					else:
						result_dict[lhs] = {r1 + " " + r2: prob}
				else:
					lhs = data[0]
					r1 = data[2]
					prob = data[3]

					prob = float(prob[1:-1])

					if lhs in result_dict:
						result_dict[lhs][r1] = prob
					else:
						result_dict[lhs] = {r1: prob}

			return result_dict

	def print_gram(self):
		for s in self.gram_dict:
			rules = self.gram_dict[s]
			for r in rules:
				print s, "->", r, rules[r]

	"""
	def get_prob(self, y, z):
		rule = y + " " + z
		for x in self.gram_dict:
			rules = self.gram_dict[x]
			if rule in rules:
				return rules[rule], x 
		return None, None
	"""
	def print_table(self):
		for i in self.parse_table:
			for j in i:
				print j

	def parseFile(self, file_name, out_name):
		out_file = open(out_name, 'w')
		with open(file_name) as input_file:
			for line in input_file:
				s = self.ckyParse(line)
				out_file.write(s)
				out_file.write('\n')
		out_file.close()

	def ckyParse(self, sent):
		sent = sent.split()
		self.parse_table = [[Cell() for i in range(len(sent)+1)] for i in range(len(sent)+1)]
		self.initChart(sent)
		
		self.fillChart(sent)
		self.print_table()
		return self.parse_table[0][len(sent) - 1].final_node.strTree()



	def initChart(self, sent):
		length = len(sent)

		for i in range(length):
			c = sent[i]
			for x in self.gram_dict:
				rules = self.gram_dict[x]

				if c in rules:
					prob = rules[c]
					self.parse_table[i][i].addNode(x, prob, None, None, terminal = c)
			self.parse_table[i][i].get_final_node()


	def fillChart(self, sent):
		length = len(sent)
		for l in range(length):
			l += 1
			for i in range(length - l):
				if i == 0 and l == (length -1):
					self.fillCell(i, i + l, True)
				else:
					self.fillCell(i, i + l)


	def fillCell(self, i, j, root=False):
		for k in range(j - i):
			k += i
			self.combineCells(i, k, j)
		if not root:
			self.parse_table[i][j].get_final_node()
		else:
			self.parse_table[i][j].get_final_node(root = True)

	def combineCells(self, i, k, j):
		left_node = self.parse_table[i][k].final_node
		right_node = self.parse_table[k+1][j].final_node
		if left_node and right_node:
			y = left_node.my_char
			z = right_node.my_char
			rule = y + " " + z
			for x in self.gram_dict:
				rules = self.gram_dict[x]
				if rule in rules:
					prob = rules[rule]
					self.parse_table[i][j].addNode(x, prob, left_node, right_node)

				
if __name__ == "__main__":
	p = Parser("3_nonrecurse_final.grammar")
	p.print_gram()
	p.ckyParse("RB , PRP VBD RB NNP NNP .")
	p.parseFile("../EM/clean/sec23.tags", "parse.txt")
