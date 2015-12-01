from os import listdir
from os.path import isfile, join

class logis:
	def __init__(self):
		self.class_counts = {}
		self.class_counts_prob = {}
		self.class_word_counts = {}
		self.class_word_counts_prob = {}
		self.all_words = set([])
		self.logis_params = {}

	def get_counts(self, train_file, c):
		with open(train_file, "r") as myfile:
			lines = myfile.readlines()
			self.class_counts[c] = len(lines)
			for line in lines:
				for w in line.split():
					if w not in self.all_words:
						self.all_words.add(w)
					tup = (c, w)
					if tup not in self.class_word_counts:
						self.class_word_counts[tup] = 1
					else:
						self.class_word_counts[tup] = self.class_word_counts[tup] + 1
					
	def read_train(self, train_file_dir):
		# read positives
		pos_file = train_file_dir + 'train_pos'
		self.get_counts(pos_file, "+")
		neg_file = train_file_dir + 'train_neg'
		self.get_counts(neg_file, "-")

	def get_prob(self):
		total = 0
		word_count = len(self.all_words)
		for c in self.class_counts:
			total = total + self.class_counts[c]
		for c in self.class_counts:
			cur_class_count = self.class_counts[c]
			self.class_counts_prob[c]= (0.0 + cur_class_count)/total
			for w in self.all_words:
				tup = (c, w)
				if tup in self.class_word_counts:
					self.class_word_counts_prob[tup] = (self.class_word_counts[tup]+1.0)/(cur_class_count+1+word_count)
				else:
					self.class_word_counts_prob[tup] = (1.0)/(cur_class_count+1+word_count)

	def train(self):
		""" train the logistic regression classifier"""
		classes = ['+', '-']
		learn_rate = 0.5
		self.logis_params = { (k, w):0 for k in classes for w in self.all_words }	
		for c in classes:
			for w in self.all_words:
				try: 
					self.logis_params[(c, w)] = self.logis_params[(c, w)] + learn_rate * self.class_word_counts[(c, w)]
					for k in classes:
						self.logis_params[(c, w)] = self.logis_params[(c, w)] - learn_rate * self.class_counts_prob[k] * self.class_word_counts[(c, w)]
				except:
					self.logis_params[(c, w)] = self.logis_params[(c, w)] + learn_rate * 0
					for k in classes:
						self.logis_params[(c, w)] = self.logis_params[(c, w)] - learn_rate * self.class_counts_prob[k] * 0

	def logis_classify(self, doc):
		""" classify a file based on logistic regression param """
		classes = ['+', '-']
		probs = {}
		for c in classes:
			p = 1
			for w in doc:
				if (c,w) in self.logis_params:
					p = p+(self.logis_params[(c, w)])
			probs[c] = p
		if (probs['+'] >= probs['-']):
			return '+'
		else:
			return '-'		

	def test_classify(self, file, c):
		total = 0
		correct = 0
		with open(file, "r") as myfile:
			lines = myfile.readlines()
			for line in lines:
				total = total + 1
				classified = self.logis_classify(line.split())
				if (classified == c):
					correct = correct + 1
		if (c=='+'):
			print "accuracy for positives: ", (correct + 0.0)/total
		else:
			print "accuracy for negatives: ", (correct + 0.0)/total

	def classify(self):
		pos_file = 'data/test_pos'
		neg_file = 'data/test_neg'
		self.test_classify(pos_file, '+')
		self.test_classify(neg_file, '-')

if __name__ == '__main__':
	my_logis = logis()
	my_logis.read_train('data/')
	my_logis.get_prob()
	my_logis.train()
	my_logis.classify()