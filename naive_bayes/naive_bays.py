""" assumed there are multiple files for positive and negative instances """

from os import listdir
from os.path import isfile, join

class naive:
	def __init__(self):
		self.class_counts = {}
		self.class_counts_prob = {}
		self.class_word_counts = {}
		self.class_word_counts_prob = {}
		self.all_words = set([])

	def get_counts(self, train_file_dir, c):
		onlyfiles = [ join(train_file_dir, f) for f in listdir(train_file_dir) if isfile(join(train_file_dir, f)) ]
		self.class_counts[c] = len(onlyfiles)
		for f in onlyfiles:
			with open(f, "r") as myfile:
				data = myfile.read().replace('\n', '')
				for w in data.split():
					if w not in self.all_words:
						self.all_words.add(w)
					tup = (c, w)
					if tup not in self.class_word_counts:
						self.class_word_counts[tup] = 0
					else:
						self.class_word_counts[tup] = self.class_word_counts[tup] + 1
					


	def read_train(self, train_file_dir):
		# read positives
		pos_dir = train_file_dir + 'pos_train'
		self.get_counts(pos_dir, "+")
		neg_dir = train_file_dir + 'neg_train'
		self.get_counts(neg_dir, "-")

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


if __name__ == '__main__':
	my_bayes = naive()
	my_bayes.read_train('txt_sentoken/')
	my_bayes.get_prob()
	for k in my_bayes.class_counts:
		tup1 = (k, "movie")
		tup2 = (k, "film")
		print k, my_bayes.class_counts[k], my_bayes.class_word_counts[tup1], my_bayes.class_word_counts[tup2]
		print my_bayes.class_counts_prob[k]