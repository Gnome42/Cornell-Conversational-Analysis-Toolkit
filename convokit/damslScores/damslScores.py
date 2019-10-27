from convokit.transformer import Transformer
from convokit.model import Corpus
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import numpy as np
import csv

class DAMSLScores(Transformer):
	"""
	Measures the quality of a conversation with the DAMSL action-tags given for
	each utterance. The scoring scheme is hand-crafted, and is stored in a csv file
	(damsl_rubric.txt). The final score for each conversation is simply the average
	score across all utterances in that conversation, and the score is recorded
	on the conversation-level metadata under 'damsl_score'
	"""

	def __init__(self, filename: str):

		self.rubric = {}
		with open(filename) as f:
			csv_reader = csv.reader(f, delimiter=',')
			for row in csv_reader:
				self.rubric[row[0]] = int(row[1])
	

	def transform(self, corpus: Corpus):

		for convo in corpus.iter_conversations():
	
			scores = []
	
			for utt in convo.iter_utterances():
				
				tags = [pair[1] for pair in utt.meta['tag']]
				for tag in tags:
					try:
						scores.append(self.rubric[tag])
					except KeyError:
						continue
		
			convo.add_meta('damsl_score', np.mean(scores))
	
		return corpus
