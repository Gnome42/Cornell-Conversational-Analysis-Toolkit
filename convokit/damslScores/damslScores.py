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
	
			length = len(convo.get_utterance_ids())
			score = 0    
	
			for utt in convo.iter_utterances():
		
				score += self.rubric[utt.meta['tag']]
		
			convo.add_meta('damsl_score', score/length)
	
		return corpus
