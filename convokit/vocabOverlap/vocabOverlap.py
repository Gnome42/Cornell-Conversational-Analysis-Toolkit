from convokit.transformer import Transformer
from convokit.model import Corpus
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import numpy as np

class VocabOverlap(Transformer):
	"""
	Calculates the degree of vocabulary repetition among the participants
	of a conversation. This metric is simply the proportion of tokens that all
	participants have used within a conversation, excluding stop words such
	as "the", "is", "at", or "which".

	This metric is added to the conversation-level metadata under 'vocabulary_overlap'
	and has a value between 0 (no overlap at all) and 1 (all tokens overlap)
	"""

	def __init__(self):
		pass

	def _tokenize_utt(self, utterance: str):
		# Strips punctuation from utterance and returns a list of tokens
		tokenizer = RegexpTokenizer(r'\w+')	
		tokens = tokenizer.tokenize(utterance)

		# Remove any tokens that are a single letter other than 'I' or 'a'
		# because they are tags for the utterance tree
		for t in tokens:
			if len(t) == 1 and not (t == 'I' or t == 'a'):
					tokens.remove(t)

		return [token.lower() for token in tokens]

	def transform(self, corpus: Corpus):

		stop_words = set(stopwords.words('english'))

		# overlap of vocabulary is a conversation-level metric
		for convo in corpus.iter_conversations():
	
			users = convo.get_usernames()
			vocabA = defaultdict(int)
			vocabB = defaultdict(int)
	
			for utt in convo.iter_utterances():
		
				# Tokenize via NLTK tokenizer
				tokens = self._tokenize_utt(utt.text)
		
				# Filter out stop words
				tokens = [token for token in tokens if not token in stop_words]
		
				if utt.user.name == users[0]: # Utterance belongs to user A
					for token in tokens:
						vocabA[token] += 1
				else: # Utterance belongs to user B
					for token in tokens:
						vocabB[token] += 1

			overlapVocab = set(vocabA.keys()).intersection(set(vocabB.keys()))
	
			# Compute total frequency of overlaps
			overlap = 0
			for k, v in vocabA.items():
				if k in overlapVocab:
					overlap += v
			for k, v in vocabB.items():
				if k in overlapVocab:
					overlap += v
		
			# Compute total number of tokens used
			total = sum(vocabA.values())+sum(vocabB.values())
			convo.add_meta('vocabulary_overlap', overlap/total)

	
		return corpus
