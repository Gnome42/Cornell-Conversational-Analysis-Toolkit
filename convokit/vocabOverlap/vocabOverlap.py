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


	# Iterate thru different vocabularies to find the intersection and the overlap ratio
	def _compute_overlap(self, vocabs):

		overlapVocab = set(list(vocabs.values())[0])
		for u, vocab in vocabs.items():
			overlapVocab = overlapVocab.intersection(set(vocab))
	
		# Compute frequency of overlaps and total number of tokens
		overlap = 0
		total = 0
		for vocab in vocabs.values():
			total += sum(vocab.values())
			for k, v in vocab.items():
				if k in overlapVocab:
					overlap += v

		if overlap == 0 or total == 0:
			return set([]), 0
		else:	
			ratio = overlap/total
			return overlapVocab, ratio

	def transform(self, corpus: Corpus):

		stop_words = set(stopwords.words('english'))

		# overlap of vocabulary is a conversation-level metric
		for convo in corpus.iter_conversations():
	
			users = convo.get_usernames()
			vocabs = {u:defaultdict(int) for u in users}
	
			for utt in convo.iter_utterances():
		
				# Tokenize via NLTK tokenizer
				tokens = self._tokenize_utt(utt.text)
		
				# Filter out stop words
				tokens = [token for token in tokens if not token in stop_words]
				
				for token in tokens:
					vocabs[utt.user.name][token] += 1

				overlapVocab, ratio = self._compute_overlap(vocabs)	
				utt.add_meta('vocabulary_overlap', {'vocab': overlapVocab, 'ratio': ratio})
	
			# Compute frequency of overlaps and total number of tokens
			overlapVocab, ratio = self._compute_overlap(vocabs)
		
			convo.add_meta('vocabulary_overlap', {'vocab': overlapVocab, 'ratio': ratio})

	
		return corpus
