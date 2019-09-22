from convokit.transformer import Transformer
from convokit.model import Corpus
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from collections import defaultdict
import numpy as np

class FirstImpression(Transformer):
	"""
	Performs sentiment analysis on the first 10% of conversations with a pretrained
	NLTK VADER analyzer. The sentiment of each user is stored on the conversation-level
	metadata under 'first_impression', and each sentiment is composed of 4 entries:
		- neg/neu/pos: the negative/neutral/positive sentiments. They range from 0
					to 1, and sum up to 1.
		- compound: An overall polarity score measured with token-level heuristics and 
					parameters from VADER. This value ranges from -1 (extremely negative) to 
					1 (extremely positive).
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

		return tokens

	def transform(self, corpus: Corpus):

		sid = SentimentIntensityAnalyzer()

		# overlap of vocabulary is a conversation-level metric
		for convo in corpus.iter_conversations():
	
			total_length = 0
			for utt in convo.iter_utterances():
				total_length += len(self._tokenize_utt(utt.text))


			users = convo.get_usernames()
			first_impressions = {u:defaultdict(float) for u in users}
			curr_user = users[0] # Current user
			curr_statement = [] # Current statement
			curr_length = 0 # How many tokens are covered so far
			
			for utt in convo.iter_utterances():
		
				# Tokenize via NLTK tokenizer
				tokens = self._tokenize_utt(utt.text)
						
				if utt.user.name == curr_user: # Utterance belongs to current user
					curr_statement += tokens
				else: # Utterance belongs to another user

					scores = sid.polarity_scores(' '.join(curr_statement))
					for k, v in scores.items():
						first_impressions[curr_user][k] += v

					# Move on to next user
					curr_user = utt.user.name
					curr_statement = []

					# stop loop if we covered 10% of total conversation
					if curr_length > total_length/10:
						break
	
				curr_length += len(tokens)

				
			# Take the average if multiple statements are counted
			for user in first_impressions.keys():
				num_statements = first_impressions[user]['neg'] + \
					first_impressions[user]['neu'] + first_impressions[user]['pos']
				if num_statements != 0:
					for k in first_impressions[user].keys():
						first_impressions[user][k] /= num_statements

			convo.add_meta('first_impression', first_impressions)

	
		return corpus
