from convokit.transformer import Transformer
from convokit.model import Corpus
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from collections import defaultdict
import numpy as np

class VaderSentiment(Transformer):
	"""
	Performs sentiment analysis on the first 10% of conversations with a pretrained
	NLTK VADER analyzer. The sentiment of each user is stored on the conversation-level
	metadata under 'initial_sentiment', and each sentiment is composed of 4 entries:
		- neg/neu/pos: the negative/neutral/positive sentiments. They range from 0
					to 1, and sum up to 1.
		- compound: An overall polarity score measured with token-level heuristics and 
					parameters from VADER. This value ranges from -1 (extremely negative) to 
					1 (extremely positive).
	In addition to the conversation-level first impression, the sentiment of each statement
	made in the conversation is also stored in the utterance-level metadata. Here,
	a statement is consistent of consecutive utterances from the same user.
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

		for convo in corpus.iter_conversations():
	
			# Compute polarity scores of each statement (set of utterances)
			users = convo.get_usernames()
			curr_user = users[0] # Current user
			curr_statement = [] # Current statement
			curr_utts = [] # Current list of utterances

			# Compute polarity score for the first 10% of the conversation
			for utt in convo.iter_utterances():
		
				# Tokenize via NLTK tokenizer
				tokens = self._tokenize_utt(utt.text)
						
				if utt.user.name == curr_user: # Utterance belongs to current user
					curr_utts.append(utt)
					curr_statement += tokens
				else: # Utterance belongs to another user

					# Compute polarity scores for current statement
					scores = sid.polarity_scores(' '.join(curr_statement))
					for u in curr_utts:
						u.add_meta('polarity', scores)

					# Move on to next user
					curr_user = utt.user.name
					curr_statement = tokens
					curr_utts = [utt]

			# flush out remaining scores
			scores = sid.polarity_scores(' '.join(curr_statement))
			for u in curr_utts:
				u.add_meta('polarity', scores)


			total_length = 0
			for utt in convo.iter_utterances():
				total_length += len(self._tokenize_utt(utt.text))


			initial_sentiment = {u:defaultdict(float) for u in users}
			curr_user = users[0] # Current user
			curr_statement = [] # Current statement
			curr_length = 0 # How many tokens are covered so far

			# Compute polarity score for the first 10% of the conversation
			for utt in convo.iter_utterances():
		
				# Tokenize via NLTK tokenizer
				tokens = self._tokenize_utt(utt.text)
						
				if utt.user.name == curr_user: # Utterance belongs to current user
					curr_statement += tokens
				else: # Utterance belongs to another user

					# Compute polarity scores for current statement
					scores = sid.polarity_scores(' '.join(curr_statement))
					for k, v in scores.items():
						initial_sentiment[curr_user][k] += v

					# Move on to next user
					curr_user = utt.user.name
					curr_statement = []

					# stop loop if we covered 10% of total conversation
					if curr_length > total_length/10:
						break
	
				curr_length += len(tokens)

				
			# Take the average if multiple statements are counted
			for user in initial_sentiment.keys():
				num_statements = initial_sentiment[user]['neg'] + \
					initial_sentiment[user]['neu'] + initial_sentiment[user]['pos']
				if num_statements != 0:
					for k in initial_sentiment[user].keys():
						initial_sentiment[user][k] /= num_statements

			convo.add_meta('initial_sentiment', initial_sentiment)

	
		return corpus
