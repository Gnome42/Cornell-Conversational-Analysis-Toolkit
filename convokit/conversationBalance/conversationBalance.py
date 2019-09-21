from convokit.transformer import Transformer
from convokit.model import Corpus
from nltk.tokenize import RegexpTokenizer
import numpy as np

class ConversationBalance(Transformer):
	"""
	Calculates the balance of a conversation by computing the ratio of tokens 
	spoken between a pair of users. The Conversation-level balance is defined 
	in conversation metadata for each pair of users (A,B) as the ratio: 
			(# tokens by A)/(# tokens by B)
	This is stored as a Numpy array of size NxN, where N is the number of Users.
	In cell (X, Y) of the array is the balance between users X and Y. Note
	that the value at cell (X,Y) is the inverse of the value at cell (Y,X)
	and that the value at cell (X,X) is always 1.

	Additionally, statement-pair balance is computed. We define statements as
	groups of consecutive utterances spoken by the same user. The balance
	between two consecutive statements (a, b) is a:b and is stored in the 
	utterance metadata of the first utterance in statement a.

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
		# This measure is calculated on a conversation-level and statement-pair
		# level
		for c in corpus.iter_conversations():

			# Update the beginning of each statement with the statement length
			user_tokens = {} 	
			user_order = []	
			prev_user = ''	# User of the last utterance 
			cur_user = ''	# User of the current utterance
			statement_root_id = ''
			statement_roots = []
			for u in c._utterance_ids:

				# Update current user
				prev_user = cur_user
				cur_user = corpus.utterances[u].user.name

				# A user has started new statement, add metadata field
				if cur_user != prev_user:
					statement_root_id = u
					statement_roots.append(u)
					corpus.utterances[u].meta['statement_len'] = 0

				# Extract tokens from this utterance
				utt_tokens = self._tokenize_utt(corpus.utterances[u].text)

				if cur_user not in user_tokens:	# New user
					user_tokens[cur_user] = 0
					user_order.append(cur_user)
				user_tokens[cur_user] += len(utt_tokens)
				corpus.utterances[statement_root_id].meta['statement_len'] += len(utt_tokens)

			# Update utterance-level metadata with balance ratio
			for i in range(len(statement_roots) - 1):
				utt_id_cur = statement_roots[i]
				utt_id_next = statement_roots[i+1]

				cur_len = corpus.utterances[utt_id_cur].meta['statement_len']
				next_len = corpus.utterances[utt_id_next].meta['statement_len']

				if next_len == 0:
					sment_balance = 1
				else:
					sment_balance = cur_len/next_len
				corpus.utterances[utt_id_cur].meta['statement_balance'] = sment_balance

			# Update conversation-level metadata with balance ratio
			convo_balance = np.zeros((len(user_order), len(user_order)))
			for i, A in enumerate(user_order):
				for j, B in enumerate(user_order):
					convo_balance[i,j] = user_tokens[A]/user_tokens[B]
			c._meta['conversation_balance'] = convo_balance

			# Add the usernames to the conversation metadata
			c._usernames = user_order
			

		return(corpus)


