
# Switchboard Telephone Conversation Corpus
Switchboard (or SWDA) contains 1155 five-minute telephone conversations between two participants. Callers question receivers on provided topics, such as child care, recycling, and news media. 440 users participate in these 1155 conversations, producing 221616 utterances. 

The original dataset, paper, and accompanying information can be found at [http://compprag.christopherpotts.net/swda.html](http://compprag.christopherpotts.net/swda.html). 

## Dataset Details
### User-Level Information
In this dataset, users are the participants in the phone conversations (two per conversation). The user index is the same as that in the original dataset. We also provide the following user information in the metadata:
* sex: user sex, 'MALE' or 'FEMALE'
* education: the user's level of education (0, 1, 2, 3, or 9)
* birth_year: the user's birth year (4-digit year)
* dialect_area: one of the following dialect areas: MIXED, NEW ENGLAND, NORTH MIDLAND, NORTHERN, NYC, SOUTH MIDLAND, SOUTHERN, UNK, WESTERN

### Utterance-Level Information
For each utterance, we include:
* id: the unique ID of the utterance. It is formatted as "_conversation_id_"-"_position_of_utterance_". For example, ID 4325-0 is the first utterance in the conversation with ID 4325.
* user: the User giving the utterance
* root: id of the root utterance of the conversation. For example, the root of the utterance with ID 4325-1 would be 4325-0.
* reply_to: id of the utterance this replies to
* timestamp: timestamp of the utterance (not applicable in Switchboard, set to *None*)
* text: text of the utterance
* metadata
  * tag: the DAMSL act-tag of the utterance
  * pos: the part-of-speech tagged portion of the utterance
  * trees: nltk-parsed tree of the utterance

### Conversation-Level Information
Conversations are indexed by the first ID of the conversation (i.e. 4325-0, 2451-0, 4171-0, etc). The conversation IDs can be found using: 
```convo_ids = swda_corpus.get_conversation_ids()```

### Corpus-Level Information
The metadata associated with the corpus includes:
* talk_day: the datetime object with the time of the conversation
* topic_description: a short description of the conversation prompt
* length: length of the conversation in minutes
* prompt: a long description of the conversation prompt

## Usage and Stats
To download the corpus:
```python
>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("swda-corpus"))
```

In this dataset, there are 440 users, 221616 utterances, and 1155 conversations.

### Contact Information
Corpus translated into ConvoKit format by [Nathan Mislang](mailto:ntm39@cornell.edu), [Noam Eshed](mailto:ne236@cornell.edu), and [Sungjun Cho](mailto:sc782@cornell.edu).

