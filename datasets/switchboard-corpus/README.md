
# Switchboard Dialog Act Corpus (SwDA)
The Switchboard Dialog Act Corpus (SWDA) contains 1,155 five-minute telephone conversations between two participants. Callers question receivers on provided topics, such as child care, recycling, and news media. 440 users participate in these 1,155 conversations, producing 221,616 utterances (we combine consecutive utterances by the same person into one utterance, so our corpus has 122,646 utterances). 

You should pull the repo at [https://github.com/cgpotts/swda](https://github.com/cgpotts/swda) in order to download the dataset and helper functions necessary to create the corpus.

Original paper: Andreas Stolcke, Klaus Ries, Noah Coccaro, Elizabeth Shriberg, Rebecca Bates, Daniel Jurafsky, Paul Taylor, Rachel Martin, Carol Van Ess-Dykema, and Marie Meteer. **Dialogue act modeling for automatic tagging and recognition of conversational speech**. Computational Linguistics, Volume 26, Number 3, September 2000.
[https://www.aclweb.org/anthology/J00-3003.pdf](https://www.aclweb.org/anthology/J00-3003.pdf)

The original dataset and additional information can be found at [http://compprag.christopherpotts.net/swda.html](http://compprag.christopherpotts.net/swda.html). 

## Dataset Details
### User-Level Information
In this dataset, users are the participants in the phone conversations (two per conversation). The user's name is the same as the ID used in the original SwDA dataset. We also provide the following user information in the metadata:
* sex: user sex, 'MALE' or 'FEMALE'
* education: the user's level of education. Options are 0 (less than high school), 1 (less than college), 2 (college), 3 (more than college), and 9 (unknown).
* birth_year: the user's birth year (4-digit year)
* dialect_area: one of the following dialect areas: MIXED, NEW ENGLAND, NORTH MIDLAND, NORTHERN, NYC, SOUTH MIDLAND, SOUTHERN, UNK, WESTERN
    * The UNK tag is used for users of unknown dialect area
    * The MIXED tag is used for users who are of multiple dialect areas


### Utterance-Level Information
For each utterance, we include:
* id: the unique ID of the utterance. It is formatted as "_conversation_id_"-"_position_of_utterance_". For example, ID 4325-0 is the first utterance in the conversation with ID 4325.
* user: the User giving the utterance
* root: id of the root utterance of the conversation. For example, the root of the utterance with ID 4325-1 would be 4325-0.
* reply_to: id of the utterance this replies to
* timestamp: timestamp of the utterance (not applicable in SwDA, set to *None*)
* text: text of the utterance
* metadata
  * tag: a dictionary with segments of the utterance text as keys and the DAMSL act-tag of the utterance as values

### Conversation-Level Information
Conversations are indexed by the original SwDa dataset IDs (i.e. 4325, 2451, 4171, etc). The conversation IDs can be found using: 
```convo_ids = swda_corpus.get_conversation_ids()```

* talk_day: the date of the conversation
* topic_description: a short description of the conversation prompt
* length: length of the conversation in minutes
* prompt: a long description of the conversation prompt
* from_caller: id of the from-caller (A) of the conversation
* to_caller: id of the to-caller (B) of the conversation


## Usage and Stats
To download the corpus:
```python
>>> from convokit import Corpus, download
>>> corpus = Corpus(filename=download("switchboard-corpus"))
```

For quick statistics:
```python
>>> corpus.print_summary_stats()
Number of Users: 440
Number of Utterances: 122646
Number of Conversations: 1155
```

## Additional Information

*Note:* In the original SwDa dataset, utterances are not separated by user, but rather by tags. This means that consecutive utterances could have been said by the same user. In the ConvoKit Corpus, we changed this so that each utterance in our corpus is a collection of the consecutive sub-utterances said by one person. The metadata on each utterance is combined from the sub-utterances of the original dataset, so that it is clear which POS and DAMSL tags correspond with which parts of each utterance.

### Licensing Information
The SWDA Switchboard work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License (see source [here](http://compprag.christopherpotts.net/swda.html))

### Contact Information
Corpus translated into ConvoKit format by [Nathan Mislang](mailto:ntm39@cornell.edu), [Noam Eshed](mailto:ne236@cornell.edu), and [Sungjun Cho](mailto:sc782@cornell.edu).

