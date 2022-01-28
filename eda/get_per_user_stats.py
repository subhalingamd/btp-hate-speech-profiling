import re
_ALL_HASH_FEATURES = r'#(HASHTAG|URL|USER|#RT#)#'

def count_per_user_features(data, feature=None, sub=None, user_fn=sum, tweet_fn=None):
    # given data: {label: [[tweets]]
    # get total count of feature for each label
    # output out: {label: [count]}
    assert user_fn is not None, 'user_fn must be specified'
    assert feature is not None or tweet_fn is not None, 'Either feature or tweet_fn must be specified'
    assert feature is None or tweet_fn is None, 'Only one of feature or tweet_fn can be specified'

    if tweet_fn is not None:
        return {label: [user_fn([tweet_fn(tweet) for tweet in user]) for user in users] for label, users in data.items()}

    if sub is None:
        return {label: [user_fn([len(re.findall(feature, tweet)) for tweet in user]) for user in users] for label, users in data.items()}
    else:
        return {label: [user_fn([len(re.findall(feature, re.sub(sub, '', tweet))) for tweet in user]) for user in users] for label, users in data.items()}


def count_per_user_hashtags(data):
    return count_per_user_features(data, feature=r'#HASHTAG#')
       
def count_per_user_urls(data):
    return count_per_user_features(data, feature=r'#URL#')   

def count_per_user_users(data):
    return count_per_user_features(data, feature=r'#USER#')

def count_per_user_rt(data):
    return count_per_user_features(data, feature=r'##RT##')

def count_per_user_uppercase_chars(data):
    return count_per_user_features(data, feature=r'[A-Z]', sub=_ALL_HASH_FEATURES)

def count_per_user_min_chars(data):
    return count_per_user_features(data, user_fn=min, tweet_fn= lambda tweet: len(re.sub(_ALL_HASH_FEATURES, '', tweet).strip()), )

def count_per_user_max_chars(data):
    return count_per_user_features(data, user_fn=max, tweet_fn= lambda tweet: len(re.sub(_ALL_HASH_FEATURES, '', tweet).strip()), )

def count_per_user_chars(data):
    return count_per_user_features(data, user_fn=sum, tweet_fn= lambda tweet: len(re.sub(_ALL_HASH_FEATURES, '', tweet).strip()), )

def count_per_user_uppercase_words(data):
    return count_per_user_features(data, feature=r'[A-Z]\w+', sub=_ALL_HASH_FEATURES)

def count_per_user_min_words(data):
    return count_per_user_features(data, user_fn=min, tweet_fn= lambda tweet: len(re.sub(_ALL_HASH_FEATURES, '', tweet).split()), )

def count_per_user_max_words(data):
    return count_per_user_features(data, user_fn=max, tweet_fn= lambda tweet: len(re.sub(_ALL_HASH_FEATURES, '', tweet).split()), )

def count_per_user_words(data):
    return count_per_user_features(data, user_fn=sum, tweet_fn= lambda tweet: len(re.sub(_ALL_HASH_FEATURES, '', tweet).split()), )

def count_per_user_min_words_alt(data):
    return count_per_user_features(data, user_fn=min, tweet_fn= lambda tweet: len(re.sub(r'^##RT##', '', tweet).split()), )

def count_per_user_max_words_alt(data):
    return count_per_user_features(data, user_fn=max, tweet_fn= lambda tweet: len(re.sub(r'^##RT##', '', tweet).split()), )

def count_per_user_words_alt(data):
    return count_per_user_features(data, user_fn=sum, tweet_fn= lambda tweet: len(re.sub(r'^##RT##', '', tweet).split()), )


## @Source:: https://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation

def count_per_user_stopwords(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get total number of stopwords for each label
    # output out: {label: [count]}
    from spacy.lang.en.stop_words import STOP_WORDS
    out = {}
    for label, users in data.items():
        out[label] = []
        for user in users:
            out[label] += [sum([len([t for t in list(re.findall(r"[\w']+|[.,!?;]", re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip().lower())) if t in STOP_WORDS]) for tweet in user])]
    return out

def count_per_user_emojis(data):
    # given data: {label: [[tweets]]
    # get total number of emojis for each label
    # output out: {label: [count]}
    # @TODO:: include emojis with just characters-- like :) and <3...
    import demoji
    out = {}
    for label, users in data.items():
        out[label] = []
        for user in users:
            out[label] += [sum([len(demoji.findall_list(tweet, desc=False)) for tweet in user])]
    return out


def get_per_user_sentiments(data):
    # given data: {label: [[tweets]]
    # get sentiment for each label
    # output out: {label: sentiment}
    from textblob import TextBlob
    out = {"positive": {}, "negative": {}, "neutral": {}}
    for label, users in data.items():
        out["positive"][label], out["negative"][label], out["neutral"][label] = [], [], []
        for user in users:
            positive, negative, neutral = 0, 0, 0
            for tweet in user:
                sentiment = TextBlob(tweet).sentiment.polarity
                if sentiment > 0:
                    positive += 1
                elif sentiment < 0:
                    negative += 1
                else:
                    neutral += 1
            out["positive"][label] += [positive]
            out["negative"][label] += [negative]
            out["neutral"][label] += [neutral]
    return out

def get_per_user_named_entities(data, corpora='en_core_web_sm'):
    # given data: {label: [[tweets]]
    # get named entities for each label
    # output out: {label: [count]}
    assert corpora in ['en_core_web_sm', 'en_core_web_md', 'xx_ent_wiki_sm']

    import spacy
    import tqdm
    nlp = spacy.load(corpora)
    out = {'PERSON': {}, 'PER': {}, 'ORG': {}, 'GPE': {}, 'LOC': {}, 'MISC': {}}
    unk = set([])
    for label, users in data.items():
        for ner in out.keys():
            out[ner][label] = []
        for user in tqdm.tqdm(users):
            for ner in out.keys():
                out[ner][label].append(0)
            for tweet in user:
                doc = nlp(tweet)
                for ent in doc.ents:
                    if ent.label_  not in out.keys():
                        unk.add(ent.label_)
                        ent.label_ = 'MISC'
                    out[ent.label_][label][-1] += 1
    print(unk)
    return out

def get_unique_tweets_ratio(data):
    # given data: {label: [[tweets]]
    # get ratio of unique tweets for each label
    # output out: {label: [count]}
    out = {}
    for label, users in data.items():
        out[label] = []
        for user in users:
            out[label] += [len(set(user))/len(user)]
    return out