import re
import tqdm
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import demoji
from textblob import TextBlob

corpora = 'xx_ent_wiki_sm'
nlp = spacy.load(corpora)

_ALL_HASH_FEATURES = r'#(HASHTAG|URL|USER|#RT#)#'

def count_per_tweet_features(tweet, feature=None, sub=None, user_fn=sum, tweet_fn=None):
    # tweet: String
    # get total count of feature in data
    # output out: int
    assert feature is not None or tweet_fn is not None, 'Either feature or tweet_fn must be specified'
    assert feature is None or tweet_fn is None, 'Only one of feature or tweet_fn can be specified'

    if tweet_fn is not None:
        return tweet_fn(tweet)

    if sub is None:
        return len(re.findall(feature, tweet))
    else:
        return len(re.findall(feature, re.sub(sub, '', tweet)))


def count_per_tweet_hashtags(data):
    return count_per_tweet_features(data, feature=r'#HASHTAG#')
       
def count_per_tweet_urls(data):
    return count_per_tweet_features(data, feature=r'#URL#')   

def count_per_tweet_users(data):
    return count_per_tweet_features(data, feature=r'#USER#')

def count_per_tweet_rt(data):
    return count_per_tweet_features(data, feature=r'##RT##')

def count_per_tweet_uppercase_chars(data):
    return count_per_tweet_features(data, feature=r'[A-Z]', sub=_ALL_HASH_FEATURES)

def count_per_tweet_chars(data):
    return count_per_tweet_features(data, tweet_fn= lambda tweet: len(re.sub(_ALL_HASH_FEATURES, '', tweet).strip()), )

def count_per_tweet_uppercase_words(data):
    return count_per_tweet_features(data, feature=r'[A-Z]\w+', sub=_ALL_HASH_FEATURES)

def count_per_tweet_words(data):
    return count_per_tweet_features(data, tweet_fn= lambda tweet: len(re.sub(r'^##RT##', '', tweet).split()), )


## @Source:: https://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation

def count_per_tweet_stopwords(tweet):
    return len([t for t in list(re.findall(r"[\w']+|[.,!?;]", re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip().lower())) if t in STOP_WORDS])

def count_per_tweet_emojis(tweet):
    return len(demoji.findall_list(tweet, desc=False))


def get_per_tweet_sentiments(tweet):
    sentiment = TextBlob(tweet).sentiment.polarity
    negative, neutral, positive = 0, 0, 0
    if sentiment > 0:
        positive = 1 
    elif sentiment < 0:
        negative = 1
    else:
        neutral = 1
    return negative, neutral, positive

def get_per_tweet_sentiments_raw(tweet):
    sentiment = TextBlob(tweet).sentiment.polarity
    return sentiment
   

def get_per_tweet_named_entities(tweet):
    out = {'PER': 0, 'ORG': 0, 'LOC': 0, 'MISC': 0}    
    
    doc = nlp(tweet)
    for ent in doc.ents:
        if ent.label_  not in out.keys():
            ent.label_ = 'MISC'
        out[ent.label_] += 1
    return out['PER'], out['ORG'], out['LOC'], out['MISC']

