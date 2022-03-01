import re
import pandas as pd

def preprocess_tweet(s):
    # replace hashtags
    s = re.sub(r'#(\w+)', '#HASHTAG#', s)

    # replace mentions
    s = re.sub(r'@(\w+)', '#USER#', s)
    
    # replace urls
    s = re.sub(r'http\S+', '#URL#', s)

    return s

def read_dataset(path):
    df = pd.read_json(path, lines=True)
    tweet_col = 'text'

    df[tweet_col] = df[tweet_col].apply(preprocess_tweet)
    
    # remove "RT #USER#: " from the beginning of each text
    df[tweet_col] = df[tweet_col].apply(lambda x: re.sub(r'^RT #USER#: ', '##RT## ', x))

    return df
