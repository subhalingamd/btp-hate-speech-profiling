import re
import pandas as pd
import demoji

try:
    from tqdm import tqdm
except:
    tqdm = lambda x: x

def to_csv(data, file_name, sep='\t'):
    df = pd.DataFrame(data)
    df.to_csv(file_name, sep=sep, index=False)

def emoji_handler(text):
    text = demoji.replace(text, repl="##EMOJI##")
    return text

def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = emoji_handler(tweet)
        tweet = tweet.lower()
        tweet = re.sub('[,.\'\"\‘\’\”\“]', '', tweet)
        # tweet = re.sub(r'([a-z\'-\’]+)', r'\1 ', tweet)
        # tweet = re.sub(r'(?<![?!:;/])([:\'\";.,?()/!])(?= )','',tweet)
        tweet = re.sub(r'([a-z0-9\'-\’\s\#]+)', r'\1 ', tweet)
        tweet = re.sub(r'#(#rt#|#emoji#|user|hashtag|url)#', r' #\1# ', tweet)
        tweet = re.sub(r'(?<![?!:;/])([:\'\";.,?()/!])(?= )','',tweet)
        tweet = re.sub(r'\s+', ' ', tweet)
        tweet = re.sub('[\n]', ' ', tweet)
        cleaned_tweets.append(tweet)
    return cleaned_tweets

def preprocess_tweets(data, num_tweets=200, merge_tweets=False):
    if merge_tweets:
        preprocessed_tweets = {"tweets": [], "label": []}
    else:
        preprocessed_tweets = {"tweet_{}".format(i+1): [] for i in range(num_tweets)}
        preprocessed_tweets.update({"label": []})
    for label, users in data.items():
        for user in tqdm(users):
            assert len(user) == num_tweets, "Number of tweets for a user is not {}".format(num_tweets)
            cleaned_tweets = clean_tweets(user)
            preprocessed_tweets["label"].append(label)
            if merge_tweets:
                preprocessed_tweets["tweets"].append(
                    " ".join(cleaned_tweets)
                )
            else:
                for num, tweet in enumerate(cleaned_tweets):
                    preprocessed_tweets["tweet_{}".format(num+1)].append(tweet)
    return preprocessed_tweets

def preprocess_data_and_save(data, file_name, sep='\t', num_tweets=200, merge_tweets=False):
    preprocessed_data = preprocess_tweets(data, num_tweets=num_tweets, merge_tweets=merge_tweets)
    to_csv(preprocessed_data, file_name, sep=sep)