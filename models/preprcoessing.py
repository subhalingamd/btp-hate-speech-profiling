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

def clean_tweet(tweet):
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
    tweet = tweet.strip()
    return tweet

def clean_tweets(tweets):
    cleaned_tweets = []
    for tweet in tweets:
        tweet = clean_tweet(tweet)
        cleaned_tweets.append(tweet)
    return cleaned_tweets

def preprocess_tweets(data, num_tweets=200, merge_tweets=False, no_clean=False):
    if merge_tweets:
        preprocessed_tweets = {"tweets": [], "label": []}
    else:
        preprocessed_tweets = {"tweet_{}".format(i+1): [] for i in range(num_tweets)}
        preprocessed_tweets.update({"label": []})
    for label, users in data.items():
        for user in tqdm(users):
            assert len(user) == num_tweets, "Number of tweets for a user is not {}".format(num_tweets)
            if no_clean is True:
                cleaned_tweets = user.copy()
            else:
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

def preprocess_data_and_save(data, file_name, sep='\t', num_tweets=200, merge_tweets=False, no_clean=False):
    preprocessed_data = preprocess_tweets(data, num_tweets=num_tweets, merge_tweets=merge_tweets, no_clean=no_clean)
    to_csv(preprocessed_data, file_name, sep=sep)


# Given a csv file with label, perform random undersampling of the majority class


def undersample(df, label_col, sampler='random', sampling_strategy=1.0, random_state=42):
    assert sampler in ['random', ], "Sampler must be 'random'"
    print(df[label_col].value_counts())
    assert df[label_col].isin([0, 1]).all(), f'Label column ({label_col}) must be binary'

    print("Original dataset shape: {}".format(df.shape))
    if sampler == 'random':
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    
    df, _ = sampler.fit_resample(df, df[label_col])
    print("Undersampled dataset shape: {}".format(df.shape))

    return df