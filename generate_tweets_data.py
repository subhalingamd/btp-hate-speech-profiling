import pandas as pd
from sklearn.model_selection import train_test_split

tweet_col = 'text'
label_col = 'HS'

dataset = 'pan21:test'
assert dataset in ('hateval2019:train', 'hateval2019:val', 'hateval2019:train+val', 'hateval2019:test', 'hs', 'pan21:train', 'pan21:test')
    
if dataset.startswith('hateval2019'):
    from preprocessing.hateval2019 import read_dataset
    if dataset.endswith('train+val'):
        train = read_dataset('./data/hateval2019/hateval2019_en_train.csv')
        val = read_dataset('./data/hateval2019/hateval2019_en_dev.csv')
        data = pd.concat([train, val])
    else:
        if dataset.endswith('train'):
            path = './data/hateval2019/hateval2019_en_train.csv'
        elif dataset.endswith('val'):
            path = './data/hateval2019/hateval2019_en_dev.csv'
        elif dataset.endswith('test'):
            path = './data/hateval2019/hateval2019_en_test.csv'
        else:
            raise ValueError('dataset should be one of (train, val, test)')
        data = read_dataset(path)

elif dataset == 'hs':
    from preprocessing.hs import read_dataset
    path0 = './data/hs/neither.json'
    path1 = './data/hs/sexism.json'
    path2 = './data/hs/racism.json'

    df0 = read_dataset(path0)
    df0 = df0[[tweet_col]]
    df0[label_col] = 0

    df1 = read_dataset(path1)
    df1 = df1[[tweet_col]]
    df1[label_col] = 1

    df2 = read_dataset(path2)
    df2 = df2[[tweet_col]]
    df2[label_col] = 1

    data = pd.concat([df0, df1, df2])

elif dataset.startswith('pan21'):
    if dataset.endswith('train'):
        path = 'inputs/data_pan21_train_en_raw.tsv'
    elif dataset.endswith('test'):
        path = 'inputs/data_pan21_test_en_raw.tsv'
    else:
        raise ValueError('dataset should be one of (train, test)')
    tweet_cols = [f'tweet_{i+1}' for i in range(200)]
    data = pd.read_csv(path, sep='\t')
    data = pd.DataFrame([
        [idx, row[col], row['label']] for idx, row in data.iterrows() for col in tweet_cols
        ], columns=['id', tweet_col, label_col]
    )

else:
    raise ValueError('dataset not supported')


print(data.head())

print(data[label_col].value_counts())


# ## Stats

from eda.get_stats_tweets import *


data['hashtags'] = data[tweet_col].apply(count_per_tweet_hashtags)
data['urls'] = data[tweet_col].apply(count_per_tweet_urls)
data['users'] = data[tweet_col].apply(count_per_tweet_users)
data['rt'] = data[tweet_col].apply(count_per_tweet_rt)


data['uppercase_chars'] = data[tweet_col].apply(count_per_tweet_uppercase_chars)
data['chars'] = data[tweet_col].apply(count_per_tweet_chars)
data['uppercase_words'] = data[tweet_col].apply(count_per_tweet_uppercase_words)
data['words'] = data[tweet_col].apply(count_per_tweet_words)

data['stopwords'] = data[tweet_col].apply(count_per_tweet_stopwords)

data['emojis'] = data[tweet_col].apply(count_per_tweet_emojis)

data['sentiment_negative'], data['sentiment_neutral'], data['sentiment_positive'] = zip(*data[tweet_col].map(get_per_tweet_sentiments))
# data['sentiment'] = data[tweet_col].apply(get_per_tweet_sentiments_raw)

# out['PER'], out['ORG'], out['LOC'], out['MISC']
data['NER_PER'], data['NER_ORG'], data['NER_LOC'], data['NER_MISC'] = zip(*data[tweet_col].map(get_per_tweet_named_entities))


# ## Save features

if dataset.startswith('hateval2019'):
    data[[
        label_col, tweet_col,
        'hashtags', 'urls', 'users', 'rt', 
        'uppercase_chars', 'chars', 'uppercase_words', 'words', 
        'stopwords', 'emojis',
        'sentiment_negative', 'sentiment_neutral', 'sentiment_positive', # 'sentiment',
        'NER_PER', 'NER_ORG', 'NER_LOC', 'NER_MISC'
    ]].to_csv(f'./inputs/tweets/hateval2019_en_{dataset.split(":")[-1]}_.tsv', sep='\t', index=False)

elif dataset == 'hs':
    data_train, data_test = train_test_split(data, test_size=0.25, random_state=42, shuffle=True, stratify=data[label_col])
    print(data_train.head())
    data_train[[
        label_col, tweet_col,
        'hashtags', 'urls', 'users', 'rt',
        'uppercase_chars', 'chars', 'uppercase_words', 'words',
        'stopwords', 'emojis',
        'sentiment_negative', 'sentiment_neutral', 'sentiment_positive', # 'sentiment',
        'NER_PER', 'NER_ORG', 'NER_LOC', 'NER_MISC'
    ]].to_csv('./inputs/tweets/hs_en_train_.tsv', sep='\t', index=False)

    data_test[[
        label_col, tweet_col,
        'hashtags', 'urls', 'users', 'rt',
        'uppercase_chars', 'chars', 'uppercase_words', 'words',
        'stopwords', 'emojis',
        'sentiment_negative', 'sentiment_neutral', 'sentiment_positive', # 'sentiment',
        'NER_PER', 'NER_ORG', 'NER_LOC', 'NER_MISC'
    ]].to_csv('./inputs/tweets/hs_en_test_.tsv', sep='\t', index=False)

elif dataset.startswith('pan21'):
    data[[
        'id',
        label_col, tweet_col,
        'hashtags', 'urls', 'users', 'rt', 
        'uppercase_chars', 'chars', 'uppercase_words', 'words', 
        'stopwords', 'emojis',
        'sentiment_negative', 'sentiment_neutral', 'sentiment_positive', # 'sentiment',
        'NER_PER', 'NER_ORG', 'NER_LOC', 'NER_MISC'
    ]].to_csv(f'./inputs/tweets/pan21_en_{dataset.split(":")[-1]}_.tsv', sep='\t', index=False)
