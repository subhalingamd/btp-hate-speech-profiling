import pandas as pd
NUM_TWEETS_PER_USER = 200

path_prefix = "outputs/feature_baselines_tweets/metrics/pan21/test/test_predictions"


df = pd.read_csv(path_prefix + '.tsv', sep='\t')

data = []

for i in range(0, len(df), NUM_TWEETS_PER_USER):
    data.append([df.iloc[i]['gold']]+df.iloc[i:i+NUM_TWEETS_PER_USER, :]["['xgb.pkl']"].to_list())

data = pd.DataFrame(data, columns=['gold']+['prediction_'+str(i+1) for i in range(NUM_TWEETS_PER_USER)])

data.to_csv(path_prefix + '_aggregated.tsv', sep='\t', index=False)


import pandas as pd
NUM_TWEETS_PER_USER = 200

path_prefix = "outputs/feature_baselines_tweets/metrics/pan21/train/test_predictions"


data = pd.read_csv(path_prefix + '.tsv', sep='\t')

data = []

for i in range(0, len(data), NUM_TWEETS_PER_USER):
    data.append([data.iloc[i]['gold']]+data.iloc[i:i+NUM_TWEETS_PER_USER]['prediction'].to_list())

data = pd.DataFrame(data, columns=['gold']+['prediction'+str(i) for i in range(NUM_TWEETS_PER_USER)])

data.to_csv(path_prefix + '_aggregated.tsv', sep='\t', index=False)


