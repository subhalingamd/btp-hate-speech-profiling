from argparse import ArgumentParser
from models.preprcoessing import clean_tweet, undersample
import pandas as pd

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default='inputs/tweets/hateval2019_en_train_.tsv', help='data path')
    parser.add_argument('--output_dir', type=str, default='inputs/tweets/', help='output directory')
    parser.add_argument('--tweet_col', type=str, default='text', help='tweet column name')
    parser.add_argument('--label_col', type=str, default='HS', help='label column name')
    parser.add_argument('--sampler', type=str, default=None, choices=[None, 'undersample-random', ], help='sampler')
    # parser.add_argument('--sep', type=str, default='\t', help='separator')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    args.sep = '\t'
    out_file = args.output_dir + '/' + args.data_path.split('/')[-1][:-len('.tsv')] + '_processed' + '.tsv'

    df = pd.read_csv(args.data_path, sep=args.sep)
    df[args.tweet_col] = df[args.tweet_col].astype('str')
    df[args.tweet_col] = df[args.tweet_col].apply(clean_tweet)

    if args.sampler is not None:
        if 'test' in args.data_path:
            raise ValueError('test data should not be undersampled')
        if args.sampler.startswith('undersample'):
            sampler_type = args.sampler.split('-')[1]
            df = undersample(df, label_col=args.label_col, sampler=sampler_type, random_state=42)

    df.to_csv(out_file, sep=args.sep, index=False)
    