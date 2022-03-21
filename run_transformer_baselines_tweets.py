from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Running baselines')
    parser.add_argument('--data_path', '-d', type=str, default='inputs/tweets/hs+hateval2019_en_train__processed.tsv', help='data path')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs/feature_baselines_tweets/hs+hateval2019', help='output directory')

    parser.add_argument('--model_name', '-m', type=str, default="cardiffnlp/twitter-roberta-base-hate", help='saved model path')
    parser.add_argument('--mode', type=str, default='test', choices=['test'], help='train or test')
    parser.add_argument('--return_proba', action='store_true', default=False, help='return probability')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('*' * 32)
    print(args)
    print('*' * 32)
    print("\n")
    
    if args.mode == 'train':
        raise NotImplementedError('Not implemented yet.')

    elif args.mode == 'test':
        if "train" in args.data_path:
            raise ValueError("Using train data for testing is not allowed.")
        out_prefix = "{}/transformers_test_".format(args.output_dir)
        from models.tweets.transformer_baseline import test
        test(data=args.data_path, model_name=args.model_name, store_scores_to=out_prefix+'scores.tsv', store_predictions_to=out_prefix+'predictions.tsv', include_all_params=True, return_proba=args.return_proba)
