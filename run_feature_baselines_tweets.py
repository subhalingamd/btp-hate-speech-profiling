from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Running baselines')
    parser.add_argument('--data_path', '-d', type=str, default='inputs/tweets/hs+hateval2019_en_train__processed.tsv', help='data path')
    parser.add_argument('--classifier', '-c', type=str, default='lr', choices=['lr', 'svm', 'nb', 'rf', 'xgb', 'lgb'], help='classifier')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs/feature_baselines_tweets/hs+hateval2019', help='output directory')

    parser.add_argument('--model_path', '-m', type=str, default='outputs/feature_baselines_tweets/hs+hateval2019/rf.pkl', help='saved model path')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('*' * 32)
    print(args)
    print('*' * 32)
    print("\n")
    
    if args.mode == 'train':
        if "test" in args.data_path:
            raise ValueError("Using test data for training is not allowed.")
        args.output_dir = args.output_dir.rstrip('/')
        out_prefix = "{}/{}".format(args.output_dir, args.classifier)
        from models.tweets.feature_baselines import train
        train(data=args.data_path, cls=args.classifier, dump_objects_to=out_prefix+'.pkl', store_params_to=out_prefix+'.tsv')
    
    elif args.mode == 'test':
        if "train" in args.data_path:
            raise ValueError("Using train data for testing is not allowed.")
        out_prefix = "{}/test_".format(args.output_dir)
        from models.tweets.feature_baselines import test
        test(data=args.data_path, model_path=args.model_path, store_scores_to=out_prefix+'scores.tsv', store_predictions_to=None, include_all_params=True)
