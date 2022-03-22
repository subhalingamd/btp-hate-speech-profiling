from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Running baselines')
    parser.add_argument('--merged_data', '-d', type=str, default='inputs/data_pan21_train_en_merged.tsv', help='merged data file')
    parser.add_argument('--dataset', type=str, default='pan21', choices=['pan21'], help='pan21')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'es'], help='en (english) or es (spanish)')
    parser.add_argument('--representation', '-r', type=str, default='tf-idf', choices=['tf-idf', 'count'] , help='representation')
    parser.add_argument('--classifier', '-c', type=str, default='lr', choices=['lr', 'svm', 'nb', 'rf', 'xgb', 'lgb'] , help='classifier')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs/baselines/', help='output directory')
    parser.add_argument('--best_metrics_file', '-b', type=str, default='outputs/baselines/best_metrics.tsv', help='best metrics file')
    parser.add_argument('--model_path', '-m', type=str, default='outputs/baselines/pan21_en_tf-idf_lr.pkl', help='saved model path')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.mode == 'train':
        if "test" in args.merged_data:
            raise ValueError("Using test data for training is not allowed.")
        args.output_dir = args.output_dir.rstrip('/')
        out_prefix = "{}/{}_{}_{}_{}".format(args.output_dir, args.dataset, args.lang, args.representation, args.classifier)
        from models.baselines import train
        train(data=args.merged_data, rep=args.representation, cls=args.classifier, dump_objects_to=out_prefix+'.pkl', store_params_to=out_prefix+'.tsv')
    
    elif args.mode == 'test':
        if "train" in args.merged_data:
            raise ValueError("Using train data for testing is not allowed.")
        out_prefix = "{}/{}_{}_test_".format(args.output_dir, args.dataset, args.lang)
        from models.baselines import test
        test(data=args.merged_data, model_path=args.model_path, store_scores_to=out_prefix+'scores.tsv', store_predictions_to=out_prefix+'predictions.tsv', include_all_params=True)
