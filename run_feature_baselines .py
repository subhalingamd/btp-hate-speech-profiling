from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Running baselines')
    parser.add_argument('--features_data', '-d', type=str, default='inputs/data_pan21_train_en_features.tsv', help='merged data file')
    parser.add_argument('--dataset', type=str, default='pan21', choices=['pan21'], help='pan21')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'es'], help='en (english) or es (spanish)')
    parser.add_argument('--classifier', '-c', type=str, default='lr', choices=['lr', 'svm', 'nb', 'rf', 'xgb', 'lgb'] , help='classifier')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs/feature_baselines/', help='output directory')
    parser.add_argument('--best_metrics_file', '-b', type=str, default='outputs/feature_baselines/best_metrics.tsv', help='best metrics file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.mode == 'train':
        args.output_dir = args.output_dir.rstrip('/')
        out_prefix = "{}/{}_{}_{}_{}".format(args.output_dir, args.dataset, args.lang, args.representation, args.classifier)
        from models.feature_baselines import train
        train(data=args.features_data, cls=args.classifier, dump_objects_to=out_prefix+'.pkl', store_params_to=out_prefix+'.tsv')
    
    elif args.mode == 'test':
        pass