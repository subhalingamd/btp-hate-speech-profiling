from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Running baselines')
    parser.add_argument('--merged_data', '-d', type=str, default='inputs/data_pan21_train_en_merged.tsv', help='merged data file')
    parser.add_argument('--dataset', type=str, default='pan21', choices=['pan21'], help='pan21')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'es'], help='en (english) or es (spanish)')
    parser.add_argument('--representation', '-r', type=str, default='tf-idf', choices=['tf-idf', 'count'] , help='representation')
    parser.add_argument('--classifier', '-c', type=str, default='lr', choices=['lr', 'svm', 'nb', 'rf', 'xgb', 'lgb'] , help='classifier')
    parser.add_argument('--output_dir', '-o', type=str, default='outputs/baselines/', help='output directory')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.mode == 'train':
        args.output_dir = args.output_dir.rstrip('/')
        out_prefix = "{}/{}_{}_{}_{}".format(args.output_dir, args.dataset, args.lang, args.representation, args.classifier)
        from models.baselines import train
        train(data=args.merged_data, rep=args.representation, cls=args.classifier, dump_objects_to=out_prefix+'.pkl', store_params_to=out_prefix+'.tsv')
    
    elif args.mode == 'test':
        pass