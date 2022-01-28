import pandas as pd
import pickle
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def train(data, rep='tf-idf', cls='lr', dump_objects_to=None, store_params_to=None):
    rep = rep.lower()
    cls = cls.lower()
    assert rep in ['tf-idf', 'count'], 'Representation must be either tf-idf or count'
    assert cls in ['lr', 'svm'], 'Classifier must be either lr or svm'

    df = pd.read_csv(data, delimiter='\t')

    pipeline, parameters = [], {}
    if rep == 'tf-idf':
        pipeline.append(('vect', TfidfVectorizer(
                                        use_idf=True, 
                                        smooth_idf=True, 
                                        sublinear_tf=True, 
                                        token_pattern=r'[^\s]+',
                                        )
                        ))

        parameters.update({
            'vect__ngram_range': [(1,1), (1,2), (2,2)],
            'vect__min_df': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        })

    elif rep == 'count':
        pass

    if cls == 'lr':
        pipeline.append(('lr', LogisticRegression(
                                    verbose=0,
                                    max_iter=10000,
                                    random_state=42,
                                    )
                        ))


        parameters.update({
            'lr__C': [.01, .05, .1, .2, .5, 1, 2, 5, 10, 100, 1000],
            'lr__penalty': ['l2'], # ['l1', 'l2']
            'lr__solver': ['liblinear', 'saga'],
        })
        # TODO: try out elasticnet + saga ?

    elif cls == 'svm':
        pass


    pipeline = Pipeline(pipeline)
    grid_search = GridSearchCV(
                        estimator = pipeline, 
                        param_grid = parameters, 
                        cv = StratifiedKFold(10, shuffle=True, random_state=42),
                        scoring = 'accuracy',
                        n_jobs = -1,
                        return_train_score = True,
                        verbose = 4,
                    )

    print('Start time:\t', time.strftime("%m/%d/%Y %H:%M:%S"))
    params_grid = grid_search.fit(df['tweets'], df['label'])
    print('End time:\t',time.strftime("%m/%d/%Y %H:%M:%S"))

    if dump_objects_to is not None:
        save_pickle(params_grid, dump_objects_to)

    if store_params_to is not None:
        df = pd.DataFrame()
        df['params'] = params_grid.cv_results_['params']
        df['train_scores'] = params_grid.cv_results_['mean_train_score']
        df['test_scores'] = params_grid.cv_results_['mean_test_score']
        df.to_csv(store_params_to, sep='\t', index=False)
