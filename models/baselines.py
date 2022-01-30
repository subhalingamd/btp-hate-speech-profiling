import pandas as pd
import pickle
import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def train(data, rep='tf-idf', cls='lr', dump_objects_to=None, store_params_to=None):
    rep = rep.lower()
    cls = cls.lower()

    ALLOWED_REP = ['tf-idf', 'count']
    assert rep in ALLOWED_REP, 'Representation must be from {}'.format(str(ALLOWED_REP))

    ALLOWED_CLS = ['lr', 'svm', 'nb', 'rf', 'xgb']
    assert cls in ALLOWED_CLS, 'Classifier must be from {}'.format(str(ALLOWED_CLS))

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
            'vect__ngram_range': 
                                    [(1,1), (1,2), (2,2)] if cls in ['lr', 'svm', 'nb', 'rf'] 
                                    else [(1,1), (1,2)] ,
            'vect__min_df': 
                                    [2, 3, 5, 7, 10] if cls in ['lr', 'svm', 'nb', 'rf']
                                    else [2, 5, 10],
            'vect__max_df': 
                                    [0.75, 1.0],
        })

    elif rep == 'count':
        pipeline.append(('vect', CountVectorizer(
                                        token_pattern=r'[^\s]+',
                                        )
                        ))

        parameters.update({
            'vect__ngram_range': 
                                    [(1,1), (1,2), (2,2)] if cls in ['lr', 'svm', 'nb', 'rf'] 
                                    else [(1,1), (1,2)] ,
            'vect__min_df': 
                                    [2, 3, 5, 7, 10] if cls in ['lr', 'svm', 'nb', 'rf']
                                    else [2, 5, 10],
            'vect__max_df': 
                                    [0.75, 1.0],
        })
    

    if cls == 'lr':
        pipeline.append(('lr', LogisticRegression(
                                    verbose=0,
                                    max_iter=10000,
                                    random_state=42,
                                    )
                        ))


        parameters.update({
            'lr__C': [.1, .2, .5, 1, 2, 5, 10, 20, 50, 100],
            'lr__penalty': ['l2'], # ['l1', 'l2']
            'lr__solver': ['saga'], # ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        })
        # TODO: try out elasticnet + saga ?

    elif cls == 'svm':
        pipeline.append(('svm', SVC(
                                    # probability = True,
                                    verbose=0,
                                    max_iter=10000,
                                    random_state=42,
                                    )
                        ))

        parameters.update({
            'svm__C': [.1, .5, 1, 2, 5, 10, 50],
            'svm__kernel': ['linear', 'rbf'], # ['rbf', 'linear', 'poly', 'sigmoid']
            # 'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        })

    elif cls == 'nb':
        pipeline.append(('nb', MultinomialNB(
                                    )
                        ))

        parameters.update({
            'nb__alpha': [0.1, 0.5, 1, 2, 5, 10, 50],
        })
    
    elif cls == 'rf':
        pipeline.append(('rf', RandomForestClassifier(
                                    verbose=0,
                                    random_state=42,
                                    )
                        ))

        parameters.update({
            'rf__n_estimators': [20, 50, 100, 200, 500],
            'rf__max_depth': [None, 5, 10, 50, 100],
            'rf__min_samples_leaf':  [1, 2, 4],
            'rf__max_features': ['auto'], # ['auto', 'sqrt', 'log2', None]
            'rf__bootstrap': [True], # [True, False]
            'rf__oob_score': [True], # [True, False]
        })

    elif cls == 'xgb':
        pipeline.append(('xgb', XGBClassifier(
                                    use_label_encoder=False,
                                    verbosity=0,
                                    random_state=42,
                                    )
                        ))

        parameters.update({
            'xgb__n_estimators': [50, 100, 200, 500],
            'xgb__max_depth': [3, 5, 7, 10],
            'xgb__eta': [0.01, 0.1, 0.3],
            "xgb__subsample": [0.6, 0.7, 0.8],
            "xgb__colsample_bytree":[0.6, 0.7, 0.8],
        })


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
