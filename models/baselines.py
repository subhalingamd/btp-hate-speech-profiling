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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

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


def test(data, model_path, store_scores_to=None, store_predictions_to=None):
    print('**Using model from {}'.format(model_path.split('/')[-1]))

    data = pd.read_csv(data, delimiter='\t')
    model = load_pickle(model_path)

    predictions = model.predict(data['tweets'])

    accuracy = accuracy_score(data['label'], predictions)
    precision = precision_score(data['label'], predictions)
    recall = recall_score(data['label'], predictions)
    f1 = f1_score(data['label'], predictions)
    best_model_params = model.best_params_

    print('Accuracy:\t{:.4f}'.format(accuracy))
    print('Precision:\t{:.4f}'.format(precision))
    print('Recall:\t\t{:.4f}'.format(recall))
    print('F1:\t\t{:.4f}'.format(f1))

    if store_scores_to is not None:
        try:
            df = pd.read_csv(store_scores_to, sep='\t')
        except FileNotFoundError:
            df = pd.DataFrame(
                columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'params']
                )
        except Exception as e:
            raise e
        finally:
            df = df.append({
                'model': model_path.split('/')[-1],
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'params': best_model_params,
            }, ignore_index=True)
            df.to_csv(store_scores_to, sep='\t', index=False)

    if store_predictions_to is not None:
        try:
            df = pd.read_csv(store_predictions_to, sep='\t')
            df[model_path.split('/')[-1]] = predictions
        except FileNotFoundError:
            df = pd.DataFrame()
            df['predictions'] = predictions
        except Exception as e:
            raise e
        finally:
            df.to_csv(store_predictions_to, sep='\t', index=False)