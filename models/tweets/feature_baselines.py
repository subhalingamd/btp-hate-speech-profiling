import pandas as pd
import pickle
import time
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

_label_col = 'HS'
_cols = [_label_col, 
        'hashtags', 'urls', 'users', 'rt',
        'uppercase_chars', 'chars', 'uppercase_words', 'words',
        'stopwords', 'emojis',
        'sentiment_negative', 'sentiment_neutral', 'sentiment_positive', # 'sentiment',
        'NER_PER', 'NER_ORG', 'NER_LOC', 'NER_MISC',]

def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def train(data, cls='lr', dump_objects_to=None, store_params_to=None):
    """
    Train a model on a given data set for a given representation and classifier.
    """
    cls = cls.lower()
    ALLOWED_CLS = ['lr', 'svm', 'nb', 'rf',]
    assert cls in ALLOWED_CLS, 'Classifier must be from {}'.format(str(ALLOWED_CLS))

    print('**Training model with {} classifier**'.format(cls))    

    df = pd.read_csv(data, delimiter='\t', usecols=_cols)

    pipeline, parameters = [], {}
    
    if cls == 'lr':
        pipeline.append(('lr', LogisticRegression(
                                    verbose=0,
                                    max_iter=10000,
                                    random_state=42,
                                    )
                        ))


        parameters.update({
            'lr__C': [.1, .2, .5, 1, 2, 5, 10, 20, 50, 100],
            'lr__penalty': ['l1', 'l2'],
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
            'svm__C': [.1, .2, .5, 1, 2, 5, 10, 20, 50, 100],
            'svm__kernel': ['linear', 'rbf'], # ['rbf', 'linear', 'poly', 'sigmoid']
            # 'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
        })

    elif cls == 'nb':
        pipeline.append(('nb', MultinomialNB(
                                    )
                        ))

        parameters.update({
            'nb__alpha': np.logspace(-5, 1, num=25),
        })
    
    elif cls == 'rf':
        pipeline.append(('rf', RandomForestClassifier(
                                    verbose=0,
                                    random_state=42,
                                    )
                        ))

        parameters.update({
            'rf__n_estimators': [20, 50, 100, 200, 500], # try -> [50, 100, 200, 500]
            'rf__max_depth': [None, 5, 10, 50, 100],
            'rf__min_samples_leaf':  [1, 2, 4],
            'rf__max_features': ['auto'], # ['auto', 'sqrt', 'log2', None]
            'rf__bootstrap': [True], # [True, False]
            'rf__oob_score': [True], # [True, False]
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
    params_grid = grid_search.fit(
        df[[f for f in df.columns if f not in [_label_col, 'id']]], 
        df[_label_col]
        )
    print('End time:\t',time.strftime("%m/%d/%Y %H:%M:%S"))

    if dump_objects_to is not None:
        save_pickle(params_grid, dump_objects_to)

    if store_params_to is not None:
        df = pd.DataFrame()
        df['params'] = params_grid.cv_results_['params']
        df['train_scores'] = params_grid.cv_results_['mean_train_score']
        df['test_scores'] = params_grid.cv_results_['mean_test_score']
        df.to_csv(store_params_to, sep='\t', index=False)


def test(data, model_path, store_scores_to=None, store_predictions_to=None, include_all_params=True):
    """
    Test a model on a given data set.
    """
    print('**Using model from {}'.format(model_path.split('/')[-1]))

    data = pd.read_csv(data, delimiter='\t', usecols=_cols)
    model = load_pickle(model_path)

    predictions = model.predict(
        data[[f for f in data.columns if f not in [_label_col, 'id']]],
    )

    accuracy = accuracy_score(data[_label_col], predictions)
    precision = precision_score(data[_label_col], predictions)
    recall = recall_score(data[_label_col], predictions)
    f1 = f1_score(data[_label_col], predictions)
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
                columns=['model', 'accuracy', 'precision', 'recall', 'f1', 'best_params'] + (['all_params'] if include_all_params else [])
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
                'best_params': str(best_model_params),
            }, ignore_index=True)

            if include_all_params:
                x = pd.DataFrame(model.cv_results_["params"])
                df.iloc[-1, df.columns.get_loc('all_params')] = str({
                                            col: x[col].unique().tolist()
                                                for col in x.columns
                                        })


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
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'params': best_model_params,
    }
