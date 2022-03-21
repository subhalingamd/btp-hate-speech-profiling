import pandas as pd
import time
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

_label_col = 'HS'
_tweet_col = 'text'
_cols = [_label_col, _tweet_col]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('#USER#') and len(t) > 1 else t
        t = '#hashtag' if t.startswith('#HASHTAG#') and len(t) > 1 else t
        t = 'http' if t.startswith('#URL#') else t
        t = 'RT @user' if t.startswith('##RT##') else t
        new_text.append(t)
    return " ".join(new_text)


def train(*args, **kwargs):
    """
    Train a model on a given data set.
    """
    raise NotImplementedError('Not implemented yet.')

def test(data, model_name="cardiffnlp/twitter-roberta-base-hate", store_scores_to=None, store_predictions_to=None, include_all_params=True, return_proba=False):
    """
    Test a model on a given data set.
    """
    print('**Using model {} and data from {}'.format(model_name, data))

    data = pd.read_csv(data, delimiter='\t', usecols=_cols)
    
    MODEL = model_name

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)

    def predict(text):
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt').to(device)
        output = model(**encoded_input)
        # scores = output[0][0].detach().numpy()
        # scores = softmax(scores)
        output = output.logits.argmax(dim=-1).tolist()[0]
        return output
    predictions = [predict(t) for t in tqdm(data[_tweet_col].tolist())]
    
    # print(classification_report(data[_label_col], predictions))

    accuracy = accuracy_score(data[_label_col], predictions)
    precision = precision_score(data[_label_col], predictions)
    recall = recall_score(data[_label_col], predictions)
    f1 = f1_score(data[_label_col], predictions)

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
                'model': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }, ignore_index=True)

            df.to_csv(store_scores_to, sep='\t', index=False)

    if store_predictions_to is not None:
        try:
            df = pd.read_csv(store_predictions_to, sep='\t')
            df[model_name] = predictions
        except FileNotFoundError:
            df = pd.DataFrame()
            df['gold'] = data[_label_col]
            df[model_name] = predictions
        except Exception as e:
            raise e
        finally:
            df.to_csv(store_predictions_to, sep='\t', index=False)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }
