import re


def count_hashtags(data):
    # given data: {label: [[tweets]]
    # get total count of #HASHTAG# for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += sum([len(re.findall('#HASHTAG#', tweet)) for tweet in user])
    return out
       
def count_urls(data):
    # given data: {label: [[tweets]]
    # get total count of #URL# for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += sum([len(re.findall('#URL#', tweet)) for tweet in user])
    return out

def count_users(data):
    # given data: {label: [[tweets]]
    # get total count of #USER# for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += sum([len(re.findall('#USER#', tweet)) for tweet in user])
    return out

def count_rt(data):
    # given data: {label: [[tweets]]
    # get total count of ##RT## for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += sum([len(re.findall('##RT##', tweet)) for tweet in user])
    return out  



def count_uppercase_chars(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT## 
    # get total count of uppercase chars for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += sum([len(re.findall(r'[A-Z]', re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet))) for tweet in user])
    return out

def count_min_chars(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get minimum number of characters for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 99999
        for user in users:
            out[label] = min(min([len(re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip()) for tweet in user]), out[label])
    return out

def count_max_chars(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get maximum number of characters for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] = max(max([len(re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip()) for tweet in user]), out[label])
    return out

def count_avg_min_chars(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get average of minimum number of characters for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += min([len(re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip()) for tweet in user])
        out[label] /= len(users)
    return out

def count_avg_max_chars(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get average of maximum number of characters for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += max([len(re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip()) for tweet in user])
        out[label] /= len(users)
    return out

def count_chars(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get total number of characters for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += sum([len(re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip()) for tweet in user])
        out[label]
    return out

def count_uppercase_words(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get total count of uppercase words for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += sum([len(re.findall(r'[A-Z]\w+', re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet))) for tweet in user])
    return out

def count_min_words(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get minimum number of words for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 99999
        for user in users:
            out[label] = min(min([len(re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip().split()) for tweet in user]), out[label])
    return out

def count_avg_min_words(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get average of minimum number of words for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += min([len(re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip().split()) for tweet in user])
        out[label] /= len(users)
    return out

def count_max_words(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get maximum number of words for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] = max(max([len(re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip().split()) for tweet in user]), out[label])
    return out

def count_avg_max_words(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get average of maximum number of words for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += max([len(re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip().split()) for tweet in user])
        out[label] /= len(users)
    return out

def count_words(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get total number of words for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += sum([len(re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip().split()) for tweet in user])
    return out

## @Source:: https://stackoverflow.com/questions/367155/splitting-a-string-into-words-and-punctuation

def count_stopwords(data):
    # given data: {label: [[tweets]]
    # remove #HASHTAG#, #URL#, #USER#, ##RT##
    # get total number of stopwords for each label
    # output out: {label: count}
    from spacy.lang.en.stop_words import STOP_WORDS
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += sum([len([t for t in list(re.findall(r"[\w']+|[.,!?;]", re.sub(r'#(HASHTAG|URL|USER|#RT#)#', '', tweet).strip().lower())) if t in STOP_WORDS]) for tweet in user])
    return out

def count_emojis(data):
    # given data: {label: [[tweets]]
    # get total number of emojis for each label
    # output out: {label: count}
    # @TODO:: include emojis with just characters-- like :) and <3...
    import demoji
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += sum([len(demoji.findall_list(tweet, desc=False)) for tweet in user])
    return out


def count_min_words_alt(data):
    # given data: {label: [[tweets]]
    # remove ##RT##
    # get minimum number of words for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 99999
        for user in users:
            out[label] = min(min([len(re.sub(r'^##RT##', '', tweet).strip().split()) for tweet in user]), out[label])
    return out

def count_avg_min_words_alt(data):
    # given data: {label: [[tweets]]
    # remove ##RT##
    # get average of minimum number of words for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += min([len(re.sub(r'^##RT##', '', tweet).strip().split()) for tweet in user])
        out[label] /= len(users)
    return out

def count_max_words_alt(data):
    # given data: {label: [[tweets]]
    # remove ##RT##
    # get maximum number of words for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] = max(max([len(re.sub(r'^##RT##', '', tweet).strip().split()) for tweet in user]), out[label])
    return out

def count_avg_max_words_alt(data):
    # given data: {label: [[tweets]]
    # remove ##RT##
    # get average of maximum number of words for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += max([len(re.sub(r'^##RT##', '', tweet).strip().split()) for tweet in user])
        out[label] /= len(users)
    return out

def count_words_alt(data):
    # given data: {label: [[tweets]]
    # remove ##RT##
    # get total number of words for each label
    # output out: {label: count}
    out = {}
    for label, users in data.items():
        out[label] = 0
        for user in users:
            out[label] += sum([len(re.sub(r'^##RT##', '', tweet).strip().split()) for tweet in user])
    return out


def get_sentiments(data):
    # given data: {label: [[tweets]]
    # get sentiment for each label
    # output out: {label: sentiment}
    from textblob import TextBlob
    out = {"positive": {}, "negative": {}, "neutral": {}}
    for label, users in data.items():
        out["positive"][label], out["negative"][label], out["neutral"][label] = 0, 0, 0
        for user in users:
            for tweet in user:
                sentiment = TextBlob(tweet).sentiment.polarity
                if sentiment > 0:
                    out['positive'][label] += 1
                elif sentiment < 0:
                    out['negative'][label] += 1
                else:
                    out['neutral'][label] += 1
    return out

def get_named_entities(data, corpora='en_core_web_sm'):
    # given data: {label: [[tweets]]
    # get named entities for each label
    # output out: {label: count}
    assert corpora in ['en_core_web_sm', 'en_core_web_md', 'xx_ent_wiki_sm']

    import spacy
    import tqdm
    nlp = spacy.load(corpora)
    out = {'PERSON': {}, 'PER': {}, 'ORG': {}, 'GPE': {}, 'LOC': {}, 'MISC': {}}
    unk = set([])
    for label, users in data.items():
        for ner in out.keys():
            out[ner][label] = 0
        for user in tqdm.tqdm(users):
            for tweet in user:
                doc = nlp(tweet)
                for ent in doc.ents:
                    if ent.label_  not in out.keys():
                        unk.add(ent.label_)
                        ent.label_ = 'MISC'
                    out[ent.label_][label] += 1
    print(unk)
    return out