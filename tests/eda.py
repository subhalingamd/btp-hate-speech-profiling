from eda import get_stats
from preprocessing import pan21

def test_count_hashtags():
    out = get_stats.count_hashtags(pan21.read_dataset('tests/resources/data/pan21'))
    assert out == {0: 3, 1: 5}

def test_count_urls():
    out = get_stats.count_urls(pan21.read_dataset('tests/resources/data/pan21'))
    assert out == {0: 1, 1: 2}

def test_count_users():
    out = get_stats.count_users(pan21.read_dataset('tests/resources/data/pan21'))
    assert out == {0: 3, 1: 4}

def test_count_rt():
    out = get_stats.count_rt(pan21.read_dataset('tests/resources/data/pan21'))
    assert out == {0: 1, 1: 1}

def test_count_uppercase_chars():
    out = get_stats.count_uppercase_chars(pan21.read_dataset('tests/resources/data/pan21'))
    assert out == {0: 6, 1: 13}

def test_count_uppercase_words():
    out = get_stats.count_uppercase_words(pan21.read_dataset('tests/resources/data/pan21'))
    assert out == {0: 5, 1: 6}

def test_count_min_chars():
    out = get_stats.count_min_chars(pan21.read_dataset('tests/resources/data/pan21'))
    assert out == {0: 7, 1: 0}

def test_count_max_chars():
    out = get_stats.count_max_chars(pan21.read_dataset('tests/resources/data/pan21'))
    assert out == {0: 12, 1: 12}

def test_count_avg_min_chars():
    out = get_stats.count_avg_min_chars(pan21.read_dataset('tests/resources/data/pan21'))
    assert out == {0: 7, 1: 3.5}

def test_count_avg_max_chars():
    out = get_stats.count_avg_max_chars(pan21.read_dataset('tests/resources/data/pan21'))
    assert out == {0: 12, 1: 8}

def test_count_chars():
    out = get_stats.count_chars(pan21.read_dataset('tests/resources/data/pan21'))
    assert out == {0: 39, 1: 45}

def test_count_stopwords():
    out = get_stats.count_stopwords({0: [['and', 'apple']], 1: [['HE sucks!!'], ['Hate her...']]})
    assert out == {0: 1, 1: 2}

def test_count_emojis():
    out = get_stats.count_emojis({0: [["🙃", "😃"], ["🍗 <3 :)) 🙃🙃..."]]})
    assert out == {0: 5} # @TODO:: {0: 7}

def test_get_sentiments():
    out = get_stats.get_sentiments({0: [['happy', 'nothing'], ['sad', 'nothing']], 1: [['crying']]})
    assert out == {"positive": {0: 1, 1: 0}, "negative": {0: 1, 1: 1}, "neutral": {0: 2, 1: 0}}

def test_get_named_entities__core_sm():
    out = get_stats.get_named_entities({0: [['New Delhi is the capital of India.', 'The Burj Khalifa is a mixed-use skyscraper located in Dubai']], 1: [['Steve Jobs is the founder of Apple']]}, corpora='en_core_web_sm')
    print(out)
    assert out == {'PERSON': {0: 1, 1: 1}, 'PER': {0: 0, 1: 0}, 'ORG': {0: 0, 1: 1}, 'GPE': {0: 3, 1: 0}, 'LOC': {0: 0, 1: 0} , 'MISC': {0: 0, 1: 0}}

def test_get_named_entities__wiki_sm():
    out = get_stats.get_named_entities({0: [['New Delhi is the capital of India.', 'The Burj Khalifa is a mixed-use skyscraper located in Dubai']], 1: [['Steve Jobs is the founder of Apple']]}, corpora='xx_ent_wiki_sm')
    print(out)
    assert out == {'PERSON': {0: 0, 1: 0}, 'PER': {0: 0, 1: 1}, 'ORG': {0: 0, 1: 1}, 'GPE': {0: 0, 1: 0}, 'LOC': {0: 4, 1: 0} , 'MISC': {0: 0, 1: 0}}
   