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

