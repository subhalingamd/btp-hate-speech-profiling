from preprocessing import pan21

def test_parse_user_file():
    out = pan21.parse_user_file('tests/resources/data/pan21/user1.xml')
    assert out == ['Tweet 1', '#USER# Tweet 2 #HASHTAG# #HASHTAG#', '#USER#: Tweet 3, #HASHTAG# #URL#', 'Tweet 4 RT #USER#: ']

def test_parse_labels_file():
    out = pan21.parse_labels_file('tests/resources/data/pan21/labels.txt')
    assert out == [('user1',0), ('user2',1), ('user3', 1)]

def test_read_dataset():
    out = pan21.read_dataset('tests/resources/data/pan21')
    assert out == {0: [['Tweet 1', '#USER# Tweet 2 #HASHTAG# #HASHTAG#', '#USER#: Tweet 3, #HASHTAG# #URL#', 'Tweet 4 RT #USER#: ']], 1: [['Tweet 1', '#USER# Tweet 2 #HASHTAG# #HASHTAG#', '#USER#: Tweet 3, #HASHTAG# #URL#', 'Tweet 4 RT #USER#: '], ['Tweet from #USER#']]}

    