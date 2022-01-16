import re
import xml.etree.ElementTree as ET

def parse_user_file(path):
    # parse file in path
    tree = ET.parse(path)
    root = tree.getroot()

    # get texts inside "./documents/document"
    texts = [doc.text for doc in root.findall('./documents/document')]
    
    # remove "RT #USER#: " from the beginning of each text
    texts = [re.sub(r'^RT #USER#: ', '##RT## ', text) for text in texts]

    return texts

def parse_labels_file(path):
    # open file in path
    with open(path, 'r') as f:
        # read all lines
        lines = f.readlines()

    # split at ':::'
    lines = [line.split(':::') for line in lines]
    lines = [(user, int(label)) for user, label in lines]

    return lines

def read_dataset(data_dir, ground_truth='truth.txt'):
    # get gold values from data_dir/ground_truth
    labels = parse_labels_file(data_dir + '/' + ground_truth)

    # define data: {label: [tweets]}
    data = {}

    # for each labels
    for user, label in labels:
        # get user's tweets from data_dir/user.xml
        tweets = parse_user_file(data_dir + '/' + str(user) + '.xml')

        # append tweets to data 
        if label not in data:
            data[label] = []
        data[label].append(tweets)

    return data