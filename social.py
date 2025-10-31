import social_tests as test

### PHASE 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def parse_label(label):
    name = label[label.find(" "):label.find("(")].strip()
    position = label[label.find("(")+1:label.find("from")].strip()
    state = label[label.find("from")+len("from"):label.find(")")].strip()
    details = {
        "name" : name,
        "position" : position,
        "state" : state
    }
    return details 

def get_region_from_state(state_df, state):
    row = state_df[state_df["state"] == state]
    return row.iloc[0]['region']

end_chars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]
def find_hashtags(message):
    hashtag_list = []
    for i in range(len(message)):
        if message[i] == '#':
            j = i + 1
            while j < len(message) and message[j] not in end_chars:
                j = j + 1
            hashtag_list.append(message[i:j])
    return hashtag_list

def find_sentiment(classifier, message):
    score_dict = classifier.polarity_scores(message)
    score = score_dict['compound']
    if score > 0.1:
        result = "positive"
    elif score < -0.1:
        result = "negative"
    else:
        result = "neutral"
    return (score,result)

def add_columns(data, state_df):
    classifier = SentimentIntensityAnalyzer()
    names, positions, states, regions = [], [], [], []
    for label in data['label']:
        details = parse_label(label)
        names.append(details["name"])
        positions.append(details["position"])
        state = details["state"]
        states.append(state)
        regions.append(get_region_from_state(state_df, state))
    
    hashtags, scores, sentiments = [], [], []
    for text in data['text']:
        hashtags.append(find_hashtags(text))
        score, sentiment = find_sentiment(classifier, text)
        scores.append(score)
        sentiments.append(sentiment)
        
    data["name"] = names
    data["position"] = positions
    data["state"] = states
    data["region"] = regions
    data["hashtags"] = hashtags
    data["score"] = scores
    data["sentiment"] = sentiments
    return

### PHASE 2 ###

def get_sentiment_quantiles(data, col_name, col_value):
    if col_name != "":
        data = data[data[col_name] == col_value]
    result = [data["score"].min()]
    result.extend(list(round(data["score"].quantile([0.25, 0.5, 0.75]), 5)))
    result.append(data["score"].max())
    return result

def get_hashtag_subset(data, col_name, col_value):
    if col_name != "":
        data = data[data[col_name] == col_value]
    all_hashtags = set()
    for hashtags in data["hashtags"]:
        for tag in hashtags:
            all_hashtags.add(tag)
    return all_hashtags

def get_hashtag_rates(data):
    d = dict()
    for hashtags in data["hashtags"]:
        for tag in hashtags:
            if tag not in d:
                d[tag] = 0
            d[tag] += 1
    return d

def most_common_hashtags(hashtags, count):
    top_tags = dict()
    while len(top_tags) < count:
        current_top = None
        current_count = 0
        for key in hashtags:
            if hashtags[key] > current_count and key not in top_tags:
                current_top = key
                current_count = hashtags[key]
        top_tags[current_top] = current_count
    return top_tags

def get_hashtag_sentiment(data, hashtag):
    total = 0
    count = 0
    for index, row in data.iterrows():
        hashtags = row["hashtags"]
        sentiment = row["sentiment"]
        if hashtag in hashtags:
            count += 1
            if sentiment == "positive":
                total += 1
            elif sentiment == "negative":
                total += -1
    return total / count


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    test.test_all()
    test.run()