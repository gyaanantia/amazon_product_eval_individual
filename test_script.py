import pandas as pd

import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence


def extract(x):
    return round(np.single(x['label'][0]) * 0.2, 1)


reviews_training = pd.read_json("./source/Toys_and_Games/train/review_training.json")
sia = TextClassifier.load('en-sentiment')
rev_list = list(reviews_training['reviewText'].apply(lambda x: x[:512] if x is not None else "None"))
none_list = reviews_training[reviews_training['reviewText'].isnull()].index.to_list()

start = datetime.datetime.now()
print("Starting")

end = datetime.datetime.now()
print(end - start)


analyzer = SentimentIntensityAnalyzer()
start = datetime.datetime.now()
vader_list = reviews_training['reviewText'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if x is not None else None)
end = datetime.datetime.now()
print(end - start)
print(vader_list)
