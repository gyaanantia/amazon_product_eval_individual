import pandas as pd
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import datetime
from nltk.downloader import download
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np


def extract(x):
    return round(np.single(x['label'][0]) * 0.2, 1)


download('vader_lexicon')
config = AutoConfig.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
config.max_length = 100000
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
device = torch.device("mps")
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, config=config, device=device)
reviews_training = pd.read_json("./source/Toys_and_Games/train/review_training.json")

rev_list = list(reviews_training['reviewText'].apply(lambda x: x[:512] if x is not None else "None"))
none_list = reviews_training[reviews_training['reviewText'].isnull()].index.to_list()

start = datetime.datetime.now()
print("Starting")
pred_list = classifier.predict(rev_list[:100])
scores_list = list(map(extract, pred_list))
end = datetime.datetime.now()
print(end - start)
print(scores_list[:30])
print(pred_list[0]['label'][0])

# analyzer = SentimentIntensityAnalyzer()
# start = datetime.datetime.now()
# vader_list = reviews_training['reviewText'].apply(lambda x: analyzer.polarity_scores(x)['compound'] if x is not None else None)
# end = datetime.datetime.now()
# print(end - start)
# print(vader_list)
