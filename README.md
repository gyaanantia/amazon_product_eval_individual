# CS349 Machine Learning - AmazonProductEvaluation
## Group Project - Final Model Preditcions ##

Individual portion for gfa2226

Team members: zni4816, pqb0185, gfa2226

Dataset used: Toys and Games

Dataset can be found [here](https://urldefense.com/v3/__https:/drive.google.com/file/d/16lrMrD3w0bnr_rzqEC7qwF2dRv2ulybv/view?usp=share_link__;!!Dq0X2DkFhyF93HkjWTBQKhk!RDEdHaAs_Vesk88fGJJBFe2xssf3I-qSUH-KVFXN-avAS4hM8M27VyhgOQJR14yUHO7Jh56gZ6DfdcIt9qOIEb69zBLFBLwL5Qg$)

Download Dataset and copy folder Toys_and_Games/ into source/ to be able to read the dataset.

Final Model Predictions on dataset ```test3```.

Python Version used: ```Python 3.9.6```

Running ```main.py``` generates two sets of feature vectors, one trained on TextBlob Sentiment Analysis and the other trained on VADER Sentiment Analysis. It then trains a Decision Tree, Logistic Regression, Random Forest, K Nearest Neighbors and Multi-Layer Perceptron models for each set of feature vectors. It also trains a Voting Classifier for Late Fusion and an AdaBoost Classifier, and then uses the AdaBoost classifier trained on the VADER set to make predictions. All models metrics are written to the `scores/` directory. 

NOTE: In the interest of time, SVM models are not trained. If you would like to see the scores for SVM models, please uncomment lines 577 and 578 in `main.py`