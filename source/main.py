import datetime
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from flair.models import TextClassifier
from flair.data import Sentence

# for the flair helper
sia = TextClassifier.load('en-sentiment')


### Helper Functions ###
def run_vader(analyzer, vader_df, start_time):
    vader_df['vaderTextScore'] = vader_df['reviewText'].apply(
        lambda x: analyzer.polarity_scores(x)['compound'] if x is not None else None)
    text_done_time = datetime.datetime.now()
    print("Done with text in ", text_done_time, start_time)
    vader_df['vaderSummScore'] = vader_df['summary'].apply(
        lambda x: analyzer.polarity_scores(x)['compound'] if x is not None else None)
    return vader_df


def flair_prediction(x):
    sentence = Sentence(x)
    sia.predict(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        return 1
    elif "NEGATIVE" in str(score):
        return -1
    else:
        return 0


def run_flair(flair_df, start_time):
    flair_df['flairTextScore'] = flair_df['reviewText'].apply(lambda x: flair_prediction(x) if x is not None else None)
    text_done_time = datetime.datetime.now()
    print("Done with text in ", text_done_time, start_time)
    flair_df['flairSummScore'] = flair_df['summary'].apply(lambda x: flair_prediction(x) if x is not None else None)
    return flair_df


def get_age_weight(rev_training, index, first_rev_time):
    return ((0.01) / (365 * 24 * 3600)) * (rev_training.unixReviewTime[index] - first_rev_time) + 1


def get_review_age(index, rev_training):
    return rev_training.unixReviewTime[index]


def get_vote_weight(rev_training, index, max_votes):
    return 1 + (0.2 * (np.log(get_vote(index, rev_training) + 1) / np.log(max_votes + 1.1)))


def get_vote(index, rev_training):
    if rev_training.vote[index] is None:
        return 0
    else:
        return int(rev_training.vote[index].replace(",", ""))


def get_image_weight(rev_training, index):
    return 1.3 if rev_training.image[index] is not None else 1


def get_verification_weight(rev_training, index):
    return 1.5 if rev_training.verified[index] else 1.0


def get_avg_weight(compound_list, weight_list):
    weight_compound_list = []
    for i in range(len(compound_list)):
        weight_prod = 1
        for j in range(4):
            weight_prod *= weight_list[j][i]
        if compound_list[i] is not None:
            weight_compound_list.append(compound_list[i] * weight_prod * 10)
    return round(np.sum(weight_compound_list) / len(weight_compound_list), 2) if len(weight_compound_list) != 0 else 0


def get_std(compound_list):
    comp_list = np.array(compound_list)
    comp_list = comp_list[comp_list != None]
    if len(comp_list) > 0:
        return round(np.std(comp_list), 4)
    return 0


def generate_bar_plot(models, model_names, scoring_method, scores_avg, w=0.2, bottom=0.6, top=1):
    x = []
    for k in range(len(models)):
        x.append(model_names[k].replace("model_", ""))

    x_axis = np.arange(len(x))
    offset = -(((len(scoring_method) - 1) * w) / 2)

    method_scores = {}

    for method in scoring_method:
        method_scores[method] = []

    for model in scores_avg:
        for method in scoring_method:
            method_scores[method].append(scores_avg[model][method])

    count = 0
    colors = ['lightseagreen', 'mediumorchid', 'steelblue']
    for method in scoring_method:
        plt.bar(x_axis + offset + (count * w),
                method_scores[method], w, color=colors[count], label=method)
        count += 1
    plt.ylim(bottom, top)
    plt.xticks(x_axis, x)
    plt.xlabel("Models")
    plt.ylabel("Performance")
    plt.title("Scores for each model")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("./plots/bar_plot.png")
    # plt.show()


### Preprocessing ###
def preprocess(associations, rev_df):
    # index texts, summary, get avg star rating
    product_text_list = []
    vader_product_text_list = []
    flair_product_text_list = []
    product_summ_list = []
    vader_product_summ_list = []
    flair_product_summ_list = []
    feature_stars_list = []
    for asin in associations.keys():
        text_list = []
        vader_text_list = []
        flair_text_list = []
        summ_list = []
        vader_summ_list = []
        flair_summ_list = []
        stars_list = []
        star_summary = ["One Star", "Two Stars",
                        "Three Stars", "Four Stars", "Five Stars"]
        for index in associations[asin]:
            text_list.append(rev_df.reviewText[index])
            vader_text_list.append(rev_df.vaderTextScore[index])
            vader_summ_list.append(rev_df.vaderSummScore[index])
            flair_text_list.append(rev_df.flairTextScore[index])
            flair_summ_list.append(rev_df.flairSummScore[index])

            if rev_df.summary[index] in star_summary:
                summ_list.append(None)
                for i in range(len(star_summary)):
                    if star_summary[i] == rev_df.summary[index]:
                        stars_list.append(i + 1)
                        break
            else:
                summ_list.append(rev_df.summary[index])
        product_text_list.append(text_list)
        vader_product_text_list.append(vader_text_list)
        flair_product_text_list.append(flair_text_list)
        product_summ_list.append(summ_list)
        vader_product_summ_list.append(vader_summ_list)
        flair_product_summ_list.append(flair_summ_list)
        feature_stars_list.append(
            round(np.mean(stars_list), 1) if stars_list != [] else 1)

    #  Age, Vote, Verification, Image weight, Amount of Reviews, Verification Percentage
    print("Compute remaining features")
    product_age_weight = []
    product_vote_weight = []
    product_verification_weight = []
    product_image_weight = []
    feature_num_rev_list = []
    feature_verification_perc_list = []
    for asin in associations.keys():
        min_age = float("inf")
        age_weight = []
        max_votes = float("-inf")
        vote_weight = []
        verification_weight = []
        image_weight = []
        feature_num_rev_list.append(len(associations[asin]))
        verification_count = 0
        for index in associations[asin]:
            age = get_review_age(index, rev_df)
            vote = get_vote(index, rev_df)
            if age < min_age:
                min_age = age
            if vote > max_votes:
                max_votes = vote

        for index in associations[asin]:
            age_weight.append(get_age_weight(rev_df, index, min_age))
            vote_weight.append(get_vote_weight(rev_df, index, max_votes))
            verification_weight.append(get_verification_weight(rev_df, index))
            image_weight.append(get_image_weight(rev_df, index))
            verification_count += rev_df.verified[index]

        product_age_weight.append(age_weight)
        product_vote_weight.append(vote_weight)
        product_verification_weight.append(verification_weight)
        product_image_weight.append(image_weight)
        feature_verification_perc_list.append(
            round(verification_count / len(associations[asin]), 2))

    # weighted compound
    feature_avg_vader_text_list = []
    feature_avg_vader_summ_list = []
    feature_std_vader_text_list = []
    feature_std_vader_summ_list = []

    # weighted sentiment
    feature_avg_flair_text_list = []
    feature_avg_flair_summ_list = []
    feature_std_flair_text_list = []
    feature_std_flair_summ_list = []
    for i in range(len(associations)):
        product_weights = [product_age_weight[i], product_vote_weight[i], product_verification_weight[i],
                           product_image_weight[i]]
        feature_avg_vader_text_list.append(get_avg_weight(
            vader_product_text_list[i], product_weights))
        feature_avg_vader_summ_list.append(get_avg_weight(
            vader_product_summ_list[i], product_weights))
        feature_std_vader_text_list.append(get_std(vader_product_text_list[i]))
        feature_std_vader_summ_list.append(get_std(vader_product_summ_list[i]))

        feature_avg_flair_text_list.append(get_avg_weight(
            flair_product_text_list[i], product_weights))
        feature_avg_flair_summ_list.append(get_avg_weight(
            flair_product_summ_list[i], product_weights))
        feature_std_flair_text_list.append(get_std(flair_product_text_list[i]))
        feature_std_flair_summ_list.append(get_std(flair_product_summ_list[i]))

    vader_features = {"avg_text": feature_avg_vader_text_list,
                      "avg_summ": feature_avg_vader_summ_list,
                      "std_text": feature_std_vader_text_list,
                      "std_summ": feature_std_vader_summ_list,
                      "pct_verif": feature_verification_perc_list,
                      "amt_reviews": feature_num_rev_list,
                      "amt_stars": feature_stars_list}

    flair_features = {"avg_text": feature_avg_flair_text_list,
                      "avg_summ": feature_avg_flair_summ_list,
                      "std_text": feature_std_flair_text_list,
                      "std_summ": feature_std_flair_summ_list,
                      "pct_verif": feature_verification_perc_list,
                      "amt_reviews": feature_num_rev_list,
                      "amt_stars": feature_stars_list}

    return vader_features, flair_features
    # return features


def add_awesomeness(associations, features, awesomeness_training):
    class_awesomeness = []
    for asin in associations.keys():
        class_awesomeness.append(
            awesomeness_training[awesomeness_training.asin == asin].awesomeness.values[0])

    features["awesomeness"] = class_awesomeness
    return features


def generate_feature_vectors():
    print('Reading json files for training and testing')
    training_rev_file = "Toys_and_Games/train/review_training.json"
    training_prod_file = "Toys_and_Games/train/product_training.json"
    testing_rev_file = "Toys_and_Games/test3/review_test.json"

    reviews_training = pd.read_json(training_rev_file)
    awesomeness_training = pd.read_json(training_prod_file)
    reviews_test = pd.read_json(testing_rev_file)

    # setup for vader sentiment
    analyzer = SentimentIntensityAnalyzer()
    print("Running VADER Sentiment Analysis on training data...")
    start_vader_training_time = datetime.datetime.now()
    vadered_training = run_vader(analyzer, reviews_training, start_vader_training_time)
    end_vader_training_time = datetime.datetime.now()
    print("Done in ", end_vader_training_time - start_vader_training_time)

    print("Running VADER Sentiment Analysis on test data...")
    start_vader_test_time = datetime.datetime.now()
    vadered_test = run_vader(analyzer, reviews_test, start_vader_test_time)
    end_vader_test_time = datetime.datetime.now()
    print("Done in ", end_vader_test_time - start_vader_test_time)

    print("Running Flair Sentiment Analysis on training data...")
    start_flair_training_time = datetime.datetime.now()
    flaired_training = run_flair(vadered_training, start_flair_training_time)
    end_flair_training_time = datetime.datetime.now()
    print("Done in ", end_flair_training_time - start_flair_training_time)

    print("Running Flair Sentiment Analysis on testing data...")
    start_flair_test_time = datetime.datetime.now()
    flaired_test = run_flair(vadered_test, start_flair_test_time)
    end_flair_test_time = datetime.datetime.now()
    print("Done in ", end_flair_test_time - start_flair_test_time)

    print("Running associations on training")
    associations_training = reviews_training.groupby(
        'asin').apply(lambda x: x.index.tolist())

    print("Running associations on test")
    associations_test = reviews_test.groupby(
        'asin').apply(lambda x: x.index.tolist())

    # associations_training = pd.read_json('associations_training.json', typ='series')
    # associations_test = pd.read_json('associations_test.json', typ='series')

    # Preprocessing may run up to 30min (recent Mac M2 Pro)
    print("Preprocessing training")
    features_vader_training, features_flair_training = preprocess(associations_training, flaired_training)  # uncomment to run
    print("Adding ground truth")
    awesomeness = add_awesomeness(associations_training, features_vader_training, awesomeness_training)
    awesomeness_sentiment = add_awesomeness(associations_training, features_flair_training, awesomeness_training)

    df_vader_training = pd.DataFrame(awesomeness, index=list(associations_training.keys()))
    df_vader_training.to_json("./features_vader_training.json")

    df_flair_training = pd.DataFrame(awesomeness_sentiment, index=list(associations_training.keys()))
    df_flair_training.to_json("./features_flair_training.json")

    print("Preprocessing test")
    features_vader_test, features_flair_test = preprocess(associations_test, flaired_test)

    df_vader_test = pd.DataFrame(features_vader_test, index=list(associations_test.keys()))
    df_vader_test.to_json("./features_vader_test3.json")

    df_flair_test = pd.DataFrame(features_flair_test, index=list(associations_test.keys()))
    df_flair_test.to_json("./features_flair_test3.json")


def grid_search(X, Y):
    print("Defining models for grid search")
    model_dt = DecisionTreeClassifier()
    model_lr = LogisticRegression(n_jobs=-1)
    model_rf = RandomForestClassifier(n_jobs=-1)
    model_svm = SVC(verbose=True, max_iter=100000)
    model_knn = KNeighborsClassifier(n_jobs=-1)
    model_ab = AdaBoostClassifier(
        RandomForestClassifier(max_depth=8, n_estimators=100, n_jobs=-1, criterion="log_loss", class_weight=None))
    model_mlp = MLPClassifier()

    dt_params = {
        "criterion": ["gini", "entropy", "log_loss"],
        "splitter": ["best", "random"],
        "class_weight": [None, "balanced"]
    }

    lr_params = {"tol": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001],
                 "C": [0.01, 0.1, 1, 10, 100],
                 "class_weight": [None, "balanced"],
                 "max_iter": [1000]
                 }

    rf_params = {
        "criterion": ["gini", "entropy", "log_loss"],
        "n_estimators": [100, 200, 300, 500],
        "class_weight": [None, "balanced"],
        "max_depth": [1, 2, 4, 8, 16]
    }

    svm_params = {
        "C": [5, 10, 20],
        "kernel": ["rbf"],  # others kernels did not seem to converge
        "gamma": ["scale", "auto"],
        "tol": [1e-1, 1e-3, 1e-5]
    }

    knn_params = {
        "n_neighbors": [5, 10, 15, 20],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [10, 20, 30, 40]
    }
    # ab_params = {'n_estimators': [5, 10, 13, 15, 17, 20],
    #              'estimator__max_depth': [5, 7, 8, 10],
    #              'learning_rate': [0.6, 0.8, 1, 1.6, 2, 2.5]
    # }
    # ab_params = {'n_estimators': [12, 13, 14, 15],
    #              'estimator__max_depth': [4, 5, 6, 7, 8],
    #              'learning_rate': [1.9, 2, 2.1, 2.2]
    #              }
    ab_params = {'n_estimators': [13],
                 'estimator__max_depth': [5],
                 'learning_rate': [2.0725, 2.075, 2.0775]
                 }

    # mlp_params = {
    #     "hidden_layer_sizes": [(5, 2), (5, 5), (7, 3), (7, 5), (5, 3, 6), (7, 4, 6)],
    #     "alpha": [1e-5, 1e-6, 1e-7],
    #     "solver": ['lbfgs', 'adam', 'sgd'],
    #     "max_iter": [10000]
    # }

    mlp_params = {
        "hidden_layer_sizes": [(5, 5), (5, 5, 5), (5, 5, 5, 5), (6, 6), (7, 5), (4), (3, 2), (6), (6, 3)],
        "alpha": [1e-7, 1e-8, 1e-9],
        "solver": ['lbfgs', 'adam', 'sgd'],
        "max_iter": [10000]
    }

    dt_grid = GridSearchCV(model_dt, dt_params, cv=10, scoring="f1", verbose=1, n_jobs=-1)
    lr_grid = GridSearchCV(model_lr, lr_params, cv=10, scoring="f1", verbose=2, n_jobs=-1)
    rf_grid = GridSearchCV(model_rf, rf_params, cv=10, scoring="f1", verbose=2, n_jobs=-1)
    svm_grid = GridSearchCV(model_svm, svm_params, cv=10, scoring="f1", verbose=2, n_jobs=-1)
    knn_grid = GridSearchCV(model_knn, knn_params, cv=10, scoring="f1", verbose=2, n_jobs=-1)
    ab_grid = GridSearchCV(model_ab, ab_params, cv=10, scoring="f1", verbose=2, n_jobs=-1)
    mlp_grid = GridSearchCV(model_mlp, mlp_params, cv=10, scoring="f1", verbose=2, n_jobs=-1)

    dt_grid.fit(X, Y)
    lr_grid.fit(X, Y)
    rf_grid.fit(X, Y)
    svm_grid.fit(X, Y)
    knn_grid.fit(X, Y)
    ab_grid.fit(X, Y)
    mlp_grid.fit(X, Y)

    dt_best_params = dt_grid.best_params_
    lr_best_params = lr_grid.best_params_
    rf_best_params = rf_grid.best_params_
    svm_best_params = svm_grid.best_params_
    knn_best_params = knn_grid.best_params_
    ab_best_params = ab_grid.best_params_
    mlp_best_params = mlp_grid.best_params_

    print("Decision tree best parameters: ", dt_best_params)
    print("LogReg best parameters: ", lr_best_params)
    print("Random Forest best parameters: ", rf_best_params)
    print("SVM best parameters: ", svm_best_params)
    print("KNN best parameters: ", knn_best_params)
    print("AdaBoost best parameters: ", ab_best_params)
    print("MLPClassifier best parameters: ", mlp_best_params)

    # return dt_best_params, lr_best_params, rf_best_params, svm_best_params, knn_best_params, ab_best_params
    return mlp_best_params


def late_fuse(estimator_list, X, Y, low1, high1, low2, high2, step):
    model_vc = VotingClassifier(estimators=estimator_list, voting='soft')
    vc_params = {
        "weights": []
    }

    for alpha1 in range(low1, high1, step):
        for alpha2 in range(low2, high2 - alpha1, step):
            alpha3 = 100 - alpha2 - alpha1
            a1 = alpha1 / 100
            a2 = alpha2 / 100
            a3 = alpha3 / 100
            vc_params["weights"].append([a1, a2, a3])

    vc_grid = GridSearchCV(model_vc, vc_params, cv=10,
                           scoring="f1", verbose=2)  # change back to cv=10
    vc_grid.fit(X, Y)
    vc_best_params = vc_grid.best_params_
    print("Voting Classifier best params: ", vc_best_params)

    return vc_best_params


def fine_late_fuse(estimator_list, X, Y, weights):
    a1 = int(weights[0] * 100)
    a2 = int(weights[1] * 100)
    a3 = int(weights[2] * 100)

    model_vc = VotingClassifier(
        estimators=estimator_list, voting='soft', n_jobs=-1)
    vc_params = {
        'weights': []
    }

    for i in range(max(a1 - 5, 0), min(a1 + 5, 101)):
        for j in range(max(a2 - 5, 0), min(a2 + 5, 101)):
            for k in range(max(a3 - 5, 0), min(a3 + 5, 101)):
                if i + j + k == 100:
                    vc_params['weights'].append([i / 100, j / 100, k / 100])

    vc_grid = GridSearchCV(model_vc, vc_params, cv=10,
                           scoring="f1", verbose=2)  # change back to cv=10
    vc_grid.fit(X, Y)
    vc_best_params = vc_grid.best_params_
    print("Voting Classifier best params: ", vc_best_params)

    return vc_best_params


### Main ###
def main():
    # Preprocessing - uncomment if needed
    generate_feature_vectors()

    ### Training ###
    # Read feature vectors from file if already preprocessed
    print("Reading feature vectors from file")
    features_vader_training = pd.read_json("features_vader_training.json")
    features_flair_training = pd.read_json("features_flair_training.json")
    features_vader_test = pd.read_json("features_vader_test3.json")
    features_flair_test = pd.read_json("features_flair_test3.json")

    feature_cols = ['avg_text', 'avg_summ', 'std_text', 'std_summ', 'pct_verif', 'amt_reviews',
                    'amt_stars']
    X_vader = features_vader_training[feature_cols]
    Y_vader = features_vader_training.awesomeness
    X_flair = features_flair_training[feature_cols]
    Y_flair = features_flair_training.awesomeness

    # Scaling
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_vader)
    X_vader = scaler.transform(X_vader)
    X_flair = scaler.transform(X_flair)

    # Perform Grid Search
    # dt_best_params, lr_best_params, rf_best_params, svm_best_params, knn_best_params, ab_best_params = grid_search(X, Y)

    mlp_best_params_vader = grid_search(X_vader, Y_vader)
    mlp_best_params_flair = grid_search(X_flair, Y_flair)
    with open("mlp_params.txt", 'a') as file:
        file.write("MLP Best Params for VADER:")
        file.write(str(mlp_best_params_vader))
        file.write("MLP Best Params for flair:")
        file.write(str(mlp_best_params_flair))

    # Put in parameters by hand to skip Grid Search - Classifiers not used are commented out
    # model_dt = DecisionTreeClassifier(class_weight=None, criterion="entropy", splitter="best")
    # model_nb = GaussianNB()
    # model_lr = LogisticRegression(C=10, class_weight=None, max_iter=1000, tol=1, n_jobs=-1)
    # model_rf = RandomForestClassifier(max_depth=8, n_estimators=100, n_jobs=-1, criterion="log_loss", class_weight=None)
    # model_svm = SVC(kernel='rbf', verbose=True, max_iter=1000000, tol=1e-5, C=20)
    # model_knn = KNeighborsClassifier(n_neighbors=15, weights='uniform', leaf_size=10, algorithm='brute', n_jobs=-1)
    ### Boosting ###
    # model_ab = AdaBoostClassifier(
    #     RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=-1, criterion="log_loss", class_weight=None),
    #     n_estimators=13, learning_rate=2.075)

    ### Neural Nets ###
    model_mlp_flair = MLPClassifier(
        alpha=1e-8, hidden_layer_sizes=(5, 5, 5), max_iter=10000, solver="adam")
    model_mlp_vader = MLPClassifier(
        alpha=1e-8, hidden_layer_sizes=(5, 5, 5), max_iter=10000, solver="adam")

    ### Late Fusion ###
    # Use VotingClassifier to implement late fusion
    # estimator_list = [('LR', model_lr), ('RF', model_rf), ('KNN', model_knn)]

    # model_vc = VotingClassifier(estimators=estimator_list, voting='soft', weights=[0.0, 1.0], verbose=True, n_jobs=-1)
    # vc_coarse_params = late_fuse(estimator_list, X, Y, 0, 101, 0, 101, 5)
    # print("After coarse search best params for late fusion: ", vc_coarse_params)
    # vc_coarse_params = {'weights': [0.15, 0.75, 0.1]}  # Coarse parameters found for LR, RF, KNN
    # vc_fine_params = fine_late_fuse(estimator_list, X, Y, vc_coarse_params['weights'])
    # print("After fine search best params for late fusion: ", vc_fine_params)

    # Put in parameters by hand to skip Grid Search
    # model_vc = VotingClassifier(estimators=estimator_list, voting='soft', weights=[0.12, 0.72, 0.16], verbose=True, n_jobs=-1)

    vader_models = [value for name, value in locals().items(
    ) if name.startswith('model_') and name.endswith('_vader')]
    vader_model_names = [name for name, value in locals().items(
    ) if name.startswith('model_') and name.endswith('_vader')]
    flair_models = [value for name, value in locals().items(
    ) if name.startswith('model_') and name.endswith('_flair')]
    flair_model_names = [name for name, value in locals().items(
    ) if name.startswith('model_') and name.endswith('_flair')]

    vader_scores = {}
    flair_scores = {}
    scoring_methods = ['recall', 'precision', 'f1', 'roc_auc', 'accuracy']
    vader_scores_avg = {}
    flair_scores_avg = {}

    # Train models using 10-fold cross validation
    for i in range(len(vader_models)):
        model_name = vader_model_names[i]
        print("Running cross validation on " + model_name)
        vader_models[i].fit(X_vader, Y_vader)
        vader_scores[model_name] = \
            cross_validate(vader_models[i], X_vader,
                           Y_vader, cv=10, scoring=scoring_methods)
        vader_scores_avg[model_name] = {}
        for method in scoring_methods:
            vader_scores_avg[model_name][method] = np.mean(
                vader_scores[model_name]["test_" + method])
        pickle.dump(vader_models[i], open(
            "./models/" + model_name + "_vader.pkl", "wb"))

    for i in range(len(flair_models)):
        model_name = flair_model_names[i]
        print("Running cross validation on " + model_name)
        flair_models[i].fit(X_flair, Y_flair)
        flair_scores[model_name] = \
            cross_validate(flair_models[i], X_flair,
                           Y_flair, cv=10, scoring=scoring_methods)
        flair_scores_avg[model_name] = {}
        for method in scoring_methods:
            flair_scores_avg[model_name][method] = np.mean(
                flair_scores[model_name]["test_" + method])
        pickle.dump(flair_models[i], open(
            "./models/" + model_name + "_flair.pkl", "wb"))

    ### Visualize Scores ###
    print("Generating bar plots")
    generate_bar_plot(vader_models, vader_model_names,
                      scoring_methods[0:3], vader_scores_avg)
    generate_bar_plot(flair_models, flair_model_names,
                      scoring_methods[0:3], flair_scores_avg)

    print("Printing scores to vader_scores_avg.csv")
    vader_scores_avg_df = pd.DataFrame(vader_scores_avg)
    vader_scores_avg_df = vader_scores_avg_df.round(4)
    vader_scores_avg_df.to_csv("./scores/vader_scores_avg.csv")

    print("Printing scores to flair_scores_avg.csv")
    flair_scores_avg_df = pd.DataFrame(flair_scores_avg)
    flair_scores_avg_df = vader_scores_avg_df.round(4)
    flair_scores_avg_df.to_csv("./scores/flair_scores_avg.csv")

    ### Read models if dumped ###
    print("Reading dumped models from file")
    # model_dt = pickle.load(open("./models/model_dt.pkl", "rb"))
    # model_nb = pickle.load(open("./models/model_nb.pkl", "rb"))
    # model_lr = pickle.load(open("./models/model_lr.pkl", "rb"))
    # model_rf = pickle.load(open("./models/model_rf.pkl", "rb"))
    # model_svm = pickle.load(open("./models/model_svm.pkl", "rb"))
    # model_knn = pickle.load(open("./models/model_knn.pkl", "rb"))
    # model_ab = pickle.load(open("./models/model_ab.pkl", "rb"))
    # model_vc = pickle.load(open("./models/model_vc.pkl", "rb"))
    model_mlp_vader = pickle.load(open("./models/model_mlp_flair.pkl", "rb"))

    final_model = model_mlp_vader

    ### Predictions ###
    print("Running predictions")
    X_test = features_vader_test[feature_cols]
    X_test = scaler.transform(X_test)

    predictions = final_model.predict(X_test)

    print("Writing predictions to predictions.json")
    asin_test = pd.read_json("Toys_and_Games/test3/product_test.json")
    asin_test.sort_index(axis=0, inplace=True)
    print(asin_test.head(10))
    asin_test.insert(1, "awesomeness", predictions)
    print(asin_test.head(10))
    asin_test.to_json("predictions.json")


if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    end = datetime.datetime.now()
    print("TOTAL TIME: ", end - start)
