from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestRegressor

##################################################
# Linear Regression
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_linear_regression(model, X):
    return model.predict(X)

def evaluate_linear_regression(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score

##################################################
# Naive Bayes
def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def predict_naive_bayes(model, X):
    return model.predict(X)

def evaluate_naive_bayes(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score

##################################################
# Decision Tree Classifier
def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def predict_decision_tree(model, X):
    return model.predict(X)

def evaluate_decision_tree(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score

##################################################
# KNN Classifier
def train_knn(X_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def predict_knn(model, X):
    return model.predict(X)

def evaluate_knn(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score

##################################################
# SVM Classifier
def train_svm(X_train, y_train):
    model = SVC()
    model.fit(X_train, y_train)
    return model

def predict_svm(model, X):
    return model.predict(X)

def evaluate_svm(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score

##################################################
# Logistic Regression
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def predict_logistic_regression(model, X):
    return model.predict(X)

def evaluate_logistic_regression(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score

##################################################
# Decision Tree Regressor
def train_decision_tree_regressor(X_train, y_train):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    return model

def predict_decision_tree_regressor(model, X):
    return model.predict(X)

def evaluate_decision_tree_regressor(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score

##################################################
# SVR
def train_svr(X_train, y_train):
    model = SVR()
    model.fit(X_train, y_train)
    return model

def predict_svr(model, X):
    return model.predict(X)

def evaluate_svr(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score

##################################################
# Random Forest Regressor
def train_random_forest_regressor(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def predict_random_forest_regressor(model, X):
    return model.predict(X)

def evaluate_random_forest_regressor(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score

##################################################
# KNN Regressor
def train_knn_regressor(X_train, y_train, n_neighbors=5):
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def predict_knn_regressor(model, X):
    return model.predict(X)

def evaluate_knn_regressor(model, X_train, y_train, X_test, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score

##################################################

def get_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred)


import re
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words('english'))
punc = string.punctuation
def clean_text(text):
    #  Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    #  Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    #  Remove newline, tab, and extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    #  Remove punctuation
    text = text.translate(str.maketrans('', '', punc))
    
    #  Remove stopwords
    text = " ".join([word for word in text.split() if word.lower() not in stopword])
    
    #  Correct spelling
    #text = str(TextBlob(text).correct())#-----> high computation
    #  lowercasing
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stopword and len(word) > 1])

    return text.strip()

def convert_price_columns(df, columns):
    for col in columns:
        df[col] = df[col].str.replace(",", "").astype(float)
    return df

import pickle
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
