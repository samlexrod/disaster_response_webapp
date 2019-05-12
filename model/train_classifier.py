import sys
import re
import pickle
import pandas as pd
import numpy as np
import nltk
import time
import cloudpickle
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download(['punkt', 'stopwords', 'wordnet'])

def extract_data(database_name):
    """
    A parameter to load the data into X, y, and category names
    parameter
    ---------
    database_name : the name of the database to get the messages from
    
    returns
    -------
    Extracted features and label variables, and category names in a tuple
    """
    sqlite_db = 'sqlite:///{}'.format(database_name)
    engine = create_engine(sqlite_db)
    print(pd.read_sql("SELECT * FROM Messages LIMIT 5", engine))
    
    # Extract the data from sqlite3
    df = pd.read_sql_table("Messages", engine)
    
    y_frame = df.select_dtypes(include=np.int64).drop('id', axis=1)
    X = df.message.str.lower().values
    y = y_frame.values
    
    category_names =  y_frame.columns
    
    return X, y, category_names

def tokenize(text):
    """
    A function to tokenize each message and convert them into words
    It is designed to be passed into the Count Vectorizer
    parameter
    ---------
    text : the messages to tokenize, stem, and lemmatize
    
    returns
    -------
    A list of cleaned words
    """
    # Clean the message
    text = re.sub("\W", " ", text)
    
    # Tokenize the words
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words('english')]
        
    # Stemmer and Lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    words = [stemmer.stem(w) for w in words]    
    words = [lemmatizer.lemmatize(w) for w in words]    
    
    return words

def build_model():
    """
    The model creator that fits the data over a series of steps preprocessing 
    before finally fitting into the classifier
    returns
    -------
    A fitted model
    """
    forest = RandomForestClassifier()
    clf_multi = MultiOutputClassifier(forest, n_jobs=-1)
    
    class MessLen(BaseEstimator, TransformerMixin):
        """
        Returns the message length. 
        """
        
        def __init__(self):
            """
            Sets the incremetal count
            """
            self.i = 1
            
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            """
            Print corss validation fold and time.
            """
            print("CrossValidationFold:", self.i, 'on', time.asctime(time.localtime(time.time())))
            self.i += 1
            return pd.DataFrame(pd.Series(X).apply(len))

    pipeline = Pipeline(
        steps=[
            ('features', FeatureUnion(
                transformer_list=[
                    ('text_pipeline', Pipeline(
                        steps=[
                            ('vect', CountVectorizer(tokenizer=tokenize)),
                            ('tfidf', TfidfTransformer())
                        ])),
                    ('sentlen', MessLen())
                ])),
            ('clf', clf_multi)
        ])
    
    
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [10, 100, 200],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'sentlen': 0.5}, 
            {'text_pipeline': 0.5, 'sentlen': 1})
    }
    
    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    To evaluate the model that predicts the test set, prints the evaluation metrics
    parameters
    ----------
    model : the pipeline to use
    X_test : the test split of the messages
    Y_test : the test split of the cleaned labels
    category_names : a list of the label names
    """
    y_pred = model.predict(X_test)
    
    for i, col in enumerate(category_names):
        true = Y_test[:, i]
        pred = y_pred[:, i]
    
        print(f"Score: {(true == pred).mean():2.2%}")
        print(col.upper(), '\n', classification_report(true, pred), '-- '*18, '\n')
        
def save_model(model, model_filepath):
    """
    To export the trained model into a pickle file that the app will use to predict
    parameters
    ----------
    model : trained model
    model_filepath : the name of the pickle file
    
    returns
    -------
    A pickle file containing the trained model
    """    
    with open(model_filepath, 'wb') as handle:
        cloudpickle.dump(model, handle)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = extract_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':    
    main()