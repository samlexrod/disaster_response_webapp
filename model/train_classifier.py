import sys
import re
import pickle
import pandas as pd
import numpy as np
import nltk
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

nltk.download(['punkt', 'stopwords', 'wordnet'])

def extract_data(database_name):
    """
    A parameter to load the data into X, y, and category names
    parameter
    ---------
    database_name : the name of the database to get the messages from
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