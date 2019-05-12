import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    A function that gets raw data, merges it, and returns it as a pandas dataframe
    parameter
    ---------
    messages_filepath : the csv or txt flat file with all the messages
    categories_filepath : the csv or txt flat file with all the categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return messages.merge(categories,
                    on='id',
                    how='left')

def clean_data(df):
    """
    A function that gets a dirty dataframe and returns a clean data frame
    parameter
    ---------
    df : the pandas dataframe to be cleaned    
    return
    ------
    A clean dataframe
    """
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.str.split('-').apply(lambda x: x[0])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    df.drop('categories', axis=1, inplace=True)
    
    df = pd.concat([df, categories], axis=1)
    
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    A procedure to save the data into sqlite3
    parameter
    ---------
    database_filename : the name of the database to be created
    """    
    sqlite_db = 'sqlite:///%s' % database_filename
    engine = create_engine(sqlite_db)

    # Create table, insert messages or replace messages
    df.to_sql('Messages', engine, index=False, if_exists='replace')  
    
    print(pd.read_sql("SELECT * FROM Messages LIMIT 5", engine))


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')