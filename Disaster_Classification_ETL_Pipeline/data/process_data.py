import sys            # System-specific functionalities and command-line arguments
import numpy as np    # Numerical computations and array operations
import pandas as pd   # Data manipulation and analysis using DataFrames and Series
from sqlalchemy import create_engine   # Create database connections and perform SQL operations


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
        messages_filepath (str): Filepath of the messages CSV file.
        categories_filepath (str): Filepath of the categories CSV file.

    Returns:
        pd.DataFrame: Merged DataFrame containing messages and categories.
    """
    # Load datasets
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    
    # Merge datasets
    df = messages.merge(categories, on='id', how='inner')
    
    # Create a DataFrame of the 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)
    
    # Extract new column names for categories from the first row
    category_colnames = categories.iloc[0,:].apply(lambda x: x[:-2])
    
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to binary (0 or 1)
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
        categories[column].replace(2, 1, inplace=True)
    
    # Drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    return df


def clean_data(df):
    """
    Clean the provided DataFrame by removing duplicates.

    Args:
        df (pd.DataFrame): The DataFrame to be cleaned.

    Returns:
        pd.DataFrame: Cleaned DataFrame with duplicates removed.
    """
    # Drop duplicates
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    """
    Save the DataFrame to an SQLite database.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        database_filename (str): Name of the SQLite database file.

    Returns:
        None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)



def main():
    """
    Main function to execute the data processing pipeline.

    This function reads command line arguments, processes data, and saves it to a database.

    Args:
        None

    Returns:
        None
    """
    # Check if the number of command line arguments is 4
    if len(sys.argv) == 4:
        # Extract command line arguments: messages file path, categories file path, database file path
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


if __name__ == '__main__':
    main()