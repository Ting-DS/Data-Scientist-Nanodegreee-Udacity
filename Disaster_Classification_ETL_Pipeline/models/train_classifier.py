import sys            # Access to system-specific functionalities and command-line arguments
from sqlalchemy import create_engine   # Create database connections and perform SQL operations
import numpy as np    # Numerical computations and array operations
import pandas as pd   # Data manipulation and analysis using DataFrames and Series

from sklearn.metrics import classification_report, accuracy_score   # Model evaluation metrics
from sklearn.model_selection import train_test_split   # Split dataset for training and testing

from sklearn.ensemble import RandomForestClassifier   # Random Forest classifier model
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer   # Text vectorization 

from sklearn.pipeline import Pipeline   # Create a pipeline of data processing steps
from sklearn.model_selection import GridSearchCV   # Grid search for hyperparameter tuning
from sklearn.multioutput import MultiOutputClassifier   # Handling multi-output classification

import nltk           # Natural Language Toolkit, used for NLP tasks
nltk.download(['punkt', 'wordnet']) 
# Download NLTK resources ('punkt' for tokenization, 'wordnet' for lemmatization)
from nltk.tokenize import word_tokenize   # Tokenization for text data
from nltk.stem import WordNetLemmatizer   # Word lemmatization
import re             # Regular expression operations, used for text preprocessing

import pickle   # Serialization and deserialization of Python objects


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df.message
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names 

def tokenize(text):
    """
    Tokenize and preprocess the input text.

    Args:
        text (str): Input text to be tokenized.

    Returns:
        list: A list of preprocessed tokens.
    """
    # Regular expression to detect URLs in the text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)

    # Replace detected URLs with a placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize the text using NLTK's word_tokenize
    tokens = word_tokenize(text)

    # Initialize the WordNet Lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    # Lemmatize and lowercase each token
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Build a machine learning model using a pipeline and grid search.

    Returns:
        GridSearchCV: A GridSearchCV object containing the pipeline and search parameters.
    """
    # Create a ML pipeline with data processing and modeling steps
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),  # Text vectorization using tokenization function
        ('tfidf', TfidfTransformer()),   # TF-IDF transformation
        ('clf', MultiOutputClassifier(RandomForestClassifier()))   # Multi-output random forest classifier
    ])
    
    # Define a parameter grid for grid search (Simple settings for production)
    parameters = {
        'clf__estimator__n_estimators': [10,20,30],  # Number of decision trees in the forest
        'clf__estimator__max_depth': [10],# Maximum depth of each decision tree
        'clf__estimator__min_samples_split': [2],  # Minimum number of samples required to split an internal node
    }
    
    # Create a GridSearchCV object with the pipeline and parameter grid
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=5)
    
    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate the trained model's performance on the test data.

    Args:
        model: Trained machine learning model.
        X_test (array-like): Test data features.
        y_test (array-like): True labels of the test data.
        category_names (list-like): Names of the output categories.

    Returns:
        None (prints the evaluation results).
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    class_report = classification_report(y_test, y_pred, target_names=category_names)
    
    print("Evaluation Results:")
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(class_report)


def save_model(model, model_filepath):
    """
    Save a trained machine learning model to a file using pickle.

    Args:
        model: Trained machine learning model.
        model_filepath (str): Filepath to save the model.

    Returns:
        None
    """
    # Open the file in write binary mode and save the model using pickle
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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