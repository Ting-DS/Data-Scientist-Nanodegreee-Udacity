import json  # Used for handling JSON data.
import plotly  # Library for interactive data visualizations.
import pandas as pd  # Library for data analysis and manipulation.

from nltk.stem import WordNetLemmatizer  # Used for word lemmatization.
from nltk.tokenize import word_tokenize  # Used for text tokenization.

from flask import Flask  # Micro web framework for building web applications.
from flask import render_template, request, jsonify  # Flask functions for web app development.

from plotly.graph_objs import Bar  # Used for creating bar charts.
from joblib import dump, load  # Library for efficient object saving/loading.
import joblib  # Library for object saving/loading.

from sqlalchemy import create_engine  # Library for database interaction.

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # category data for plotting
    categories =  df[df.columns[4:]]
    cate_counts = (categories.mean()*categories.shape[0]).sort_values(ascending=False)
    cate_names = list(cate_counts.index)
    
    # Plotting of Categories Distribution in news Genre
    news_cate = df[df.genre == 'news']
    news_cate_counts = (news_cate.mean()*news_cate.shape[0]).sort_values(ascending=False)
    news_cate_names = list(news_cate_counts.index)
    
    # Plotting of Categories Distribution in direct Genre
    direct_cate = df[df.genre == 'news']
    direct_cate_counts = (direct_cate.mean()*direct_cate.shape[0]).sort_values(ascending=False)
    direct_cate_names = list(direct_cate_counts.index)
    
    # Plotting of Categories Distribution in social Genre
    social_cate = df[df.genre == 'news']
    social_cate_counts = (social_cate.mean()*social_cate.shape[0]).sort_values(ascending=False)
    social_cate_names = list(social_cate_counts.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # category plotting (Visualization#2)
        {
            'data': [
                Bar(
                    x=cate_names,
                    y=cate_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
            
        },
        # Categories Distribution in news Genre (Visualization#3)
        {
            'data': [
                Bar(
                    x=direct_cate_names[1:],
                    y=direct_cate_counts[1:]
                )
            ],

            'layout': {
                'title': 'Categories Distribution in <b>Direct</b> Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories in direct Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=news_cate_names[1:],
                    y=news_cate_counts[1:]
                )
            ],

            'layout': {
                'title': 'Categories Distribution in <b>News</b> Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories in news Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=social_cate_names[1:],
                    y=social_cate_counts[1:]
                )
            ],

            'layout': {
                'title': 'Categories Distribution in <b>Social</b> Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories in social Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
