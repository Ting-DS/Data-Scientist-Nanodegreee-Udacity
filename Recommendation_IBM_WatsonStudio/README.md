## Recommendation System Design With IBM Watson Studio
### Udacity Data Scientist NanoDegree Project
### Project Description
Within IBM Watson Studio, there exists a vast collaborative community ecosystem encompassing a multitude of **articles**, datasets, notebooks, and other resources pertaining to Artificial Intelligence (AI) and Machine Learning (ML). Users have the opportunity to engage with these resources directly. With the aim of enhancing **user experiences** and facilitating seamless access to these resources, we have undertaken a recommendation system project. The objective of this project is to provide personalized recommendations for each user, thereby tailoring their experience according to their preferences. Our plan involves analyzing user interactions with articles on the IBM Watson Studio platform and subsequently **suggesting new articles** that align with their interests and preferences. Through this project, we aspire to enable users to effortlessly discover the wealth of resources available on the platform, thereby enhancing their overall usage experience.
 
Below is an example of what the dashboard could look like displaying articles on the **IBM Platform**.
Althouh The dashboard is just showing the newest articles. It could imagine having a recommendation board available here that shows the articles that are most pertinent to a specific user. In order to determine which articles to show to each user, It will be performing a study of the data available on the [IBM Watson Studio platform](https://dataplatform.cloud.ibm.com/).

<img src="img/Recommendations_with_IBM.png" width="80%" alt="Recommendations with IBM">

### Installation - Python 3

- **pandas**: A library for data manipulation and analysis, providing versatile tools for working with structured data.

- **numpy**: A numerical computing library for efficient operations on arrays and matrices.

- **matplotlib**: A visualization library for creating plots and charts.

- **pickle**: A module for serializing and deserializing Python objects, useful for saving and loading data.

- **re**: A library for working with regular expressions, enabling pattern matching and manipulation of text.

- **nltk**: The Natural Language Toolkit, designed for natural language processing tasks.

- **sklearn**: Scikit-learn, a comprehensive machine learning library with various tools for data mining and analysis.

- **jupyter**: An interactive environment for creating documents containing live code and explanations.

Make sure you have these libraries installed before running any scripts in this project. You can usually install them using pip:

```bash
pip install pandas numpy matplotlib nltk scikit-learn
```


### Data
<p>2 csv files</p>
<ul>
  <li>user-item-interactions.csv: Interactions between users and articles.</li>
  <li>articles_community.csv: Contents of articles.</li>
</ul>

### Project Contents

#### I. Exploratory Data Analysis

- Initial data exploration to understand the dataset.
- Addressing basic questions about the data.
- Set the foundation for the recommendation system.

#### II. Rank Based Recommendations

- Identifying most popular articles based on interactions.
- Recommending these popular articles to users.
- Simple recommendation strategy without ratings.

#### III. User-User Based Collaborative Filtering

- Finding similar users based on their interactions.
- Recommending items to users similar to others.
- Progress toward personalized recommendations.

#### IV. Content Based Recommendations

- Exploring content-based recommendation methods.
- Utilizing NLP techniques for creative recommendations.
- Optional content-based approach to enhance recommendations.

#### V. Matrix Factorization

- Applying **machine learning with SVD matrix decomposition**.
- Predicting new article interactions using the model.
- Assessing the effectiveness of predictions.
- Discussing methods for further improvement and testing.

### Acknowledgement
The dataset has been graciously provided by <a href="https://www.udacity.com/">Udacity</a> in collaboration with <a href="https://www.ibm.com/products/watson-studio">IBM Watson Studio</a>. Your access to this valuable data is made possible through their contribution.
