# Disaster Response Classification Pipeline

## Introduction
This project leverages a dataset provided by [Appen](https://www.appen.com/), containing tens of thousands of real messages from various sources during disaster events. We have constructed a powerful **multi-label classification machine learning (ML) model** aimed at efficiently categorizing messages sent during disasters. The dataset encompasses information across 36 predefined categories, including but not limited to aid-related, medical assistance, search and rescue, among others. By effectively categorizing these messages, we are able to swiftly relay them to the appropriate disaster relief organizations, thereby enhancing their information processing efficiency and enabling targeted and timely rescue efforts.

Within the project, we have established a comprehensive technical framework, encompassing **ETL** (Extract, Transform, Load), **NLP** (Natural Language Processing), and **ML pipelines**, all implemented using the **SQLite database** and **Python** programming language. Given that disaster-related information can fall into multiple categories, we are dealing with a multi-label classification task. This implies that a single message can belong to one or more categories simultaneously.

Ultimately, we have utilized **Flask** and **Plotly** technologies to create an intuitive web application for showcasing the visualization of data. This application allows stakeholders to easily input information and instantly obtain classification results. Below is the screenshots of the web application, showcasing the user interface and visual representation of classification outcomes:

![Screenshot1 of Web App](https://github.com/Ting-DS/Disaster_Response_Classification_Pipeline/blob/main/Web_App.png)

![Screenshot2 of Web App](https://github.com/Ting-DS/Disaster_Response_Classification_Pipeline/blob/main/Distribution_Message_Categories.png)

## Installation
To ensure proper functionality, it is required to run this project using Python 3 along with the following libraries:

- **numpy**
- **pandas**
- **sqlalchemy**
- **re**
- **NLTK**
- **pickle**
- **Sklearn**
- **plotly**
- **flask**

Please ensure that you have these libraries installed before proceeding with the project setup.

You can install these libraries using the following command:
```bash
pip install numpy pandas sqlalchemy nltk scikit-learn plotly flask
```

## File Description
~~~~~~~
        Disaster_Response_Classification_Pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- classifier.pkl
                |-- train_classifier.py
          |-- Preparation
                |-- categories.csv
                |-- ETL Pipeline Preparation.ipynb
                |-- ETL_Preparation.db
                |-- messages.csv
                |-- ML Pipeline Preparation.ipynb
                |-- README
          |-- README
~~~~~~~

## File Descriptions

1. **App Folder**: This folder includes the "run.py" script and the "templates" folder for the web application.

2. **Data Folder**: Inside this folder, you will find the "DisasterResponse.db" SQLite database file, along with "disaster_categories.csv" and "disaster_messages.csv" datasets. The "process_data.py" script is also included for data cleaning and transfer.

3. **Models Folder**: This folder contains the trained machine learning model "classifier.pkl" and the "train_classifier.py" script used for model training.

4. **README File**: This file provides an overview of the project, its structure, and instructions for setup and usage.

5. **Preparation Folder**: This folder contains various files that were used during the project's development. **Please note that this folder is not necessary for the project's operation and can be disregarded**.

## Instructions
To properly set up your database and model for this project, follow these steps in the project's root directory:

1. Run the following commands:

    - To execute the ETL pipeline for data cleaning and database storage:
      ```
      python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
      ```

    - To run the ML pipeline for training the classifier and saving the model:
      ```
      python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
      ```

2. After completing the above steps, navigate to the app's directory and run the following command to launch the web app:
   ```
   python run.py
   ```

3. Open your web browser and visit the following URL:
   ```
   http://0.0.0.0:3001/
   ```
## Discussion (Imbalanced Problem)
The provided dataset demonstrates class imbalance, with certain labels, such as "water," having relatively fewer instances. This imbalance can significantly impact model training, as the model might exhibit a bias towards majority classes, struggling to make accurate predictions for minority classes. To address this, techniques like **stratified sampling** during the train-test split have been applied to maintain class distribution.

Given the nature of discerning disaster-related messages, striking a balance between precision and recall becomes imperative. Emphasizing **precision** holds value when aiming to minimize false positives and ensuring that predicted labels are correct. Conversely, prioritizing **recall** is pivotal when the objective is to minimize false negatives and correctly classify as many relevant instances as possible.

Within the `train_classifier.py` script, options for parameter tuning have been integrated using **GridSearchCV**, affording the ability to tailor the model's behavior according to the problem's specifics. Experimenting with these parameters offers the opportunity to achieve a suitable trade-off between precision and recall for different categories, thereby enhancing the overall performance of the model.

## Licensing, Authors, Acknowledgements
I would like to extend my sincere gratitude to [Appen](https://www.appen.com/) for their contribution in making this valuable resource available to the public. A special acknowledgment goes to Udacity for their exceptional guidance throughout this project. Feel free to utilize the contents of this work, and when doing so, please remember to appropriately attribute the contributions of myself, Udacity, and/or Figure Eight."

