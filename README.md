# Disaster_Response_WebApp

## Project Summary
This project analyses messages provided by [FigureEight](https://www.figure-eight.com/). The dataset contains messages drawn from natural desaster events and news articles about different disasters.

There are 36 labels related to disaster response.

Our jog here is to extract, transform, and load the data into a sqlit3 database. Then extract the loaded data to fit it into a machine learning classifier using a pipeline.

The trained model will live in a web app using Flask where users can enter a message and see the related labels that pertains to that message.

## Instructions

These are the paramters to pass to the Python scripts:
- process_data.py | python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
- train_classifier.py | python model/train_classifier.py data/DisasterResponse.db model/classifier.pkl