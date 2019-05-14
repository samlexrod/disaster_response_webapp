# Disaster_Response_WebApp

## Project Summary
This project analyses messages provided by [FigureEight](https://www.figure-eight.com/). The dataset contains messages drawn from natural disaster events and news articles about different disasters.

There are 36 labels related to disaster response.

Our job here is to extract, transform, and load the data into a sqlit3 database. Then extract the loaded data to fit it into a machine learning classifier using a pipeline.

The trained model will live in a web app using Flask where users can enter a message and see the related labels that pertain to that message.

## Instructions
Get the the root folder where the data, model, and app folders are located and run the following lines in anaconda prompt (if you are using anaconda).

These are the instructions to run the scripts and web app:
1. python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
1. python model/train_classifier.py data/DisasterResponse.db model/classifier.pkl
1. cd app 
1. python run.py
1. In a browser, type localhost:3001 and press enter and the web app will load.


## Files and Folder Descriptions

**data** : It contains the figure-eight disaster response datasets.
* disaster_categories.csv - includes the label categories to predict 
* disaster_messages.csv - includes the messages to use to predict the labels
* DisasterResponse.db - it holds the cleaned dataset loaded by the process_data.py script
* process_data.py - it extracts, transforms, and loads the data into a sqlite3 database

**model** : It contains the trained model in a pkl file and the script that dumps the classifier.
* classifier.pkl - holds the trained model dumped by the train_classifier.py script
* train_classifier.py - the script to extract the data from the DisasterResponse database, preprocess the data, trains the model, and dumps the model in a pickle file

**app** : It contains the web app files.
* run.py - it contains the script that starts the local host and runs the web app

**app/templates** : contains the HTML templates for the app to work
* go.html - the html template that shows the results of the classified message
* master.html - the HTML template that contains the user input form and graphs about the data.

**.gitignore** : this is used to ignore files on GitHub. To know more about this file visit: https://www.git-scm.com/docs/gitignore

**LICENSE** : This is the license to use the code. It basically says, "Use it however you like at your own risk."

**README** : You are reading me!

**airbnb_ml.yml** : This is the anaconda environment used for the analysis. The instructions on how to recreate this environment are going to be in the installation section below.
