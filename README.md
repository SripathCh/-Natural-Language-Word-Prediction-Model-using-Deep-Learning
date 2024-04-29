# Natural Language Word Prediction Model using Deep Learning

## Project Overview
This project involves the development and deployment of a language model using LSTM (Long Short-Term Memory) networks. The primary goal is to predict the next word in a sequence, enhancing the capabilities of predictive text applications.

## Repository Structure
- **Flask_Deployment_Code/**: Contains the Flask application for deploying the LSTM model as a web service.
  - `app.py`: The Flask application file.
  - `model.py`: Python script for loading and running the LSTM model.
  - `nextwordmodel1.h5`: The trained LSTM model.
  - `static/`: Contains CSS files for the web application.
  - `templates/`: Contains HTML files for the web application.
- **Model_Development_Code/**: Jupyter notebooks and assets used in the development of the LSTM model.
  - `Model.ipynb`: Notebook for training the LSTM model.
  - `Testing.ipynb`: Notebook for testing the LSTM model.
  - `Trial.ipynb`: Preliminary experiments with the LSTM architecture.
  - `nxtwordmodel.h5`: The saved LSTM model.
- `B12.pdf`: Documentation or report related to the project.

## Features
- **Word Prediction**: Uses LSTM networks to predict the next word based on the input sequence.
- **Flask Web Application**: A simple web interface to interact with the model in real time.
