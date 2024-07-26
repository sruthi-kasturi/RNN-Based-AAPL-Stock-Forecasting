# RNN-Based-AAPL-Stock-Forecasting

## Overview

Welcome to the AAPL Stock Price Prediction project! This repository demonstrates the use of Recurrent Neural Networks (RNNs) to forecast stock prices of Apple Inc. (AAPL) using historical data. By leveraging the power of LSTM layers, we aim to capture the temporal dependencies and trends in stock price movements.

## Objective

The primary goal of this project is to develop a predictive model that can accurately forecast the future stock prices of AAPL, aiding in better investment decisions and market analysis.

## Approach

### Data Collection

- **Source**: Historical stock price data for Apple Inc. (AAPL) was collected from Yahoo Finance.
- **Time Period**: Data spans from January 2014 to January 2019, with the training set covering January 2014 to January 2018, and the test set covering January 2018 to January 2019.

### Data Preprocessing

- **Scaling**: Applied MinMaxScaler to normalize the stock price data.
- **Sequence Creation**: Created sequences of 60 days of historical data to predict the stock price of the 61st day.

### Model Building

- **Architecture**: Built a stacked LSTM model with four layers to capture the complex patterns in stock price data.
- **Regularization**: Added Dropout layers to prevent overfitting.
- **Output**: The model outputs the predicted stock price for the next day.

### Model Training

- **Optimizer**: Used Adam optimizer.
- **Loss Function**: Mean Squared Error (MSE) was used as the loss function.
- **Epochs and Batch Size**: Trained the model for 100 epochs with a batch size of 30.

### Evaluation

- **Performance Metrics**: Evaluated the model on the test set by comparing predicted stock prices with actual prices using visualizations.
- **Classification Task**: Additionally, implemented a classification task to predict the direction of stock price movement (up or down) and evaluated it using accuracy and F1-score.

## Results

- The RNN model demonstrated promising results in capturing the trend of AAPL stock prices.
- The classification model showed an accuracy of 53.4% and an F1-score of 69.6%, indicating room for improvement and further fine-tuning.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/sruthi-kasturi/aapl-stock-prediction-rnn.git
   cd aapl-stock-prediction-rnn


2. Install the required dependencies:
   pip install -r requirements.txt

3. Run the Jupyter Notebook to see the code and results for the project:
   jupyter notebook

4. Open and execute the notebook AAPL_RNN_Time_Series_Prediction.ipynb.

## Dependencies
- Python 3.7+
- Jupyter Notebook
- Keras
- TensorFlow
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- yfinance

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.


