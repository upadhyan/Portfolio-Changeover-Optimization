import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ray import tune

# from neuralforecast.auto import AutoNHITS
# from neuralforecast.core import NeuralForecast

from neuralforecast.models import NHITS, NBEATS
from neuralforecast.auto import AutoNBEATS
from neuralforecast import NeuralForecast
# from neuralforecast.losses.pytorch import HuberLoss, MQLoss

from neuralforecast.losses.pytorch import MAE, RMSE
from neuralforecast.losses.numpy import mae, mse

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

DEFAULT_HP = {
                "h": 90,
                "input_size": 2 * 90,
                "loss": MAE(),
                "max_steps": 1000,
                "stack_types": ['trend', 'seasonality', 'identity'],
                "n_polynomials": 4,
                "n_blocks": [1, 1, 1],
                "mlp_units": [[512, 512],[512, 512], [512, 512]],
                "val_check_steps": 10,
                "step_size": 90
}


DEFAULT_HP_AUTO = {
    "input_size": tune.qrandint(90, 270, 90),
    "loss": tune.choice([MAE()]),
    "max_steps": tune.randint(250, 2000),
    "stack_types": tune.choice([['trend', 'seasonality', 'identity']]),
    "n_polynomials": tune.randint(1, 5),
    "n_blocks": tune.choice([[1, 1, 1]]),
    "mlp_units": tune.choice([[[512, 512],[512, 512], [512, 512]]]),
    "val_check_steps": tune.choice([10]),
    "step_size": tune.choice([90]),
    "lr": tune.loguniform(1e-5, 1e-1),
}



def load_data(stock_price_data_path = "raw_data/spx_stock_prices.parquet"):
    prices = pd.read_parquet(stock_price_data_path)
    # create an daterange 
    date_range = pd.date_range(start=min(prices.index), end=max(prices.index), freq='B')
    prices = prices.reindex(date_range, method='ffill')
    # replace zeros with nan
    prices = prices.replace(0, np.nan)
    # name the index
    prices.index.name = 'ds'
    # Convert from wide to long format
    prices = prices.reset_index()
    melted_df = pd.melt(prices, id_vars=['ds'], value_vars=prices.columns[1:], var_name='unique_id', value_name='y')
    melted_df['ds'] = pd.to_datetime(melted_df['ds'])
    melted_df = melted_df.sort_values(['unique_id', 'ds'])
    melted_df = melted_df.reset_index(drop=True)
    melted_df = melted_df.dropna()
    return melted_df

def train_model(train_dataset, hyperparams = DEFAULT_HP):
    models = [NBEATS(**hyperparams)]
    nf = NeuralForecast(models=models, freq='B')
    nf.fit(df=train_dataset)
    return nf

def train_auto_model(train_dataset, hyperparams = DEFAULT_HP_AUTO):
    models = [AutoNBEATS(
        h=90,
        num_samples=10,
        config=hyperparams,
        alias='auto_nbeats'
    )]
    nf = NeuralForecast(models=models, freq='B')
    nf.fit(df=train_dataset)
    return nf

def retrieve_valid_tickers(melted_data, split_date):
    first_date_df = melted_data.groupby('unique_id')['ds'].min().reset_index()
    first_date_df = first_date_df.rename(columns={'ds': 'first_date'})
    # get the last date of each ticker
    last_date_df = melted_df.groupby('unique_id')['ds'].max().reset_index()
    last_date_df = melted_data.rename(columns={'ds': 'last_date'})
    # merge the first and last date
    first_last_date_df = first_date_df.merge(last_date_df, on='unique_id', how='left')
    # get the tickers whose first date is at least 180 days before the split_date and last date is at least 90 days after split_date
    valid_tickers = first_last_date_df[(first_last_date_df['first_date'] < pd.to_datetime(split_date) - pd.Timedelta(days=180)) & (first_last_date_df['last_date'] > pd.to_datetime(split_date) + pd.Timedelta(days=90))]['unique_id'].tolist()
    # get the random tickers
    return valid_tickers

# create the if name == main function
if __name__ == "__main__":
    melted_df = load_data()
    split_date = '2019-01-01'
    valid_tickers = retrieve_valid_tickers(melted_df, split_date)
    # save the valid tickers to a csv file
    pd.DataFrame(valid_tickers).to_csv(f'raw_data/valid_tickers_{split_date}.csv', index=False)
    valid_ticker_df = melted_df[melted_df['unique_id'].isin(valid_tickers)]
    Y_train_df = valid_ticker_df[valid_ticker_df['ds'] < split_date]
    nf = train_model(Y_train_df)
    nf.save(path='./nf_models/',
        model_index=None, 
        overwrite=True,
        save_dataset=True)

    forecast_date = '2019-03-02'
    other_valid_tickers = retrieve_valid_tickers(valid_ticker_df, forecast_date)
    # Select 10 random tickers
    ten_tickers = np.random.choice(other_valid_tickers, 10)

    
    test_input = valid_ticker_df[(valid_ticker_df['unique_id'].isin(ten_tickers)) & (valid_ticker_df['ds'] <= forecast_date)]
    print(test_input.head())
    fcst = nf.predict(df=test_input).reset_index()
    #rename the nbeatsx column to yhat
    fcst = fcst.rename(columns={'NBEATS': 'yhat'})
    fcst = fcst.merge(valid_ticker_df, on=['unique_id', 'ds'], how='left')
    print(fcst.head())
    print(r2_score(fcst['y'], fcst['yhat']))
    print(mean_absolute_error(fcst['y'], fcst['yhat']))

    first_ticker = fcst[fcst['unique_id'] == ten_tickers[0]]
    plt.plot(first_ticker['ds'], first_ticker['y'], label='actual')
    plt.plot(first_ticker['ds'], first_ticker['yhat'], label='forecast')
    plt.legend()
    plt.show()
    

