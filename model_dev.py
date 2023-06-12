import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.distributions import Distribution
from torch.distributions import Bernoulli, Normal, StudentT, Poisson, NegativeBinomial



# Hyperparameter Tuning
import ray
from ray import tune
ray.shutdown()
ray.init(num_cpus=20, num_gpus=1)
## set raytune working directory os variable
import os
os.environ['TUNE_ORIG_WORKING_DIR'] = os.getcwd()


from neuralforecast.models import NHITS, NBEATSx, PatchTST
from neuralforecast.auto import AutoNBEATSx, AutoPatchTST
from neuralforecast import NeuralForecast

from neuralforecast.losses.pytorch import MAE, RMSE
from neuralforecast.losses.numpy import mae, mse

import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from typing import Optional, Union, Tuple


HORIZON = 90

import gc
gc.collect()

class HuberLoss(torch.nn.Module):
    """Huber Loss

    The Huber loss, employed in robust regression, is a loss function that
    exhibits reduced sensitivity to outliers in data when compared to the
    squared error loss. This function is also refered as SmoothL1.

    The Huber loss function is quadratic for small errors and linear for large
    errors, with equal values and slopes of the different sections at the two
    points where $(y_{\\tau}-\hat{y}_{\\tau})^{2}$=$|y_{\\tau}-\hat{y}_{\\tau}|$.

    $$ L_{\delta}(y_{\\tau},\; \hat{y}_{\\tau})
    =\\begin{cases}{\\frac{1}{2}}(y_{\\tau}-\hat{y}_{\\tau})^{2}\;{\\text{for }}|y_{\\tau}-\hat{y}_{\\tau}|\leq \delta \\\
    \\delta \ \cdot \left(|y_{\\tau}-\hat{y}_{\\tau}|-{\\frac {1}{2}}\delta \\right),\;{\\text{otherwise.}}\end{cases}$$

    where $\\delta$ is a threshold parameter that determines the point at which the loss transitions from quadratic to linear,
    and can be tuned to control the trade-off between robustness and accuracy in the predictions.

    **Parameters:**<br>
    `delta`: float=1.0, Specifies the threshold at which to change between delta-scaled L1 and L2 loss.

    **References:**<br>
    [Huber Peter, J (1964). "Robust Estimation of a Location Parameter". Annals of Statistics](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full)
    """

    def __init__(self, delta: float = 1.0):
        super(HuberLoss, self).__init__()
        self.outputsize_multiplier = 1
        self.delta = delta
        self.output_names = [""]
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        """
        Univariate loss operates in dimension [B,T,H]/[B,H]
        This changes the network's output from [B,H,1]->[B,H]
        """
        return y_hat.squeeze(-1)

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        """
        **Parameters:**<br>
        `y`: tensor, Actual values.<br>
        `y_hat`: tensor, Predicted values.<br>
        `mask`: tensor, Specifies date stamps per serie to consider in loss.<br>

        **Returns:**<br>
        `huber_loss`: tensor (single value).
        """
        if mask is None:
            mask = torch.ones_like(y_hat)

        huber_loss = F.huber_loss(
            y * mask, y_hat * mask, reduction="mean", delta=self.delta
        )
        return huber_loss



DEFAULT_HP_NBEATSx = {
                "h": HORIZON,
                "input_size": HORIZON,
                "loss": HuberLoss(),
                "max_steps": 1000,
                "stack_types": ['trend', 'seasonality', 'identity'],
                "n_polynomials": 4,
                "n_blocks": [1, 1, 1],
                "mlp_units": [[512, 512],[512, 512], [512, 512]],
                "val_check_steps": 10,
                "step_size": HORIZON,
                'alias': 'NBEATSx'
}

DEFAULT_HP_NHITS = {
                "h": HORIZON,
                "loss": HuberLoss(),
                "input_size": HORIZON,
                "step_size": HORIZON,
                "alias": 'NHITS'
}


DEFAULT_AUTO_NBEATSx = {
    "input_size": tune.qrandint(HORIZON, 3 * HORIZON, HORIZON),
    "n_harmonics": tune.qrandint(1, 4, 1),
    "n_polynomials": tune.qrandint(1, 10, 1),
    "learning_rate": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.qrandint(32, 512, 32),
}

DEFAULT_AUTO_PATCHTST = {
    "input_size": tune.qrandint(HORIZON, 3 * HORIZON, HORIZON),
    "encoder_layers": tune.qrandint(2, 4, 1),
    "batch_normalization": tune.choice([True, False]),
    "learning_rate": tune.loguniform(1e-5, 1e-1),
    "batch_size": tune.qrandint(32, 512, 32),
    "res_attention": tune.choice([True, False]),
    "linear_hidden_size": tune.qrandint(32, 256, 32),
    "stride": tune.qrandint(4, 32, 4),
}

 

def load_data(stock_price_data_path = "raw_data/spx_stock_prices.parquet"):
    prices = pd.read_parquet(stock_price_data_path)
    # create an daterange 
    date_range = pd.date_range(start=min(prices.index), end=max(prices.index), freq='B')
    prices = prices.reindex(date_range, method='ffill')
    # replace zeros with nan
    prices = prices.replace(0, np.nan)
    # name the index
    

def train_model(train_dataset):
    models = [NBEATSx(**DEFAULT_HP_NBEATSx), NHITS(**DEFAULT_HP_NHITS)]
    nf = NeuralForecast(models=models, freq='B')
    nf.fit(df=train_dataset)
    return nf

def train_auto_model(train_dataset):
    models = [AutoNBEATSx(h = HORIZON, 
                          loss = HuberLoss(),
                          config = DEFAULT_AUTO_NBEATSx,
                          num_samples=10)] 
    # , 
    #         AutoPatchTST(h=HORIZON,
    #                      loss=MAE(),
    #                      config=DEFAULT_AUTO_PATCHTST)
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
    return first_last_date_df[
        (
            first_last_date_df['first_date']
            < pd.to_datetime(split_date) - pd.Timedelta(days=180)
        )
        & (
            first_last_date_df['last_date']
            > pd.to_datetime(split_date) + pd.Timedelta(days=90)
        )
    ]['unique_id'].unique().tolist()


if __name__ == "__main__":
    melted_df = load_data()

    split_date = '2018-01-01'
    valid_tickers_numb = retrieve_valid_tickers(melted_df, split_date)
    
    # split the entires in valid ticker_numbs by the _ character
    valid_tickers = [x.split('_')[0] for x in valid_tickers_numb]
    valid_ticker_ids = [x.split('_')[1] for x in valid_tickers_numb]
    
    # create a dataframe with the valid tickers and their ids
    valid_tickers_df = pd.DataFrame({'unique_id': valid_tickers, 'unique_id_num': valid_ticker_ids})

    # save the valid tickers to a csv file
    pd.DataFrame(valid_tickers_df).to_csv(f'raw_data/valid_tickers_{split_date}.csv', index=False)

    valid_ticker_df = melted_df[melted_df['unique_id'].isin(valid_tickers_numb)]
    Y_train_df = valid_ticker_df[valid_ticker_df['ds'] < split_date]
    nf = train_model(Y_train_df)
    nf.save(path='./nf_models/',
        model_index=None, 
        overwrite=True,
        save_dataset=True)
    nf2 = NeuralForecast.load(path='./models/')

    forecast_date = '2019-03-02'
    other_valid_tickers = retrieve_valid_tickers(valid_ticker_df, forecast_date)
    # Select 10 random tickers
    ten_tickers = np.random.choice(other_valid_tickers, 10)

    test_input = valid_ticker_df[(valid_ticker_df['unique_id'].isin(ten_tickers)) & (valid_ticker_df['ds'] <= forecast_date)]
    print(test_input.head())
    fcst = nf2.predict(df=test_input).reset_index()
    print(fcst.head())
    #rename the nbeatsx column to yhat
    fcst = fcst.rename(columns={'PatchTST': 'yhat_patchtst', 'NBEATSx': 'yhat_nbeatsx', 'NHITS': 'yhat_nhits'})
    fcst = fcst.merge(valid_ticker_df, on=['unique_id', 'ds'], how='left')
    r2_scores = []
    mape_scores = []
    spearman_scores = []
    fig, ax = plt.subplots(nrows = 5, ncols = 2, figsize=(15, 10))
    for i, ticker in enumerate(ten_tickers):
        ticker_data = fcst[fcst['unique_id'] == ticker]
        ax[i//2, i%2].plot(ticker_data['ds'], ticker_data['y'], label='actual')
        ax[i//2, i%2].plot(ticker_data['ds'], ticker_data['yhat_nbeatsx'], label='forecast')
        ax[i//2, i%2].legend()
        ax[i//2, i%2].set_title(ticker)
        r2_scores.append(r2_score(ticker_data['y'], ticker_data['yhat_nbeatsx']))
        mape_scores.append(mean_absolute_percentage_error(ticker_data['y'], ticker_data['yhat_nbeatsx']))
        # calculate spearman correlation
        spearman_scores.append(ticker_data[['y', 'yhat_nbeatsx']].corr(method='spearman').iloc[0, 1])
    print("nbeatsx--------------------------------")
    print(np.mean(r2_scores))
    print(np.mean(mape_scores))
    print(np.mean(spearman_scores))
    # r2_scores = []
    # mape_scores = []
    # spearman_scores = []
    # fig1, ax1 = plt.subplots(nrows = 5, ncols = 2, figsize=(15, 10))
    # for i, ticker in enumerate(ten_tickers):
    #     ticker_data = fcst[fcst['unique_id'] == ticker]
    #     ax1[i//2, i%2].plot(ticker_data['ds'], ticker_data['y'], label='actual')
    #     ax1[i//2, i%2].plot(ticker_data['ds'], ticker_data['yhat_patchtst'], label='forecast')
    #     ax1[i//2, i%2].legend()
    #     ax1[i//2, i%2].set_title(ticker)
    #     r2_scores.append(r2_score(ticker_data['y'], ticker_data['yhat_patchtst']))
    #     mape_scores.append(mean_absolute_percentage_error(ticker_data['y'], ticker_data['yhat_patchtst']))
    #     # calculate spearman correlation
    #     spearman_scores.append(ticker_data[['y', 'yhat_patchtst']].corr(method='spearman').iloc[0, 1])
    # print("patchtst--------------------------------")
    # print(np.mean(r2_scores))
    # print(np.mean(mape_scores))
    # print(np.mean(spearman_scores))
    plt.tight_layout()
    plt.show()
