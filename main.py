from objects import *
import pickle
import os
import plotly.express as px

# Write if name == main
if __name__ == "__main__":
    # check if the directory exists
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
    # check if experiments is empty
    if len(os.listdir('experiments')) == 0:
        prices = pd.read_parquet("raw_data/spx_stock_prices.parquet")
        treasury_rate_files = ["daily-treasury-rates.csv"] + [f"daily-treasury-rates ({i}).csv" for i in range(1, 25)]
        rates_df = [pd.read_csv(f"raw_data/{file}", index_col=0) for file in treasury_rate_files]
        rates_df = pd.concat(rates_df)
        rates_df.index = pd.to_datetime(rates_df.index)
        # sort rates_df by date
        rates_df = rates_df.sort_index()
        print("Start Generation")
        generate_experiments(prices, rates_df, 250, "experiments", lookback=48, error_max=10)

    if not os.path.exists('results'):
        os.makedirs('results')
    policies = [DIRECTIONAL_TRADING] + \
               [DIRECTIONAL_INCENTIVE_TRADING(0), DIRECTIONAL_INCENTIVE_TRADING(.25), DIRECTIONAL_INCENTIVE_TRADING(.5),
                DIRECTIONAL_INCENTIVE_TRADING(.75), DIRECTIONAL_INCENTIVE_TRADING(5)] + \
               [NAIVE]
    multi_sim = MultiSimRunner('experiments', policies)
    multi_sim.get_results("results/DI_with_ve.csv")
