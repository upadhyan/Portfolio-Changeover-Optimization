import gurobipy as gp
from gurobipy import GRB
from abc import ABC, abstractmethod
from src.experimental_config import *
from time import time
import itertools


class TradingPolicy(ABC):
    """Abstract class for trading policies. All trading policies must inherit from this class."""

    def __init__(self, experiment: ExperimentInfo, verbose=False, **kwargs):
        """Generate a Trading Policy.

        Args:
            experiment (ExperimentInfo): pre-generated experiment using experimental_config.py
            verbose (bool, optional): Defaults to False.
        """
        self.exp = experiment
        self.verbose = verbose

    @abstractmethod
    def get_trades(self, portfolio, t, price_data):
        pass

    def vprint(self, print_statement):
        if self.verbose:
            print(print_statement)


class DirectionalTradingPolicy(TradingPolicy):
    """This policy restrics trades in the direction of the final portfolio. It uses a constraint to enforce this."""

    def __init__(self, experiment: ExperimentInfo, verbose=True, **kwargs):
        """Generate a Directional Trading Policy.

        Args:
            experiment (ExperimentInfo): pre-generated experiment using experimental_config.py
            verbose (bool, optional): Defaults to True.
        """
        super().__init__(experiment, verbose, **kwargs)
        self.F = np.ones(self.exp.initial_portfolio.values[:-1].shape) * self.exp.tx_cost

    def get_trades(self, portfolio, t, price_data):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        env.setParam("TimeLimit", 300)
        env.start()
        #######################
        p = portfolio.values[:-1]  # portfolio of number of positions
        current_cash = portfolio.values[-1]  # cash amount

        value = price_data.loc[t] @ p + current_cash
        assert value > 0.0
        self.vprint(f"Current Portfolio Value at {t}: {value}")
        prob_arr = []
        z_arr = []
        cash_arr = []
        p_arr = []

        ## Direction Definition
        initial_portfolio = self.exp.initial_portfolio.copy()[:-1]
        final = self.exp.target_portfolio.copy()[:-1]

        difference = final - initial_portfolio

        buy_constraint = (difference >= 0) * 1
        F = self.F

        returns = price_data.pct_change().fillna(0) + 1

        port_value_bounds = value * returns.max(axis=1).cumprod()

        p_next = None
        m = gp.Model(env=env)
        time_steps = price_data.index
        n_timesteps = len(time_steps)
        cash = current_cash
        ### Generate all possible trade matrices -> store it in a matrix list of size len(trading_info)
        for time_step in time_steps:
            prices = price_data.loc[time_step].values

            z_buy = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            z_sell = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            y_sell = m.addMVar(p.shape, vtype=GRB.BINARY)
            y_buy = m.addMVar(p.shape, vtype=GRB.BINARY)
            ## lambda_it_sell = admvar(L(## of choice), vtype = binary )
            ## lambda_it_buy = admvar(L(## of choice), vtype = binary )
            ## m.add constr (sum lambdas  == 1)

            ## Directional Constraints
            bound_at_time = port_value_bounds.loc[time_step]
            M = np.ceil(bound_at_time / prices)
            # m.addConstr(z_buy <= buy_constraint.values * M)
            # m.addConstr(z_sell <= (1 - buy_constraint.values) * M)
            m.addConstr(y_buy <= buy_constraint.values)
            m.addConstr(y_sell <= (1 - buy_constraint.values))

            # pull relevant vectors
            # vectors_buy = [buy_matrix[:,timestep] for matrix in buy_matrix_list]
            # vectors_buy = [sell_matrix[:,timestep] for matrix in sell_matrix_list]

            # Next Portfolio
            p_next = p + z_buy - z_sell
            # y_sell = lambda_it_sell^T @ vectors_sell
            # y_buy = lambda_it_buy^T @ vectors_buy
            cash_next = prices @ (z_sell - z_buy) - F @ y_sell - F @ y_buy + cash

            ## Trading fees
            ### if we sell a stock, we pay a trading price
            m.addConstr(M * y_sell >= z_sell)

            ## If we buy a stock, we pay a trading fee
            m.addConstr(M * y_buy >= z_buy)

            ## No borrowing
            m.addConstr(cash_next >= 0)

            ## No shorting
            m.addConstr(p_next >= 0)

            prob_arr.append(-F @ y_sell - F @ y_buy)
            cash = cash_next
            p = p_next

            z_arr.append(z_buy - z_sell)
            cash_arr.append(cash_next)
            p_arr.append(p_next)

        final_p = self.exp.target_portfolio.values[:-1]
        # Terminal constraint.
        m.addConstr(p_next >= final_p, name="terminal")

        prob_arr.append(prices @ p_next + cash_next)

        # Combine all time instances
        obj = sum(prob_arr)

        m.setObjective(obj, GRB.MAXIMIZE)
        m.update()
        # get model constraints
        self.vprint(
            f"\t Optimizing with {n_timesteps} time steps, "
            f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        )
        t1 = time()
        m.optimize()
        t2 = time()
        self.vprint(
            f"\t Optimized. Time taken: {t2 - t1}",
        )
        try:
            self.p_vals = [p.getValue() for p in p_arr]
            self.cash_vals = [_cash.getValue() for _cash in cash_arr]
            self.z_vals = [z.getValue() for z in z_arr]
        except Exception as e:
            self.vprint(e)
            return (
                pd.Series(index=portfolio.index, data=0, name=t),
                price_data.loc[t] @ portfolio[:-1] + portfolio[-1],
            )
        assert (np.round(self.p_vals[0]) >= 0).all()
        del m
        del env
        gc.collect()
        return (
            pd.Series(
                index=portfolio.index, data=(np.append(self.z_vals[0], self.cash_vals[0] - current_cash)), name=t
            ),
            value,
        )


class ColumnGenerationPolicy(TradingPolicy):
    """WIP: This policy uses column generation to solve the problem. It enumerates feasible directional trades
    and uses a column generation approach to select the best trades to execute.

    """

    def __init__(self, experiment: ExperimentInfo, verbose=True, sell_switch=True, **kwargs):
        """Generate a Column Generation Trading Policy.

        Args:
            experiment (ExperimentInfo): pre-generated experiment using experimental_config.py
            verbose (bool, optional): Defaults to True.
            sell_switch (bool, optional): When true, it will apply Column Generation method on sell trades. Defaults to True.
        """
        super().__init__(experiment, verbose, **kwargs)
        self.F = np.ones(self.exp.initial_portfolio.values[:-1].shape) * self.exp.tx_cost
        self.sell_switch = sell_switch

    def get_trades(self, portfolio, t, price_data):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        env.setParam("TimeLimit", 300)
        env.start()
        #######################
        p = portfolio.values[:-1]  # portfolio of number of positions
        current_cash = portfolio.values[-1]  # cash amount

        value = price_data.loc[t] @ p + current_cash
        assert value > 0.0
        self.vprint(f"Current Portfolio Value at {t}: {value}")
        prob_arr = []
        z_arr = []
        cash_arr = []
        p_arr = []

        ## Direction Definition
        initial_portfolio = self.exp.initial_portfolio.copy()[:-1]
        final = self.exp.target_portfolio.copy()[:-1]

        difference = final - initial_portfolio

        buy_constraint = (difference >= 0) * 1
        F = self.F

        returns = price_data.pct_change().fillna(0) + 1

        port_value_bounds = value * returns.max(axis=1).cumprod()

        p_next = None
        m = gp.Model(env=env)
        time_steps = price_data.index
        n_timesteps = len(time_steps)
        cash = current_cash
        buy_enumerations = self.get_possible_enumerations(portfolio, price_data, mode="buy")  # [LxNxT]
        lambda_buy = m.addMVar((buy_enumerations.shape[0],), vtype=GRB.BINARY)

        if self.sell_switch:
            sell_enumerations = self.get_possible_enumerations(portfolio, price_data, mode="sell")  # [KxNxT]
            lambda_sell = m.addMVar((sell_enumerations.shape[0]), vtype=GRB.BINARY)
            m.addConstr(sum(lambda_sell) == 1)
        m.addConstr(sum(lambda_buy) == 1)

        for i, time_step in enumerate(time_steps):
            prices = price_data.loc[time_step].values

            z_buy = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            z_sell = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)

            if not self.sell_switch:
                y_sell = m.addMVar(p.shape, vtype=GRB.BINARY)
            # y_buy = m.addMVar(p.shape, vtype=GRB.BINARY)

            vectors_buy = buy_enumerations[:, :, i]  # [LxN]
            y_buy = lambda_buy @ vectors_buy  # [1xL] @ [LxN] = [1xN]

            if self.sell_switch:
                vectors_sell = sell_enumerations[:, :, i]  # [KxN]
                y_sell = lambda_sell @ vectors_sell  # [1xK] @ [KxN] = [1xN]

            ## Directional Constraints
            bound_at_time = port_value_bounds.loc[time_step]
            M = np.ceil(bound_at_time / prices)
            m.addConstr(y_buy <= buy_constraint.values)
            m.addConstr(y_sell <= (1 - buy_constraint.values))

            # Next Portfolio
            p_next = p + z_buy - z_sell
            cash_next = prices @ (z_sell - z_buy) - F @ y_sell - F @ y_buy + cash

            ## Trading fees
            ### if we sell a stock, we pay a trading price
            m.addConstr(M * y_sell >= z_sell)

            ## If we buy a stock, we pay a trading fee
            m.addConstr(M * y_buy >= z_buy)

            ## No borrowing
            m.addConstr(cash_next >= 0)

            ## No shorting
            m.addConstr(p_next >= 0)

            prob_arr.append(-F @ y_sell - F @ y_buy)
            cash = cash_next
            p = p_next

            z_arr.append(z_buy - z_sell)
            cash_arr.append(cash_next)
            p_arr.append(p_next)

        final_p = self.exp.target_portfolio.values[:-1]
        # Terminal constraint.
        m.addConstr(p_next >= final_p, name="terminal")

        prob_arr.append(prices @ p_next + cash_next)

        # Combine all time instances
        obj = sum(prob_arr)

        m.setObjective(obj, GRB.MAXIMIZE)
        m.update()
        # get model constraints
        self.vprint(
            f"\t Optimizing with {n_timesteps} time steps, "
            f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        )
        t1 = time()
        m.optimize()
        t2 = time()
        self.vprint(
            f"\t Optimized. Time taken: {t2 - t1}",
        )
        try:
            self.p_vals = [p.getValue() for p in p_arr]
            self.cash_vals = [_cash.getValue() for _cash in cash_arr]
            self.z_vals = [z.getValue() for z in z_arr]
            self.lambda_buy = np.argwhere(lambda_buy.X >= 1)
            self.strat_b = buy_enumerations[self.lambda_buy, :, :][0][0]
            if self.sell_switch:
                self.lambda_sell = np.argwhere(lambda_sell.X >= 1)
                self.strat_s = sell_enumerations[self.lambda_sell, :, :][0][0]
        except Exception as e:
            self.vprint(e)
            return (
                pd.Series(index=portfolio.index, data=0, name=t),
                price_data.loc[t] @ portfolio[:-1] + portfolio[-1],
            )
        assert (np.round(self.p_vals[0]) >= 0).all()
        del m
        del env
        gc.collect()
        return (
            pd.Series(
                index=portfolio.index, data=(np.append(self.z_vals[0], self.cash_vals[0] - current_cash)), name=t
            ),
            value,
        )

    def get_possible_enumerations(self, portfolio, trading_information, mode="buy"):
        # Get the current portfolio
        p = portfolio.values[:-1]
        current_cash = portfolio.values[-1]
        p_final = self.exp.target_portfolio.values[:-1]
        # prices = trading_information.loc[time_step].values

        p_diff = p_final - p
        if mode == "buy":
            p_tx = np.where(p_diff > 0, 1, 0)
        if mode == "sell":
            p_tx = np.where(p_diff < 0, 1, 0)
        B = []
        iter_combinations = self.get_combinations(p_tx, len(trading_information.index))

        for c in iter_combinations:
            B.append(np.array(c))

        B = np.array(B)

        return B

    def get_combinations(self, p, t):
        relevant_matrices = [np.zeros(t) if i == 0 else np.identity(t) for i in p]
        combinations = []
        for i, vec in enumerate(p):
            if vec == 0:
                asset_comb = [np.zeros(t)]
            else:
                asset_comb = [relevant_matrices[i][j] for j in range(t)]
            combinations.append(asset_comb)
        iter_combinations = itertools.product(*combinations)

        return iter_combinations


class DirectionalPenaltyTradingPolicy(TradingPolicy):
    """This policy is a modification of the Directional Trading Policy. Instead of a constraint on trading in one direction only,
    it uses a penalty term in the objective function to penalize trades that are not in the direction of the final portfolio.
    """

    def __init__(self, experiment: ExperimentInfo, verbose=True, lambda_=0.5, **kwargs):
        """Generate a Directional Trading Policy with a penalty term in the objective function.

        Args:
            experiment (ExperimentInfo): pre-generated experiment using experimental_config.py
            verbose (bool, optional): Defaults to True.
            lambda_ (float, optional): Penalty importance in the objective. Defaults to 0.5.
        """
        super().__init__(experiment, verbose, **kwargs)
        self.F = np.ones(self.exp.initial_portfolio.values[:-1].shape) * self.exp.tx_cost
        self.lambda_ = lambda_

    def get_trades(self, portfolio, t):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        env.setParam("TimeLimit", 300)
        env.start()
        #######################
        p = portfolio.values[:-1]  # portfolio of number of positions
        current_cash = portfolio.values[-1]  # cash amount

        value = self.known_dict[t] @ p + current_cash
        assert value > 0.0
        previous_time_string = self.known_dict[t].name.strftime("%Y-%m-%d")
        self.vprint(f"Current Portfolio Value at {previous_time_string}: {value}")
        prob_arr = []
        z_arr = []
        cash_arr = []
        p_arr = []

        ## Direction Definition
        initial_portfolio = portfolio.copy()[:-1]
        final = self.exp.target_portfolio.copy()[:-1]

        difference = final - initial_portfolio

        buy_constraint = (difference >= 0) * 1  # 1 means we are allowed to buy, 0 means we are not allowed to buy

        F = self.F
        M = value * 1e3

        trading_information = pd.concat([pd.DataFrame(self.known_dict[t]).T, self.price_dict[t]])
        returns = trading_information.pct_change().fillna(0) + 1

        port_value_bounds = value * returns.max(axis=1).cumprod()
        p_next = None
        m = gp.Model(env=env)
        n_timesteps = len(trading_information.index)
        cash = current_cash
        for time_step in trading_information.index:
            prices = trading_information.loc[time_step].values

            z_buy = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            z_sell = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
            y_sell = m.addMVar(p.shape, vtype=GRB.BINARY)
            y_buy = m.addMVar(p.shape, vtype=GRB.BINARY)

            # Next Portfolio
            p_next = p + z_buy - z_sell
            cash_next = prices @ (z_sell - z_buy) - F @ y_sell - F @ y_buy + cash

            bound_at_time = port_value_bounds.loc[time_step]
            M = np.ceil(bound_at_time / prices)
            ## Trading fees
            ### if we sell a stock, we pay a trading price
            m.addConstr(M * y_sell >= z_sell)

            ## If we buy a stock, we pay a trading fee
            m.addConstr(M * y_buy >= z_buy)

            ## Valid Inequality
            m.addConstr(y_sell + y_buy <= 1)

            ## No borrowing
            m.addConstr(cash_next >= 0)

            ## No shorting
            m.addConstr(p_next >= 0)

            # ## Directional Constraints
            # m.addConstr(z_buy <= buy_constraint.values * M)
            # m.addConstr(z_sell <= (1 - buy_constraint.values) * M)
            # 1 means we are allowed to buy, 0 means we are not allowed to buy

            sell_penalty = (self.lambda_ * buy_constraint.values * prices) @ z_sell
            buy_penalty = (self.lambda_ * (1 - buy_constraint.values) * prices) @ z_buy
            prob_arr.append(-F @ y_sell - F @ y_buy - sell_penalty - buy_penalty)
            cash = cash_next
            p = p_next

            z_arr.append(z_buy - z_sell)
            cash_arr.append(cash_next)
            p_arr.append(p_next)

        final_p = self.exp.target_portfolio.values[:-1]
        # Terminal constraint.
        m.addConstr(p_next >= final_p, name="terminal")

        prob_arr.append(prices @ p_next + cash_next)
        # Combine all time instances
        obj = sum(prob_arr)

        m.setObjective(obj, GRB.MAXIMIZE)
        m.update()
        # get model constraints
        self.vprint(
            f"\t Optimizing with {n_timesteps} time steps, "
            f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        )
        t1 = time()
        m.optimize()
        t2 = time()
        self.vprint(
            f"\t Optimized. Time taken: {t2 - t1}",
        )
        try:
            self.p_vals = [p.getValue() for p in p_arr]
            self.cash_vals = [_cash.getValue() for _cash in cash_arr]
            self.z_vals = [z.getValue() for z in z_arr]
        except Exception as e:
            self.vprint(e)
            return (
                pd.Series(index=portfolio.index, data=0, name=t),
                self.known_dict[t] @ portfolio[:-1] + portfolio[-1],
            )
        assert (np.round(self.p_vals[0]) >= 0).all()
        del m
        del env
        gc.collect()
        return (
            pd.Series(
                index=portfolio.index, data=(np.append(self.z_vals[0], self.cash_vals[0] - current_cash)), name=t
            ),
            value,
        )


class NaivePolicy(TradingPolicy):
    """This policy naively executes all the necessary trades to get to the final portfolio in the immediate time step."""

    def __init__(self, experiment: ExperimentInfo, verbose=True, **kwargs):
        """Naive Policy

        Args:
            experiment (ExperimentInfo): pre-generated experiment using experimental_config.py
            verbose (bool, optional): Defaults to True.
        """
        super().__init__(experiment, verbose, **kwargs)
        self.initial = True
        self.F = np.ones(self.exp.initial_portfolio.values[:-1].shape) * self.exp.tx_cost

    def get_trades(self, portfolio, t):
        if not self.initial:
            return (
                pd.Series(index=portfolio.index, data=0, name=t),
                self.known_dict[t] @ portfolio[:-1] + portfolio[-1],
            )
        self.initial = False

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", False)
        env.start()
        #######################
        p = portfolio.values[:-1]  # portfolio of number of positions
        cash = portfolio.values[-1]  # cash amount

        value = self.known_dict[t] @ p + cash
        assert value > 0.0
        previous_time_string = self.known_dict[t].name.strftime("%Y-%m-%d")
        self.vprint(f"Current Portfolio Value at {previous_time_string}: {value}")

        F = self.F
        M = value * 1e3

        trading_information = pd.concat([pd.DataFrame(self.known_dict[t]).T, self.price_dict[t]])
        m = gp.Model(env=env)

        prices = self.known_dict[t].values
        # prices = trading_information.loc[time_step].values

        z_buy = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
        z_sell = m.addMVar(p.shape, vtype=GRB.INTEGER, lb=0)
        y_sell = m.addMVar(p.shape, vtype=GRB.BINARY)
        y_buy = m.addMVar(p.shape, vtype=GRB.BINARY)

        # Next Portfolio
        p_next = p + z_buy - z_sell
        cash_next = prices @ (z_sell - z_buy) - F @ y_sell - F @ y_buy + cash
        M_buy = np.ceil(value / prices)
        M_sell = p
        ## Trading fees
        ### if we sell a stock, we pay a trading price
        m.addConstr(M_sell * y_sell >= z_sell)

        ## If we buy a stock, we pay a trading fee
        m.addConstr(M_buy * y_buy >= z_buy)

        ## No borrowing
        m.addConstr(cash_next >= 0)

        ## No shorting
        m.addConstr(p_next >= 0)

        obj = prices @ p_next - F @ y_sell - F @ y_buy

        final_p = self.exp.target_portfolio.values[:-1]
        # Terminal constraint.
        m.addConstr(p_next >= final_p)

        m.setObjective(obj, GRB.MAXIMIZE)
        m.update()
        # get model constraints
        # self.vprint(
        #     f"\t Optimizing with {n_timesteps} time steps, "
        #     f"{len(m.getConstrs())} constraints, and {len(m.getVars())} variables..."
        # )
        t1 = time()
        m.optimize()
        t2 = time()
        self.vprint(
            f"\t Optimized. Time taken: {t2 - t1}",
        )
        try:
            self.z = (z_buy - z_sell).getValue()
            self.cash_next = cash_next.getValue()
            self.p_next = p_next.getValue()
        except Exception as e:
            self.vprint(e)
            return (
                pd.Series(index=portfolio.index, data=0, name=t),
                self.known_dict[t] @ portfolio[:-1] + portfolio[-1],
            )
        assert (np.round(self.p_next) >= 0).all()
        del m
        del env
        gc.collect()
        return pd.Series(index=portfolio.index, data=(np.append(self.z, self.cash_next - cash)), name=t), value
