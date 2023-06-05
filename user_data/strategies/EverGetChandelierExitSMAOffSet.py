# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union
# from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
#                                IntParameter, IStrategy, merge_informative_pair)
from freqtrade.strategy import IStrategy, merge_informative_pair
from sqlalchemy.ext.declarative import declarative_base
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib
import sys
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, \
    CategoricalParameter
import technical.indicators as ftt
import math
import logging
from functools import reduce


def EWO(dataframe, ema_length=5, ema2_length=35):
    # df = dataframe.copy()
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    emadif = (ema1 - ema2) / dataframe['close'] * 100
    return emadif


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolate on the scale given by a to b, using t as the point on that scale.
    Examples
    --------
        50 == lerp(0, 100, 0.5)
        4.2 == lerp(1, 5, 0.8)
    """
    return (1 - t) * a + t * b


logger = logging.getLogger(__name__)


class EverGetChandelierExitSMAOffSet(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v1.0.10"

    overbuy_factor = 1.295

    position_adjustment_enable = True
    initial_safety_order_trigger = -0.02
    max_so_multiplier_orig = 3
    safety_order_step_scale = 2
    safety_order_volume_scale = 1.8

    # just for initialization, now we calculate it...
    max_so_multiplier = max_so_multiplier_orig
    # We will store the size of stake of each trade's first order here
    cust_proposed_initial_stakes = {}
    # Amount the strategy should compensate previously partially filled orders for successive safety orders (0.0 - 1.0)
    partial_fill_compensation_scale = 1

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        tag = super().custom_sell(pair, trade, current_time,
                                  current_rate, current_profit, **kwargs)
        if tag:
            return tag

        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        buy_tags = buy_tag.split()

        if current_profit <= -0.35:
            return f'stop_loss ({buy_tag})'

        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        # remove pair from custom initial stake dict only if full exit
        if trade.amount == amount:
            del self.cust_proposed_initial_stakes[pair]
        return True

        # Let unlimited stakes leave funds open for DCA orders

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:
        custom_stake = proposed_stake / self.max_so_multiplier * self.overbuy_factor
        self.cust_proposed_initial_stakes[
            pair] = custom_stake  # Setting of first stake size just before each first order of a trade
        return custom_stake  # set to static 10 to simulate partial fills of 10$, etc

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs) -> Optional[float]:
        if current_profit > self.initial_safety_order_trigger:
            return None

        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        if 1 <= count_of_buys <= self.max_so_multiplier_orig:
            # if (1 <= count_of_buys) and (open_trade_value < self.stake_amount * self.overbuy_factor):
            safety_order_trigger = (
                abs(self.initial_safety_order_trigger) * count_of_buys)
            if self.safety_order_step_scale > 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (
                    abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (
                        math.pow(self.safety_order_step_scale, (count_of_buys - 1)) - 1) / (
                        self.safety_order_step_scale - 1))
            elif self.safety_order_step_scale < 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (
                    abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (
                        1 - math.pow(self.safety_order_step_scale, (count_of_buys - 1))) / (
                        1 - self.safety_order_step_scale))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    # This returns first order actual stake size
                    actual_initial_stake = filled_buys[0].cost

                    # Fallback for when the initial stake was not set for whatever reason
                    stake_amount = actual_initial_stake

                    already_bought = sum(
                        filled_buy.cost for filled_buy in filled_buys)

                    if self.cust_proposed_initial_stakes[trade.pair] > 0:
                        # This calculates the amount of stake that will get used for the current safety order,
                        # including compensation for any partial buys
                        proposed_initial_stake = self.cust_proposed_initial_stakes[trade.pair]
                        current_actual_stake = already_bought * math.pow(self.safety_order_volume_scale,
                                                                         (count_of_buys - 1))
                        current_stake_preposition = proposed_initial_stake * math.pow(self.safety_order_volume_scale,
                                                                                      (count_of_buys - 1))
                        current_stake_preposition_compensation = current_stake_preposition + abs(
                            current_stake_preposition - current_actual_stake)
                        total_so_stake = lerp(current_actual_stake, current_stake_preposition_compensation,
                                              self.partial_fill_compensation_scale)
                        # Set the calculated stake amount
                        stake_amount = total_so_stake
                    else:
                        # Fallback stake amount calculation
                        stake_amount = stake_amount * \
                            math.pow(self.safety_order_volume_scale,
                                     (count_of_buys - 1))

                    amount = stake_amount / current_rate
                    logger.info(
                        f"Initiating safety order buy #{count_of_buys} "
                        f"for {trade.pair} with stake amount of {stake_amount}. "
                        f"which equals {amount}. "
                        f"Previously bought: {already_bought}. "
                        f"Now overall:{already_bought + stake_amount}. ")
                    return stake_amount
                except Exception as exception:
                    logger.info(
                        f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                    # print(f'Error occured while trying to get stake amount for {trade.pair}: {str(exception)}')
                    return None

        return None

        # Modified Buy / Sell params - 20210619
        # Buy hyperspace params:

    buy_params = {
        "base_nb_candles_buy": 16,
        "ewo_high": 5.672,
        "ewo_low": -19.931,
        "low_offset": 0.973,
        "rsi_buy": 59,

    }

    buy_signals = {
        "buy_condition_1_enable": True,
        "buy_condition_2_enable": True,
        "buy_condition_3_enable": True,
        "buy_condition_4_enable": True
    }

    sell_signals = {
        "sell_condition_1_enable": True,
        "sell_condition_2_enable": True,
        "sell_condition_3_enable": False,
        "sell_condition_4_enable": False
    }
    # Sell hyperspace params:
    sell_params = {
        "base_nb_candles_sell": 20,
        "high_offset": 1.010,
    }

    # Optimal timeframe for the strategy.
    timeframe = '5m'
    informative_timeframe = '1h'
    info_timeframes = ['15m', '1h']

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.03,
        "10": 0.024,
        "30": 0.016,
        "40": 0.012,
        "50": 0.0008
    }
    # SMAOffset
    base_nb_candles_buy = IntParameter(
        5, 80, default=buy_params['base_nb_candles_buy'], space='buy', optimize=True)
    base_nb_candles_sell = IntParameter(
        5, 80, default=sell_params['base_nb_candles_sell'], space='sell', optimize=True)
    low_offset = DecimalParameter(
        0.9, 0.99, default=buy_params['low_offset'], space='buy', optimize=True)
    high_offset = DecimalParameter(
        0.99, 1.1, default=sell_params['high_offset'], space='sell', optimize=True)

    # Protection
    fast_ewo = 50
    slow_ewo = 200
    ewo_low = DecimalParameter(-20.0, -8.0,
                               default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(
        2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    rsi_buy = IntParameter(
        30, 70, default=buy_params['rsi_buy'], space='buy', optimize=True)

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.99
    use_custom_stoploss = False

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.01
    trailing_only_offset_is_reached = True

    # Sell signal
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False
    process_only_new_candles = False
    startup_candle_count: int = 576

    # Strategy parameters
    atr_period = 22
    atr_multiplier = 3.0
    showLabels = True
    useClose = True
    highlightState = True
    zlsma_length = 50
    zlsma_offset = 0
    # SuperTrend params
    super_atr_period = 10
    super_atr_multiplier = 3.0
    change_atr = True
    show_signals = True
    highlighting = True
    ######

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    @property
    def plot_config(self):
        return {
            "main_plot":
                {},
            "subplots":
                {
                    "sub":
                        {
                            "rsi":
                                {
                                    "color": "#80dea8",
                                    "type": "line"
                                }
                        },
                    "sub2":
                        {
                            "ma_buy_16":
                                {
                                    "color": "#db1ea2",
                                    "type": "line"
                                },
                            "ma_sell_20":
                                {
                                    "color": "#645825",
                                    "type": "line"
                                },
                            "EWO":
                                {
                                    "color": "#1e5964",
                                    "type": "line"
                                },
                            "missing_data":
                                {
                                    "color": "#26b08d",
                                    "type": "line"
                                }
                        }
                }
        }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe)
                             for pair in pairs]

        return informative_pairs

    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        """
        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        # Calculate all ma_sell values
        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)
        """
        if self.change_atr:
            dataframe['atr'] = ta.ATR(
                dataframe, timeperiod=self.super_atr_period)
        else:
            dataframe['atr'] = ta.SMA(
                dataframe, timeperiod=self.super_atr_period)

            # Calculate Supertrend lines
        dataframe['up'] = dataframe['close'] - \
            (self.super_atr_multiplier * dataframe['atr'])
        dataframe['up1'] = dataframe['up'].shift(1).fillna(dataframe['up'])
        dataframe['up'] = dataframe.apply(lambda x: max(x['up'], x['up1']) if x['close'] > x['up1'] else x['up'],
                                          axis=1)

        dataframe['dn'] = dataframe['close'] + \
            (self.super_atr_multiplier * dataframe['atr'])
        dataframe['dn1'] = dataframe['dn'].shift(1).fillna(dataframe['dn'])
        dataframe['dn'] = dataframe.apply(lambda x: min(x['dn'], x['dn1']) if x['close'] < x['dn1'] else x['dn'],
                                          axis=1)

        # Calculate trend
        dataframe['trend'] = 1
        dataframe['trend'] = dataframe['trend'].fillna(method='ffill')
        dataframe.loc[(dataframe['trend'] == -1) &
                      (dataframe['close'] > dataframe['dn1']), 'trend'] = 1
        dataframe.loc[(dataframe['trend'] == 1) & (
            dataframe['close'] < dataframe['up1']), 'trend'] = -1

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Check for 0 volume candles in the last day
        dataframe['missing_data'] = \
            (dataframe['volume'] <= 0).rolling(
                window=self.startup_candle_count,
                min_periods=self.startup_candle_count).sum()

        # Momentum Indicators
        # ------------------------------------
        high = dataframe['high']  # Replace with your high data
        low = dataframe['low']  # Replace with your low data
        close = dataframe['close']  # Replace with your close data
        # atr = ta.ATR(high, low, close, self.atr_period) * self.atr_multiplier
        atr = ta.ATR(high, low, close, 22) * 3
        longStop = (high.rolling(self.atr_period).max() if self.useClose else high.rolling(self.atr_period).apply(
            lambda x: x[:-1].max())) - atr
        longStopPrev = longStop.shift(1).fillna(longStop)
        longStop = close.shift(1).where(
            close.shift(1) > longStopPrev, longStop)

        shortStop = (low.rolling(self.atr_period).min() if self.useClose else low.rolling(self.atr_period).apply(
            lambda x: x[:-1].min())) + atr
        shortStopPrev = shortStop.shift(1).fillna(shortStop)
        shortStop = close.shift(1).where(
            close.shift(1) < shortStopPrev, shortStop)

        # dir = close.apply(lambda x: 1 if x > shortStopPrev.iloc[-1] else -1 if x < longStopPrev.iloc[-1] else dir[-1])
        dataframe['dir'] = 1
        dataframe.loc[dataframe['close'] <= longStopPrev, 'dir'] = -1
        dataframe.loc[dataframe['close'] > shortStopPrev, 'dir'] = 1

        longColor = 'green'
        shortColor = 'red'

        longStopPlot = longStop.where(dataframe['dir'] == 1, None)
        buySignal = (dataframe['dir'] == 1) & (dataframe['dir'].shift(1) == -1)
        buySignalPlot = longStop.where(buySignal, None)
        buyLabel = pd.Series(['Buy' if x else '' for x in buySignal]).where(
            self.showLabels & buySignal, None).any()

        shortStopPlot = shortStop.where(dataframe['dir'] == -1, None)
        sellSignal = (dataframe['dir'] == -
                      1) & (dataframe['dir'].shift(1) == 1)
        sellSignalPlot = shortStop.where(sellSignal, None)
        sellLabel = pd.Series(['Sell' if x else '' for x in sellSignal]).where(
            self.showLabels & sellSignal, None).any()

        midPricePlot = close

        longFillColor = longColor if self.highlightState and (
            dataframe['dir'] == 1).any() else None
        shortFillColor = shortColor if self.highlightState and (
            dataframe['dir'] == -1).any() else None
        # fill = lambda x, y, color: plt.fill_between(x.index, y, x, where=y < x, interpolate=True, color=color)
        # fig, ax = plt.subplots()
        # fill(midPricePlot, longStopPlot, longFillColor)
        # fill(midPricePlot, shortStopPlot, shortFillColor)
        # ax.plot(midPricePlot.index, midPricePlot.values)
        # ax.plot(longStopPlot.index, longStopPlot.values, color=longColor)
        # ax.scatter(buySignalPlot.index, buySignalPlot.values, color=longColor, marker='o', s=10)
        # ax.text(buySignalPlot.index, buySignalPlot.values, buyLabel, color='white', fontsize=8, ha='center',
        #        va='center')
        # ax.plot(shortStopPlot.index, shortStopPlot.values, color=shortColor)
        # ax.scatter(sellSignalPlot.index, sellSignalPlot.values, color=shortColor, marker='o', s=10)
        # ax.text(sellSignalPlot.index, sellSignalPlot.values, sellLabel, color='white', fontsize=8, ha='center',
        #        va='center')
        # plt.show()
        # Calculate the Linear Regression
        lsma = pta.linreg(
            dataframe['close'], length=self.zlsma_length, offset=self.zlsma_offset)
        lsma2 = pta.linreg(lsma, length=self.zlsma_length,
                           offset=self.zlsma_offset)

        # Calculate the Zero Lag SMA
        eq = lsma - lsma2
        zlsma = lsma + eq
        dataframe['zlsma'] = zlsma
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_25'] = ta.RSI(dataframe, timeperiod=25)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"]
             ) / dataframe["bb_middleband"]
        )

        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']

        # MACD
        # macd = ta.MACD(dataframe)
        # dataframe['macd'] = macd['macd']
        # dataframe['macdsignal'] = macd['macdsignal']
        # dataframe['macdhist'] = macd['macdhist']

        # # EMA - Exponential Moving Average
        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema7'] = ta.EMA(dataframe, timeperiod=7)
        dataframe['ema12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema30'] = ta.EMA(dataframe, timeperiod=30)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # # SMA - Simple Moving Average
        dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['sma7'] = ta.SMA(dataframe, timeperiod=7)
        # # dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
        dataframe['sma30'] = ta.SMA(dataframe, timeperiod=30)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        # dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] = \
            ta.EMA(dataframe, timeperiod=self.base_nb_candles_buy.value)
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        for buy_enable in self.buy_signals:
            #print(buy_enable)
            index = int(buy_enable.split('_')[2])
            #print(index)
            # item_buy_protection_list = [True]
            if self.buy_signals[f'{buy_enable}']:
                item_buy_logic = []
                if index == 1:
                    item_buy_logic.append(
                        (dataframe['close'] < (
                         dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                        (dataframe['EWO'] > self.ewo_high.value) &
                        (dataframe['rsi'] < self.rsi_buy.value) &
                        (dataframe['missing_data'] < 1)
                    )
                if index == 2:
                    item_buy_logic.append(
                        (dataframe['close'] < (
                            dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)) &
                        (dataframe['EWO'] < self.ewo_low.value) &
                        (dataframe['missing_data'] < 1)
                    )
                    # Buy 3: Chandelier exit
                if index == 3:
                    item_buy_logic.append((dataframe['dir'] == 1)
                                          & (dataframe['dir'].shift(1) == -1)
                                           & (dataframe['rsi_25'] > 30)
                                           & (dataframe['rsi_25'] < 70)
                                           & (dataframe['close'] > dataframe['zlsma'])
                                           & (dataframe['ema26'] > (dataframe['ema12']))
                                           & (dataframe['ema26'].shift() - dataframe['ema12'].shift()) > (dataframe['open'] / 100)
                                          )
                    # Buy 4: SuperTrend
                if index == 4:
                    item_buy_logic.append((dataframe['trend'] == 1) & (dataframe['trend'].shift(1) == -1)
                                          & (dataframe['rsi_25'] > 24)
                                          & (dataframe['rsi_25'] < 70)
                                          & (dataframe['close'] > dataframe['zlsma'])
                                          & (dataframe['ema21'] > (dataframe['ema7'])))

                item_buy_logic.append(dataframe['volume'] > 0)
                item_buy = reduce(lambda x, y: x & y, item_buy_logic)
                dataframe.loc[item_buy, 'enter_tag'] += f"{index} "
                conditions.append(item_buy)
                dataframe.loc[:, 'enter_long'] = item_buy
        if conditions:
            dataframe.loc[:, 'enter_long'] = reduce(
                lambda x, y: x | y, conditions)

        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value)) &  # Signal: RSI crosses above sell_rsi
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'enter_short'] = 1
        """

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        # Speed optimization for dry / live runs, not looping through for ... values with it, nothing else.
        dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] = \
            ta.EMA(dataframe, timeperiod=self.base_nb_candles_sell.value)

        conditions = []

#        conditions.append(
#            (
#                    (dataframe['close'] > (
#                            dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
#                    (dataframe['volume'] > 0)
#            )
#        )
        dataframe.loc[:, 'exit_tag'] = ''
        for sell_enable in self.sell_signals:
            # print(buy_enable)
            index = int(sell_enable.split('_')[2])
            # print(index)
            # item_sell_protection_list = [True]
            if self.sell_signals[f'{sell_enable}']:
                item_sell_logic = []
                # Sell 1: SMAOffSet
                if index == 1:
                    item_sell_logic.append(
                        (
                            (dataframe['close'] > (
                                dataframe[
                                    f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value)) &
                            (dataframe['volume'] > 0)
                        )
                    )
                    # Sell 2: SMAOffSet
                if index == 2:
                    item_sell_logic.append(
                        (
                            (dataframe['close'] > (
                                dataframe[
                                    f'ma_sell_{self.base_nb_candles_sell.value}'] * self.low_offset.value)) &
                            (dataframe['volume'] > 0)
                        )
                    )

                # Sell 3: Chandelier Exit
                if index == 3:
                    item_sell_logic.append(
                        (dataframe['dir'] == -1) & (dataframe['dir'].shift(1) == 1))
                    # Sell 4: SuperTrend
                if index == 4:
                    item_sell_logic.append(
                        (dataframe['trend'] == -1) & (dataframe['trend'].shift(1) == 1))

                item_sell_logic.append(dataframe['volume'] > 0)
                item_sell = reduce(lambda x, y: x & y, item_sell_logic)
                dataframe.loc[item_sell, 'exit_tag'] += f"{index} "
                conditions.append(item_sell)
                dataframe.loc[:, 'exit_long'] = item_sell
        if conditions:
            dataframe.loc[:, 'exit_long'] = reduce(
                lambda x, y: x | y, conditions)

        # Uncomment to use shorts (Only used in futures/margin mode. Check the documentation for more info)
        """
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value)) &  # Signal: RSI crosses above buy_rsi
                (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            'exit_short'] = 1
        """
        return dataframe
