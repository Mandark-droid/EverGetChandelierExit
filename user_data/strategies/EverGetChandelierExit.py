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

#from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
#                                IntParameter, IStrategy, merge_informative_pair)

from freqtrade.strategy import IStrategy,merge_informative_pair
from sqlalchemy.ext.declarative import declarative_base
# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class EverGetChandelierExit(IStrategy):
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

    # Optimal timeframe for the strategy.
    timeframe = '1h'

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 100.0,
        "60": 0.12,
        "120": 0.07,
        "180": 0.03,
        "240": 0.01,
        "300": 0.008
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.99

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.05  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = False

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 100

    # Strategy parameters
    atr_period = 22
    atr_multiplier = 3.0
    showLabels = True
    useClose = True
    highlightState = True

    ######
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
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'tema': {},
                'sar': {'color': 'white'},
            },
            'subplots': {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    'macd': {'color': 'blue'},
                    'macdsignal': {'color': 'orange'},
                },
                "RSI": {
                    'rsi': {'color': 'red'},
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
        longStop = close.shift(1).where(close.shift(1) > longStopPrev, longStop)

        shortStop = (low.rolling(self.atr_period).min() if self.useClose else low.rolling(self.atr_period).apply(
            lambda x: x[:-1].min())) + atr
        shortStopPrev = shortStop.shift(1).fillna(shortStop)
        shortStop = close.shift(1).where(close.shift(1) < shortStopPrev, shortStop)

        # dir = close.apply(lambda x: 1 if x > shortStopPrev.iloc[-1] else -1 if x < longStopPrev.iloc[-1] else dir[-1])
        dataframe['dir'] = 1
        dataframe.loc[dataframe['close'] <= longStopPrev, 'dir'] = -1
        dataframe.loc[dataframe['close'] > shortStopPrev, 'dir'] = 1

        longColor = 'green'
        shortColor = 'red'

        longStopPlot = longStop.where(dataframe['dir'] == 1, None)
        buySignal = (dataframe['dir'] == 1) & (dataframe['dir'].shift(1) == -1)
        buySignalPlot = longStop.where(buySignal, None)
        buyLabel = pd.Series(['Buy' if x else '' for x in buySignal]).where(self.showLabels & buySignal, None).any()

        shortStopPlot = shortStop.where(dataframe['dir'] == -1, None)
        sellSignal = (dataframe['dir'] == -1) & (dataframe['dir'].shift(1) == 1)
        sellSignalPlot = shortStop.where(sellSignal, None)
        sellLabel = pd.Series(['Sell' if x else '' for x in sellSignal]).where(self.showLabels & sellSignal, None).any()

        midPricePlot = close

        longFillColor = longColor if self.highlightState and (dataframe['dir'] == 1).any() else None
        shortFillColor = shortColor if self.highlightState and (dataframe['dir'] == -1).any() else None
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

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_25'] = ta.RSI(dataframe, timeperiod=25)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe["bb_percent"] = (
                (dataframe["close"] - dataframe["bb_lowerband"]) /
                (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )
        dataframe["bb_width"] = (
                (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe["bb_middleband"]
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
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        # dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        dataframe['ema30'] = ta.EMA(dataframe, timeperiod=30)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # # SMA - Simple Moving Average
        dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
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
        dataframe.loc[
            (
                    dataframe['dir'] == 1) &
            (dataframe['dir'].shift(1) == -1) &
            (dataframe['rsi_25'] > 30) & (dataframe['close'] > dataframe['bb_middleband']) & (dataframe['rsi_25'] < 70)
            & (dataframe['close'] > dataframe['sma3'])
            & (dataframe['close'] > dataframe['ema3'])
            ,
            'buy'
        ] = 1
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
        dataframe.loc[
            (dataframe['dir'] == -1) &
            (dataframe['dir'].shift(1) == 1)
                # &(dataframe['close'] < dataframe['bb_middleband']) &
                # & (dataframe['rsi_25'] > 60)

            ,
            'sell'
        ] = 1
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
