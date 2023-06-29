# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from datetime import datetime
from typing import Optional, Union
#from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
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
import time
log = logging.getLogger(__name__)


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
        return "v1.0.11"

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

        if current_profit <= -0.15:
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
        "sell_condition_4_enable": True
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
        "10": 0.028,
        "30": 0.025,
        "40": 0.018,
        "50": 0.015,
        "60": 0.01,
        "70": 0.005
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

    # Trailing stoploss
    trailing_stop = True
    trailing_only_offset_is_reached = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.011  # Disabled / not configured

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
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = []
        for info_timeframe in self.info_timeframes:
            informative_pairs.extend([(pair, info_timeframe) for pair in pairs])

        #if self.config['stake_currency'] in ['USDT','BUSD','USDC','DAI','TUSD','PAX','USD','EUR','GBP']:
        #    btc_info_pair = f"BTC/{self.config['stake_currency']}"
        #else:
        #    btc_info_pair = "BTC/USDT"
#
 #       informative_pairs.extend([(btc_info_pair, btc_info_timeframe) for btc_info_timeframe in self.btc_info_timeframes])

        return informative_pairs
        # Range midpoint acts as Support

    def is_support(row_data) -> bool:
        conditions = []
        for row in range(len(row_data) - 1):
            if row < len(row_data) // 2:
                conditions.append(row_data[row] > row_data[row + 1])
            else:
                conditions.append(row_data[row] < row_data[row + 1])
        result = reduce(lambda x, y: x & y, conditions)
        return result

        # Range midpoint acts as Resistance

    def is_resistance(row_data) -> bool:
        conditions = []
        for row in range(len(row_data) - 1):
            if row < len(row_data) // 2:
                conditions.append(row_data[row] < row_data[row + 1])
            else:
                conditions.append(row_data[row] > row_data[row + 1])
        result = reduce(lambda x, y: x & y, conditions)
        return result

        # Peak Percentage Change

    def range_percent_change(self, dataframe: DataFrame, method, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param method: High to Low / Open to Close
        :param length: int The length to look back
        """
        if method == 'HL':
            return (dataframe['high'].rolling(length).max() - dataframe['low'].rolling(length).min()) / dataframe[
                'low'].rolling(length).min()
        elif method == 'OC':
            return (dataframe['open'].rolling(length).max() - dataframe['close'].rolling(length).min()) / dataframe[
                'close'].rolling(length).min()
        else:
            raise ValueError(f"Method {method} not defined!")

        # Williams %R

    def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
        """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
           of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
          Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
          of its recent trading range.
          The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
        """

        highest_high = dataframe["high"].rolling(center=False, window=period).max()
        lowest_low = dataframe["low"].rolling(center=False, window=period).min()

        WR = Series(
            (highest_high - dataframe["close"]) / (highest_high - lowest_low),
            name=f"{period} Williams %R",
        )

        return WR * -100

    def williams_fractals(dataframe: pd.DataFrame, period: int = 2) -> tuple:
        """Williams Fractals implementation

        :param dataframe: OHLC data
        :param period: number of lower (or higher) points on each side of a high (or low)
        :return: tuple of boolean Series (bearish, bullish) where True marks a fractal pattern
        """

        window = 2 * period + 1

        bears = dataframe['high'].rolling(window, center=True).apply(lambda x: x[period] == max(x), raw=True)
        bulls = dataframe['low'].rolling(window, center=True).apply(lambda x: x[period] == min(x), raw=True)

        return bears, bulls
    def get_informative_indicators(self, metadata: dict):

        dataframe = self.dp.get_pair_dataframe(
            pair=metadata['pair'], timeframe=self.informative_timeframe)

        return []

        # Coin Pair Indicator Switch Case
        # ---------------------------------------------------------------------------------------------
    def informative_15m_indicators(self, metadata: dict, info_timeframe) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."

        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=info_timeframe)

        # Indicators
        # -----------------------------------------------------------------------------------------

        # RSI
        informative_15m['rsi_3'] = ta.RSI(informative_15m, timeperiod=3)
        informative_15m['rsi_14'] = ta.RSI(informative_15m, timeperiod=14)

        # EMA
        informative_15m['ema_12'] = ta.EMA(informative_15m, timeperiod=12)
        informative_15m['ema_26'] = ta.EMA(informative_15m, timeperiod=26)

        # SMA
        informative_15m['sma_200'] = ta.SMA(informative_15m, timeperiod=200)

        # BB - 20 STD2
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_15m), window=20, stds=2)
        informative_15m['bb20_2_low'] = bollinger['lower']
        informative_15m['bb20_2_mid'] = bollinger['mid']
        informative_15m['bb20_2_upp'] = bollinger['upper']

        # CTI
        informative_15m['cti_20'] = pta.cti(informative_15m["close"], length=20)

        # Downtrend check
        informative_15m['not_downtrend'] = ((informative_15m['close'] > informative_15m['open']) | (informative_15m['close'].shift(1) > informative_15m['open'].shift(1)) | (informative_15m['close'].shift(2) > informative_15m['open'].shift(2)) | (informative_15m['rsi_14'] > 50.0) | (informative_15m['rsi_3'] > 25.0))

        # Volume
        informative_15m['volume_mean_factor_12'] = informative_15m['volume'] / informative_15m['volume'].rolling(12).mean()

        # Performance logging
        # -----------------------------------------------------------------------------------------
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_15m_indicators took: {tok - tik:0.4f} seconds.")

        return informative_15m

    def informative_1h_indicators(self, metadata: dict, info_timeframe) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=info_timeframe)

        # Indicators
        # -----------------------------------------------------------------------------------------
        # Calculate the Linear Regression
        lsma = pta.linreg(informative_1h['close'], length=50, offset=0)
        lsma2 = pta.linreg(lsma, length=50, offset=0)
        # Calculate the Zero Lag SMA
        if lsma is not None and lsma2 is not None:
            eq = lsma - lsma2
            zlsma = lsma + eq
            informative_1h['zlsma'] = zlsma
        else:
            eq = 0
            zlsma = 0
            informative_1h['zlsma'] = zlsma

        # EverGet ChandilerExit
        high = informative_1h['high']  # Replace with your high data
        low = informative_1h['low']  # Replace with your low data
        close = informative_1h['close']  # Replace with your close data
        # atr = ta.ATR(high, low, close, self.atr_period) * self.atr_multiplier
        atr = ta.ATR(high, low, close, 22) * 3
        longStop = (high.rolling(22).max() if True else high.rolling(22).apply(
            lambda x: x[:-1].max())) - atr
        longStopPrev = longStop.shift(1).fillna(longStop)
        longStop = close.shift(1).where(close.shift(1) > longStopPrev, longStop)

        shortStop = (low.rolling(22).min() if True else low.rolling(22).apply(
            lambda x: x[:-1].min())) + atr
        shortStopPrev = shortStop.shift(1).fillna(shortStop)
        shortStop = close.shift(1).where(close.shift(1) < shortStopPrev, shortStop)

        # dir = close.apply(lambda x: 1 if x > shortStopPrev.iloc[-1] else -1 if x < longStopPrev.iloc[-1] else dir[-1])
        informative_1h['dir'] = 1
        informative_1h.loc[informative_1h['close'] <= longStopPrev, 'dir'] = -1
        informative_1h.loc[informative_1h['close'] > shortStopPrev, 'dir'] = 1

        longColor = 'green'
        shortColor = 'red'

        longStopPlot = longStop.where(informative_1h['dir'] == 1, None)
        buySignal = (informative_1h['dir'] == 1) & (informative_1h['dir'].shift(1) == -1)
        buySignalPlot = longStop.where(buySignal, None)
        buyLabel = pd.Series(['Buy' if x else '' for x in buySignal]).where(True & buySignal, None).any()

        shortStopPlot = shortStop.where(informative_1h['dir'] == -1, None)
        sellSignal = (informative_1h['dir'] == -1) & (informative_1h['dir'].shift(1) == 1)
        sellSignalPlot = shortStop.where(sellSignal, None)
        sellLabel = pd.Series(['Sell' if x else '' for x in sellSignal]).where(True & sellSignal, None).any()

        midPricePlot = close

        longFillColor = longColor if True and (informative_1h['dir'] == 1).any() else None
        shortFillColor = shortColor if True and (informative_1h['dir'] == -1).any() else None

        # RSI
        informative_1h['rsi_3'] = ta.RSI(informative_1h, timeperiod=3)
        informative_1h['rsi_14'] = ta.RSI(informative_1h, timeperiod=14)
        informative_1h['rsi_25'] = ta.RSI(informative_1h, timeperiod=25)

        # EMA
        informative_1h['ema_12'] = ta.EMA(informative_1h, timeperiod=12)
        informative_1h['ema_21'] = ta.EMA(informative_1h, timeperiod=21)
        informative_1h['ema_26'] = ta.EMA(informative_1h, timeperiod=26)
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)

        informative_1h['ema_200_dec_48'] = ((informative_1h['ema_200'].isnull()) | (
                    informative_1h['ema_200'] <= informative_1h['ema_200'].shift(48)))

        # SMA
        informative_1h['sma_12'] = ta.SMA(informative_1h, timeperiod=12)
        informative_1h['sma_21'] = ta.SMA(informative_1h, timeperiod=21)
        informative_1h['sma_26'] = ta.SMA(informative_1h, timeperiod=26)
        informative_1h['sma_50'] = ta.SMA(informative_1h, timeperiod=50)
        informative_1h['sma_100'] = ta.SMA(informative_1h, timeperiod=100)
        informative_1h['sma_200'] = ta.SMA(informative_1h, timeperiod=200)

        # BB
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb20_2_low'] = bollinger['lower']
        informative_1h['bb20_2_mid'] = bollinger['mid']
        informative_1h['bb20_2_upp'] = bollinger['upper']

        informative_1h['bb20_2_width'] = (
                    (informative_1h['bb20_2_upp'] - informative_1h['bb20_2_low']) / informative_1h['bb20_2_mid'])

        # Williams %R
        #informative_1h['r_14'] = williams_r(informative_1h, period=14)
        #informative_1h['r_96'] = williams_r(informative_1h, period=96)
        #informative_1h['r_480'] = williams_r(informative_1h, period=480)

        # CTI
        informative_1h['cti_20'] = pta.cti(informative_1h["close"], length=20)
        informative_1h['cti_40'] = pta.cti(informative_1h["close"], length=40)

        # SAR
        informative_1h['sar'] = ta.SAR(informative_1h)

        # S/R
      #  res_series = informative_1h['high'].rolling(window=5, center=True).apply(lambda row: is_resistance(row),
      #                                                                           raw=True).shift(2)
      #  sup_series = informative_1h['low'].rolling(window=5, center=True).apply(lambda row: is_support(row),
      #                                                                          raw=True).shift(2)
      #  informative_1h['res_level'] = Series(np.where(res_series,
      #                                                np.where(informative_1h['close'] > informative_1h['open'],
      #                                                         informative_1h['close'], informative_1h['open']),
      #                                                float('NaN'))).ffill()
      #  informative_1h['res_hlevel'] = Series(np.where(res_series, informative_1h['high'], float('NaN'))).ffill()
      #  informative_1h['sup_level'] = Series(np.where(sup_series,
      #                                                np.where(informative_1h['close'] < informative_1h['open'],
      #                                                         informative_1h['close'], informative_1h['open']),
      #                                                float('NaN'))).ffill()

        # Pump protections
        #informative_1h['hl_pct_change_48'] = range_percent_change(self, informative_1h, 'HL', 48)
        #informative_1h['hl_pct_change_36'] = range_percent_change(self, informative_1h, 'HL', 36)
        #informative_1h['hl_pct_change_24'] = range_percent_change(self, informative_1h, 'HL', 24)
        #informative_1h['hl_pct_change_12'] = range_percent_change(self, informative_1h, 'HL', 12)
        #informative_1h['hl_pct_change_6'] = range_percent_change(self, informative_1h, 'HL', 6)

        # Downtrend checks
        informative_1h['not_downtrend'] = (
                    (informative_1h['close'] > informative_1h['close'].shift(2)) | (informative_1h['rsi_14'] > 50.0))

        informative_1h['is_downtrend_3'] = ((informative_1h['close'] < informative_1h['open']) & (
                    informative_1h['close'].shift(1) < informative_1h['open'].shift(1)) & (
                                                        informative_1h['close'].shift(2) < informative_1h['open'].shift(
                                                    2)))

        informative_1h['is_downtrend_5'] = ((informative_1h['close'] < informative_1h['open']) & (
                    informative_1h['close'].shift(1) < informative_1h['open'].shift(1)) & (
                                                        informative_1h['close'].shift(2) < informative_1h['open'].shift(
                                                    2)) & (
                                                        informative_1h['close'].shift(3) < informative_1h['open'].shift(
                                                    3)) & (
                                                        informative_1h['close'].shift(4) < informative_1h['open'].shift(
                                                    4)))

        # Wicks
        informative_1h['top_wick_pct'] = (
                    (informative_1h['high'] - np.maximum(informative_1h['open'], informative_1h['close'])) / np.maximum(
                informative_1h['open'], informative_1h['close']))

        # Candle change
        informative_1h['change_pct'] = (informative_1h['close'] - informative_1h['open']) / informative_1h['open']

        # Max highs
        informative_1h['high_max_3'] = informative_1h['high'].rolling(3).max()
        informative_1h['high_max_6'] = informative_1h['high'].rolling(6).max()
        informative_1h['high_max_12'] = informative_1h['high'].rolling(12).max()
        informative_1h['high_max_24'] = informative_1h['high'].rolling(24).max()
        informative_1h['high_max_36'] = informative_1h['high'].rolling(36).max()
        informative_1h['high_max_48'] = informative_1h['high'].rolling(48).max()

        # Volume
        informative_1h['volume_mean_factor_12'] = informative_1h['volume'] / informative_1h['volume'].rolling(12).mean()

        # Performance logging
        # -----------------------------------------------------------------------------------------
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_1h_indicators took: {tok - tik:0.4f} seconds.")

        return informative_1h

    def info_switcher(self, metadata: dict, info_timeframe) -> DataFrame:
            if info_timeframe == '1h':
                return self.informative_1h_indicators(metadata, info_timeframe)
           # elif info_timeframe == '4h':
           #     return self.informative_4h_indicators(metadata, info_timeframe)
           # elif info_timeframe == '1d':
           #     return self.informative_1d_indicators(metadata, info_timeframe)
            elif info_timeframe == '15m':
                return self.informative_15m_indicators(metadata, info_timeframe)
            else:
                raise RuntimeError(f"{info_timeframe} not supported as informative timeframe for USDT pairs.")

    # Coin Pair Base Timeframe Indicators
    # ---------------------------------------------------------------------------------------------
    def base_tf_5m_indicators(self,  metadata: dict, dataframe: DataFrame) -> DataFrame:
        tik = time.perf_counter()
        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        dataframe['ha_high'] = heikinashi['high']
        dataframe['ha_low'] = heikinashi['low']
        src = (dataframe['ha_high'] + dataframe['ha_low'])/2
        # Indicators
        # -----------------------------------------------------------------------------------------
        # RSI
        if self.change_atr:
            dataframe['atr'] = ta.ATR(
                dataframe, timeperiod=self.super_atr_period)
        else:
            dataframe['atr'] = ta.SMA(
                dataframe, timeperiod=self.super_atr_period)

            # Calculate Supertrend lines
        dataframe['up'] = src - \
                          (self.super_atr_multiplier * dataframe['atr'])
        dataframe['up1'] = dataframe['up'].shift(1).fillna(dataframe['up'])
        dataframe['up'] = dataframe.apply(lambda x: max(x['up'], x['up1']) if x['ha_close'] > x['up1'] else x['up'],
                                          axis=1)

        dataframe['dn'] = src + \
                          (self.super_atr_multiplier * dataframe['atr'])
        dataframe['dn1'] = dataframe['dn'].shift(1).fillna(dataframe['dn'])
        dataframe['dn'] = dataframe.apply(lambda x: min(x['dn'], x['dn1']) if x['ha_close'] < x['dn1'] else x['dn'],
                                          axis=1)

        # Calculate trend
        dataframe['trend'] = 1
        dataframe['trend'] = dataframe['trend'].fillna(method='ffill')
        dataframe.loc[(dataframe['trend'] == -1) &
                      (dataframe['ha_close'] > dataframe['dn1']), 'trend'] = 1
        dataframe.loc[(dataframe['trend'] == 1) & (
                dataframe['ha_close'] < dataframe['up1']), 'trend'] = -1

        # Elliot
        dataframe['EWO'] = EWO(dataframe, self.fast_ewo, self.slow_ewo)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_25'] = ta.RSI(dataframe, timeperiod=25)

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
        atr = ta.ATR(high, low, close, self.atr_period) * self.atr_multiplier
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
        # Parabolic SAR
        dataframe['sar'] = ta.SAR(dataframe)

        # Performance logging
        # -----------------------------------------------------------------------------------------
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] base_tf_5m_indicators took: {tok - tik:0.4f} seconds.")
        return dataframe

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
        tik = time.perf_counter()
        '''
          --> Indicators on informative timeframes
          ___________________________________________________________________________________________
          '''
        for info_timeframe in self.info_timeframes:
            info_indicators = self.info_switcher(metadata, info_timeframe)
            dataframe = merge_informative_pair(dataframe, info_indicators, self.timeframe, info_timeframe, ffill=True)
            # Customize what we drop - in case we need to maintain some informative timeframe ohlcv data
            # Default drop all except base timeframe ohlcv data
            drop_columns = {
           #     '1d': [f"{s}_{info_timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']],
           #     '4h': [f"{s}_{info_timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']],
                '1h': [f"{s}_{info_timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']],
                '15m': [f"{s}_{info_timeframe}" for s in ['date', 'high', 'low', 'volume']]
            }.get(info_timeframe, [f"{s}_{info_timeframe}" for s in ['date', 'open', 'high', 'low', 'close', 'volume']])
            dataframe.drop(columns=dataframe.columns.intersection(drop_columns), inplace=True)

        '''
        --> The indicators for the base timeframe  (5m)
        ___________________________________________________________________________________________
        '''
        dataframe = self.base_tf_5m_indicators(metadata, dataframe)

        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] Populate indicators took a total of: {tok - tik:0.4f} seconds.")

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
                    ## Protections
                    item_buy_logic.append(dataframe['cti_20_1h'] < 0.8)
                    item_buy_logic.append(dataframe['rsi_14_1h'] < 80.0)
                    item_buy_logic.append(dataframe['high_max_24_1h'] < (dataframe['close'] * 1.5))
                    #item_buy_logic.append(dataframe['hl_pct_change_6_1h'] < 0.4)
                    #item_buy_logic.append(dataframe['hl_pct_change_12_1h'] < 0.5)
                    #item_buy_logic.append(dataframe['hl_pct_change_24_1h'] < 0.75)
                    #item_buy_logic.append(dataframe['hl_pct_change_48_1h'] < 0.9)
                    item_buy_logic.append((dataframe['cti_20_15m'] < -0.5)
                                          | (dataframe['rsi_3_15m'] > 25.0)
                                          | (dataframe['rsi_14_15m'] < 30.0)
                                          | (dataframe['cti_20_1h'] < 0.5)
                                          | (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(96)))
                    item_buy_logic.append((dataframe['cti_20_15m'] < -0.8)
                                          | (dataframe['rsi_14_15m'] < 30.0)
                                          | (dataframe['cti_20_1h'] < 0.5)
                                          | (dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(96)))

                    ## Logic
                    item_buy_logic.append((dataframe['dir_1h'] == 1)
                                          & (dataframe['dir_1h'].shift(1) == -1)
                                           & (dataframe['rsi_25_1h'] > 30)
                                           & (dataframe['rsi_25_1h'] < 70)
                                           & (dataframe['close'] > dataframe['zlsma_1h'])
                                           & (dataframe['ema26'] > (dataframe['ema12']))
                                           #& (dataframe['ema26'].shift() - dataframe['ema12'].shift()) > (dataframe['open'] / 100)
                                           & (dataframe['close'] > dataframe['close'].shift())  # Current close is higher than previous close
                                           & (dataframe['sar'] < dataframe['low'])  # SAR is below the low price
                                          )
                    # Buy 4: SuperTrend
                if index == 4:
                    item_buy_logic.append((dataframe['trend'] == 1) & (dataframe['trend'].shift(1) == -1)
                                          & (dataframe['rsi_25'] > 24)
                                          & (dataframe['rsi_25'] < 70)
                                          & (dataframe['close'] > dataframe['zlsma'])
                                          & (dataframe['ema21'] > (dataframe['ema7']))
                                          & (dataframe['close'] > dataframe['close'].shift())  # Current close is higher than previous close
                                          & (dataframe['sar'] < dataframe['low'])  # SAR is below the low price
                                          )

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
                            & (dataframe['close'] < dataframe[
                            'close'].shift()) &  # Current close is lower than previous close
                            (dataframe['sar'] < dataframe['high'])  # SAR is below the high price
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
                            & (dataframe['close'] < dataframe[
                            'close'].shift()) &  # Current close is lower than previous close
                            (dataframe['sar'] < dataframe['high'])  # SAR is below the high price
                        )
                    )

                # Sell 3: Chandelier Exit
                if index == 3:
                    item_sell_logic.append(
                        ((dataframe['dir_1h'] == -1) & (dataframe['dir_1h'].shift(1) == 1))
                        & (dataframe['close'] < dataframe['close'].shift()) &  # Current close is lower than previous close
                        (dataframe['sar'] < dataframe['high'])  # SAR is below the high price
                    )
                    # Sell 4: SuperTrend
                if index == 4:
                    item_sell_logic.append(
                        ((dataframe['trend'] == -1) & (dataframe['trend'].shift(1) == 1))
                        & (dataframe['close'] < dataframe[
                            'close'].shift()) &  # Current close is lower than previous close
                        (dataframe['sar'] < dataframe['high'])  # SAR is below the high price
                    )

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
    def save_to_csv(self, dataframe: DataFrame) -> None:
        dataframe.to_csv('output.csv', index=False)

    def on_postback(self, dataframe: DataFrame) -> None:
        # Save the final DataFrame to a CSV file after backtesting or live trading
        self.save_to_csv(dataframe)

