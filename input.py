import matplotlib
matplotlib.use('Agg')
import pandas as pd
from copy import deepcopy
import numpy as np



class InputModule:
    def __init__(self, input_path, start_date, num_dates, num_tests):
        self.input_path = input_path
        self.start_date = start_date
        self.num_dates = num_dates
        self.num_tests = num_tests

    def get_date_list(self):
        """
        get all dates from data
        """
        date_list = []
        lines = open(self.input_path, "r").readlines()
        for line in lines:
            token = line.split(",")
            date_list.append(token[0])
        return date_list

    def get_predict_dates(self):
        """
        get list of test date
        """
        all_dates = self.get_date_list()
        for i in range(len(all_dates)):
            if all_dates[i] >= self.start_date:
                predict_dates = all_dates[i+self.num_dates - self.num_tests:i+self.num_dates]
                break
        return predict_dates


    def get_input(self, pivot_date='9999-99-99'):
        """
        get data from start_date to pivot_date
        input: 
        - pivot_date: current date to predict
        return:
        - data: series of all history prices
        - real_price: real price of date to be predict.
        - last_price: the price of date before the date which is predicted
        """
        dates = []
        prices = []
        real_price = None
        lines = open(self.input_path, 'r').readlines()
        for line in lines:
            token = line.split(',')
            if self.start_date <= token[0] < pivot_date:
                dates.append(token[0])
                prices.append(float(token[1]))
            if token[0] == pivot_date:
                real_price = float(token[1])
        last_price = prices[-1]
        time_index = pd.DatetimeIndex(dates)
        data = pd.Series(prices, index=time_index)
        return data, real_price, last_price


    def get_diff_data(self, data, time_diff=1):
        data_to_return = data
        for i in range(time_diff):
            data_to_return = pd.Series.diff(data_to_return)
        return data_to_return[1:]

    def get_return_value_from_diff(sefl, predict_value, diff_data_list):
        value = predict_value
        for i in range(len(diff_data_list) - 1):
            value += diff_data_list[i][0] + diff_data_list[i+1].sum()
        return value


    def makeInputSequence(self, len):
        return np.array(list(range(1, len + 1)))


    def get_components(self, data):
        trend = deepcopy(data[5:])
        for i in range(len(trend)):
            trend[i] = data[i:i+5].describe()[1]
        de_trend_series = data[5:]/trend
        temprate0 = []
        temprate1 = []
        temprate2 = []
        temprate3 = []
        temprate4 = []
        for i in range(len(de_trend_series)):
            if de_trend_series.index[i].weekday() == 0:
                temprate0.append(de_trend_series[i])
            if de_trend_series.index[i].weekday() == 1:
                temprate1.append(de_trend_series[i])
            if de_trend_series.index[i].weekday() == 2:
                temprate2.append(de_trend_series[i])
            if de_trend_series.index[i].weekday() == 3:
                temprate3.append(de_trend_series[i])
            if de_trend_series.index[i].weekday() == 4:
                temprate4.append(de_trend_series[i])

        seasonal_components = np.array([np.mean(temprate0), np.mean(temprate1), np.mean(temprate2), np.mean(temprate3), np.mean(temprate4)])
        seasonal = deepcopy(data[5:])
        for i in range(len(seasonal)):
            if seasonal.index[i].weekday() == 0:
                seasonal[i] = seasonal_components[0]
            if seasonal.index[i].weekday() == 1:
                seasonal[i] = seasonal_components[1]
            if seasonal.index[i].weekday() == 2:
                seasonal[i] = seasonal_components[2]
            if seasonal.index[i].weekday() == 3:
                seasonal[i] = seasonal_components[3]
            if seasonal.index[i].weekday() == 4:
                seasonal[i] = seasonal_components[4]

        residual = de_trend_series/seasonal
        return trend, seasonal, residual

        residual = de_trend_series - seasonal
        print("trend: %s, seasonal: %s, residual: %s" %(trend, seasonal, residual))
        return trend, seasonal, residual

