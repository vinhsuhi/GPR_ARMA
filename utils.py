import pandas as pd
import os
import glob
import numpy as np
import os
from statsmodels import api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.stats as stats

class Visualizer:
    def __init__(self):
        if not os.path.exists('Plots'):
            os.makedirs('Plots')
        pass

    def drawComponents(self, real_price_list):
        res = sm.tsa.seasonal_decompose(
            real_price_list, freq=5, model = 'additive')
        resplot = res.plot()
        resplot.savefig("Plots/components.png")

    def drawHistoryPrice(self, real_price_list):
        fig, ax = plt.subplots(figsize=(20,10))
        plt.plot(np.arange(len(real_price_list)), real_price_list, color='blue')
        plt.xlabel('Days')
        plt.ylabel('Value')
        plt.title("History Prices")
        plt.legend()
        plt.savefig("Plots/HistoryPrice.png")

    def drawACFofResidual(self, residual):
        n = residual.shape[0]
        acf = sm.tsa.acf(residual, nlags = 250)
        plt.figure(figsize=(12,6), facecolor='white')
        plt.bar(range(len(acf)), acf, width=0.001)
        z = stats.norm.ppf(0.95)
        plt.axhline(y=z/np.sqrt(n), linestyle='--', color='magenta')
        plt.axhline(y=-z/np.sqrt(n), linestyle='--', color='magenta')
        plt.axhline(0, color='black', lw=1)
        plt.title("residual ACF")
        plt.xlim(xmax=251, xmin=-1)
        plt.savefig("Plots/ACFResidual.png")

    def drawResult(self, predictResults1, predictResults2, predictResults3, realPrices, title):
        fig, ax = plt.subplots(figsize=(20,10))
        lw = 2
        plt.plot(np.arange(len(realPrices)), predictResults1, color='green', label = 'GPR-ARMA')
        plt.plot(np.arange(len(realPrices)), predictResults2, color='red', label = 'GPR')
        plt.plot(np.arange(len(realPrices)), predictResults3, color='magenta', label = 'ARMA')
        plt.plot(np.arange(len(realPrices)), realPrices, color='black', lw=lw, label = 'real prices')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.savefig("Plots/" + title + ".png")



class Log:
    def __init__(self):
        pass

    def update(self, date, correct, realPrice, lastPrice, predictPrice, method, out_path):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        columns = ["method", "Date", "realPrice", "lastPrice", "predictPrice", "correct"]
        outFile =  out_path + "/" + "%s %s.txt"%(method, date)
        open(outFile, "w").write("%s\n"%(",".join(columns)))
        open(outFile, "a").write("%s,%s,%s,%s,%s,%s"\
            %(method, date, realPrice, lastPrice, predictPrice, correct))
        return True

    def load_results_from_file(self, name, out_path):
        predict_items = [] 
        assert os.path.exists(out_path), "Have not had data yet!"
        for file in glob.glob(out_path + "/" + name + "/" + "*.txt"):
            lines = open(file, 'r').readlines()
            headers = lines[0].split(",")
            tokens = lines[1].split(",")
            item = {}
            for j in range(len(headers)):
                item[headers[j]] = tokens[j]
            predict_items.append(item)
        predict_items.sort(key=lambda x:x['Date'])
        return predict_items


    def load_all_results(self, out_path):
        methods = ['GPR-ARMA', 'GPR', 'ARMA']
        predict_items = {}
        for i in range(3):
            predict_items[str(i+1)] = self.load_results_from_file(methods[i], out_path)
        
        GPR_ARMA_prices = np.zeros(len(predict_items['1']))
        GPR_prices = np.zeros(len(predict_items['2']))
        ARMA_prices = np.zeros(len(predict_items['3']))
        real_prices = np.zeros(len(predict_items['1']))
        for i in range(len(predict_items['1'])):
            GPR_ARMA_prices[i] = predict_items['1'][i]['predictPrice']
            real_prices[i] = predict_items['1'][i]['realPrice']
        for i in range(len(predict_items['2'])):
            GPR_prices[i] = predict_items['2'][i]['predictPrice']
        for i in range(len(predict_items['3'])):
            ARMA_prices[i] = predict_items['3'][i]['predictPrice']

        return GPR_ARMA_prices, GPR_prices, ARMA_prices, real_prices
