import pyGPs
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima_model import ARMA
import numpy as np



class AutoRegressiveMovingAverage:
    def __init__(self, ic, trend, max_ar, max_ma, method):
        self.ic = ic
        self.trend = trend
        self.max_ar = max_ar
        self.max_ma = max_ma
        self.method = method

    def arma_prediction(self, data):
        ic_order = arma_order_select_ic(data,
                                        ic = self.ic,
                                        trend = self.trend,
                                        max_ar = self.max_ar,
                                        max_ma = self.max_ma,
                                        fit_kw = {'method': self.method})
        aicOrder = icOrder['aic_min_order']
        model = ARMA(data, order=aicOrder)
        results = model.fit(trend=self.trend, method=method, disp=-1)
        predict_result, _, _ = results.forecast(steps=1)
        return predict_result[0]


class GaussianProcess:
    def __init__(self, optimizer, num_restarts):
        self.optimizer = optimizer
        self.num_restarts = num_restarts

    def gpr_prediction(self, data):
        model = pyGPs.GPR()
        x = np.arange(len(data))
        xt = np.array(x[-1] + 1)
        model.setData(x, data)
        k1 = pyGPs.cov.RBF()
        model.setPrior(kernel=k1)
        model.setOptimizer(self.optimizer, num_restarts=self.num_restarts)
        model.optimize(x, data)
        ym = model.predict(xt)[0]
        predict = ym[0][0]
        return predict
