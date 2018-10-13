import argparse
import traceback
from algorithms import AutoRegressiveMovingAverage, GaussianProcess
from utils import Visualizer, Log
from input import InputModule
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="ARMA_GPR-ARMA-GPR_in_StockPrice_Prediction")
    parser.add_argument('--mode', default='train')
    parser.add_argument('--input_path',   default='data/vn_index.txt')
    parser.add_argument('--output_path',   default='results')
    parser.add_argument('--start_date',   default='2015-02-12')
    parser.add_argument('--num_dates',     default=377, type=int)
    parser.add_argument('--num_tests',  default=81, type=int)
    parser.add_argument('--model', default='GPR-ARMA')
    return parser.parse_args()

def arma_block(data, diff_data_list, iARMA, Input):
    diff_data = data
    while True:
        try:
            result = iARMA.arma_prediction(np.array([diff_data[i] for i in range(len(diff_data))]))
            print(result)
        except:
            diff_data = Input.get_diff_data(diff_data)
            diff_data_list.append(diff_data)
            continue
        break
    return_data = Input.get_predict_price_from_diff(result, diff_data_list)
    return return_data


def gpr_block(data, diff_data_list, iGPR, Input):
    diff_data = data
    while True:
        try:
            print(diff_data[0])
            result = iGPR.gpr_prediction(np.array([diff_data[i] for i in range(len(diff_data))]))
            print(result)
        except:
            diff_data = Input.get_diff_data(diff_data)
            diff_data_list.append(diff_data)
            continue
        break
    return_data = Input.get_predict_price_from_diff(result, diff_data_list)
    return return_data


def update_result(real_price, last_price, predict_price, date, method, output_path):
    logger = Log()
    correct = 0
    if predict_price > last_price and real_price > last_price:
        correct = 1
    elif predict_price < last_price and real_price < last_price:
        correct = 1
    logger.update(date, correct, real_price, last_price, predict_price, method, output_path)
    return correct


def gpr_arma(Input, iARMA, iGPR, output_path):
    predict_dates = Input.get_predict_dates()
    final_results = []
    trend_results = []
    real_price_list = []
    for date in predict_dates:
        data, real_price, last_price = Input.get_input(pivot_date=date)
        trend, seasonal, residual = Input.get_components(data)

        print("Start predict trend using GPR for date: ", date)
        diff_trend_list = [trend]
        predicted_trend = gpr_block(trend, diff_trend_list, iGPR, Input)

        print("Start predict residual using ARMA for date: ", date)
        diff_residual_list = [residual]
        predicted_residual = arma_block(residual, diff_residual_list, iARMA, Input)

        weekday = pd.datetime.strptime(date, '%Y-%m-%d').weekday()
        predicted_seasonal = seasonal[weekday]

        predict_price = predicted_residual * predicted_trend * predicted_seasonal

        correct = update_result(real_price, last_price, predict_price, date, 'GPR-ARMA', output_path)

        final_results.append(predict_price)
        real_price_list.append(real_price)
        trend_results.append(correct)
        print("Predict result for Date: %s, correctPredict: %s, predictPrice: %s" %(date, correct, predict_price))

    return trend_results, real_price_list, final_results



def arma(Input, iARMA, output_path):
    predict_dates = Input.get_predict_dates()
    final_results = []
    trend_results = []
    real_price_list = []
    for date in predict_dates:
        data, real_price, last_price = Input.get_input(pivot_date=date)
        diff_data_list = [data]
        print("Start predict using ARMA for date: ", date)

        predict_price = arma_block(data, diff_data_list, iARMA, Input)

        correct = update_result(real_price, last_price, predict_price, date, 'ARMA', output_path)

        final_results.append(predict_price)
        real_price_list.append(real_price)
        trend_results.append(correct)
        print("Predict result for Date: %s, correctPredict: %s, predictPrice: %s" %(date, correct, predict_price))

    return trend_results, real_price_list, final_results


def gpr(Input, iGPR, output_path):
    predict_dates = Input.get_predict_dates()
    final_results = []
    trend_results = []
    real_price_list = []
    for date in predict_dates:
        data, real_price, last_price = Input.get_input(pivot_date=date)
        diff_data_list = [data]
        print("Start predict using ARMA for date: ", date)

        predict_price = gpr_block(data, diff_data_list, iGPR, Input)

        correct = update_result(real_price, last_price, predict_price, date, 'GPR', output_path)

        final_results.append(predict_price)
        real_price_list.append(real_price)
        trend_results.append(correct)
        print("Predict result for Date: %s, correctPredict: %s, predictPrice: %s" %(date, correct, predict_price))

    return trend_results, real_price_list, final_results


if __name__ == '__main__':
    args = parse_args()
    print(args)


    frist_date = args.start_date
    if args.mode == 'train':
        args.output_path += "/" + args.model

        Input = InputModule(args.input_path, args.start_date, args.num_dates, args.num_tests)
        iARMA = AutoRegressiveMovingAverage(['aic'], 'c', 5, 6, 'mle')
        iGPR = GaussianProcess("Minimize", 30)
        if args.model == 'GPR-ARMA':        
            trend_results, real_price_list, final_results = gpr_arma(Input, iARMA, iGPR, args.output_path)
        if args.model == 'GPR':
            trend_results, real_price_list, final_results = gpr(Input, iGPR, args.output_path)
        if args.model == 'ARMA':
            trend_results, real_price_list, final_results = arma(Input, iARMA, args.output_path)

        correct_ratio = np.sum(trend_results) / len(trend_results)
        print("Number of predict days: ", len(trend_results))
        print("Trend predict result: ", correct_ratio)


    elif args.mode == 'statistic':
        logger = Log()    
        if args.model == 'all':
            final_results1, final_results2, final_results3, real_price_list = logger.load_all_results(args.output_path)
            error1 = final_results1 - real_price_list
            error2 = final_results2 - real_price_list
            error3 = final_results3 - real_price_list

            MAPE1 = np.mean(np.absolute(np.divide(error1, real_price_list)))
            MAPE2 = np.mean(np.absolute(np.divide(error2, real_price_list)))
            MAPE3 = np.mean(np.absolute(np.divide(error3, real_price_list)))

            RMSE1 = np.sqrt(np.mean(np.square(error1)))
            RMSE2 = np.sqrt(np.mean(np.square(error2)))
            RMSE3 = np.sqrt(np.mean(np.square(error3)))

            MAD1 = np.mean(np.absolute(error1))
            MAD2 = np.mean(np.absolute(error2))
            MAD3 = np.mean(np.absolute(error3))

            print("MAPE, GPR-ARMA: %s, GPR: %s, ARMA: %s" %(MAPE1, MAPE2, MAPE3))
            print("RMSE, GPR-ARMA: %s, GPR: %s, ARMA: %s" %(RMSE1, RMSE2, RMSE3))
            print("MAD, GPR-ARMA: %s, GPR: %s, ARMA: %s" %(MAD1, MAD2, MAD3))

            vizualizer = Visualizer()
            vizualizer.drawResult(final_results1, final_results2, final_results3, real_price_list, 'result')

        else:
            final_results = logger.load_results_from_file(args.model, args.output_path)
            print("Number of predict days: ", len(final_results))
            correct_predict = np.zeros(len(final_results))
            real_price_list = np.zeros(len(final_results))
            for i in range(len(final_results)):
                correct_predict[i] = final_results[i]['correct\n']
                real_price_list[i] = final_results[i]['realPrice']
            num_correct = np.sum(correct_predict)
            print("Number of correct predict: %s" %num_correct)
            print("Result: %s" %np.mean(correct_predict))

            Input = InputModule(args.input_path, args.start_date, args.num_dates, args.num_tests)
            data, _, _ = Input.get_input()
            trend, seasonal, residual = Input.get_components(data)
            real_price_list = np.array([data[i] for i in range(len(data))])
            vizualizer = Visualizer()
            vizualizer.drawComponents(real_price_list)
            vizualizer.drawHistoryPrice(real_price_list)
            vizualizer.drawACFofResidual(residual)





    