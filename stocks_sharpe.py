# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 20:27:35 2016

@author: lenovo1
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 13:33:58 2016

@author: sb
"""

import pandas as pd
import numpy as np
import datetime 
from pandas.io.data import DataReader
import seaborn as sns
import matplotlib.pyplot as plt

stocks_name_full = []
stocks_name = []

import csv
with open('C:\doc\python\ccar_banks_list.csv') as csvfile:
    file = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in file:
        stocks_name_full.append(row)
        stocks_name.append(row[0][0:19])
        
stocks_name_df = pd.DataFrame([stocks_name_full, stocks_name]).transpose()
stocks_name_df.columns = ['Full Name', 'Name']

stocks_ticker = []
stocks_ticker_name = []
import csv
with open('C:\doc\python\companylist_nasdaq.csv') as csvfile:
    file = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in file:
        stocks_ticker.append(row[0])
        stocks_ticker_name.append(row[1][0:19])

import csv
with open('C:\doc\python\companylist_nyse.csv') as csvfile:
    file = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in file:
        stocks_ticker.append(row[0])
        stocks_ticker_name.append(row[1][0:19])

stock_tickers_df = pd.DataFrame([stocks_ticker, stocks_ticker_name]).transpose()
stock_tickers_df.columns = ['Ticker', 'Name']

bank_stocks = pd.merge(stocks_name_df, stock_tickers_df, how='inner', on='Name')

symbols = ['JPM', 'ZION', 'C', 'WFC', 'BAC', 'STI', 'GS', 'MS', 'COF', 'PNC', 'YHOO', 'AAPL']
stocks_data = {}
#stocks_mean_rtn = pd.DataFrame()
#stocks_risk = pd.DataFrame()

#rfr = DataReader('DTB3', "fred", datetime.date(2016, 1, 1), datetime.date(2016,5,31))

start = datetime.date(2000, 1, 1)
end = datetime.date(2015, 12, 31)
stocks_data = DataReader(symbols, "yahoo", start, end)['Adj Close']
stocks_rtn = stocks_data.pct_change()
rfr = DataReader('DTB3', "fred", start, end)
stocks_mean_rtn = pd.DataFrame(data=stocks_rtn.mean())
rfr_mean = pd.DataFrame(data=rfr.mean() / (100 * rfr.count()))
stocks_eff_mean_rtn = stocks_mean_rtn - rfr_mean
stocks_risk = pd.DataFrame(data=stocks_rtn.std())
stocks_sharpe = pd.DataFrame((stocks_eff_mean_rtn - rfr_mean) / stocks_risk)
        
x = stocks_sharpe.index.values
y = stocks_sharpe.columns

stocks_sharpe.plot(x, y, '-')


#sns.distplot(stocks_rtn['ZION'].dropna(), bins=100)

#sns.pairplot(stocks_rtn.dropna(), bins=25)

#sns.corrplot(stocks_rtn.dropna(), annot=True)

stocks_mean_rtn_all = stocks_mean_rtn.mean()
#stocks_std_rtn_all = stock_risk.

stocks_plot = pd.concat([stocks_mean_rtn, stocks_risk], axis = 1)

plt.scatter(x=stocks_plot[0], y=stocks_plot[1])
plt.ylim(0.02, 0.04)
plt.xlim(-0.000, 0.001)

plt.xlabel('Mean return')
plt.ylabel('Risk')

for label, x, y in zip(stocks_plot.index, stocks_plot[0], stocks_plot[1]):
    plt.annotate(label, xy = (x, y), xytext = (50, 50), 
                 textcoords = 'offset points', ha='right',
                 va='bottom', arrowprops=dict(arrowstyle='-', connectionstyle='arc3, rad=-0.3'))
