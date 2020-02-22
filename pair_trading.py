import bs4
import numpy as np
import pandas_datareader.data as web
import random
import requests
import time

from datetime import datetime
from dateutil.relativedelta import relativedelta

import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

MONTHS = 12
SAMPLE = 1.0

def get_s_and_p_symbols():
    response = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs4.BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        tickers.append(ticker)

    return tickers


def query_time_series(symbol, start_date, end_date, source='yahoo', col='Adj Close'):
    try:
        df = web.DataReader(symbol, source, start_date, end_date)

        # Drop all columns but the date and the column.
        df = df[[col]]
        df = df.rename({col: symbol}, axis=1)

        return df
    except KeyError:
        return None


def cointegrated_pairs(df):
    n = df.shape[1]

    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))

    keys = df.keys()
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            S1 = df[keys[i]]
            S2 = df[keys[j]]

            result = coint(S1, S2)

            score = result[0]
            pvalue = result[1]

            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue

            if pvalue < 0.05:
                pairs.append((keys[i], keys[j], pvalue))

    return score_matrix, pvalue_matrix, pairs


def z_score(series):
    return (series - series.mean()) / np.std(series)


def fit(x, y):
    S1 = x
    S2 = y

    S1 = sm.add_constant(S1)

    results = sm.OLS(S2, S1).fit()


    S1 = x
    b = results.params[x.name]

    spread = S2 - b * S1

    return (results.rsquared, spread)


if __name__ == '__main__':
    start_time = time.time()

    symbols = get_s_and_p_symbols()
    symbols = random.sample(symbols, int(SAMPLE * len(symbols)))

    print('')
    print('Downloaded {:} symbols.'.format(len(symbols)))

    end_date = datetime.now().date()
    start_date = end_date + relativedelta(months=-MONTHS)

    print('')
    print('Start date:  {:}'.format(start_date))
    print('  End date:  {:}'.format(end_date))
    print('')

    print('Downloading prices...')

    prices_df = None
    for symbol in symbols:
        df = query_time_series(symbol, start_date, end_date)

        if df is not None:
            if prices_df is None:
                prices_df = df
            else:
                prices_df = prices_df.merge(df, left_index=True, right_index=True, how='inner')

    print('')
    print(prices_df.head(5))
    print('')

    candidates = []

    _, _, pairs = cointegrated_pairs(prices_df)
    for symbol1, symbol2, p_value in sorted(pairs, key=lambda x: x[2]):
        r_squared, spread = fit(prices_df[symbol1], prices_df[symbol2])
        signal = z_score(spread)[-1:][0]

        if r_squared > 0.80 and (signal < -1.0 or signal > 1.0):
            candidates.append((symbol1, symbol2, p_value, r_squared, signal))

    candidates = sorted(candidates, key=lambda x: abs(x[4]), reverse=True)
    for candidate in candidates:
        print(candidate)

    end_time = time.time()

    print('')
    print('Execution time:  {:}s'.format(end_time-start_time))

    exit(0)