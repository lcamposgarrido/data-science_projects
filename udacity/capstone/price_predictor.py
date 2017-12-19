# Luis Campos Garrido 2017

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def get_data(symbol):

    # Technical Indicators
    ti = TechIndicators(key='4BTFICZGTPWZRRQS', output_format='pandas')
    sma, _ = ti.get_sma(symbol=symbol, interval='daily')
    wma, _ = ti.get_wma(symbol='SPX', interval='daily')
    ema, _ = ti.get_ema(symbol=symbol, interval='daily')
    macd, _ = ti.get_macd(symbol=symbol, interval='daily')
    stoch, _ = ti.get_stoch(symbol=symbol, interval='daily')
    rsi, _ = ti.get_rsi(symbol=symbol, interval='daily')
    adx, _ = ti.get_adx(symbol=symbol, interval='daily')
    cci, _ = ti.get_cci(symbol=symbol, interval='daily')
    aroon, _ = ti.get_aroon(symbol=symbol, interval='daily')
    bbands, _ = ti.get_bbands(symbol='SPX', interval='daily')
    ad, _ = ti.get_ad(symbol='SPX', interval='daily')
    obv, _ = ti.get_obv(symbol='SPX', interval='daily')
    mom, _ = ti.get_mom(symbol='SPX', interval='daily')
    willr, _ = ti.get_willr(symbol='SPX', interval='daily')
    tech_ind = pd.concat([sma, ema, macd, stoch, rsi, adx, cci, aroon, bbands, ad, obv, wma, mom, willr], axis=1)

    ts = TimeSeries(key='4BTFICZGTPWZRRQS', output_format='pandas')
    close = ts.get_daily(symbol=symbol, outputsize='full')[0]['close']   # compact/full
    direction = (close > close.shift()).astype(int)
    target = direction.shift(-1).fillna(0).astype(int)
    target.name = 'target'

    data = pd.concat([tech_ind, close, target], axis=1)

    return data


def get_indicators(data, n):

    hh = data['high'].rolling(n).max()
    ll = data['low'].rolling(n).min()
    up, dw = data['close'].diff(), -data['close'].diff()
    up[up<0], dw[dw<0] = 0, 0
    macd = data['close'].ewm(12).mean() - data['close'].ewm(26).mean()
    macd_signal = macd.ewm(9).mean()
    tp = (data['high'] + data['low'] + data['close']) / 3
    tp_ma = tp.rolling(n).mean()
    indicators = pd.DataFrame(data=0, index=data.index,
                              columns=['sma', 'ema', 'momentum',
                                       'sto_k', 'sto_d', 'rsi',
                                       'macd', 'lw_r', 'a/d', 'cci'])
    indicators['sma'] = data['close'].rolling(10).mean()
    indicators['ema'] = data['close'].ewm(10).mean()
    indicators['momentum'] = data['close'] - data['close'].shift(n)
    indicators['sto_k'] = (data['close'] - ll) / (hh - ll) * 100
    indicators['sto_d'] = indicators['sto_k'].rolling(n).mean()
    indicators['rsi'] = 100 - 100 / (1 + up.rolling(14).mean() / dw.rolling(14).mean())
    indicators['macd'] = macd - macd_signal
    indicators['lw_r'] = (hh - data['close']) / (hh - ll) * 100
    indicators['a/d'] = (data['high'] - data['close'].shift()) / (data['high'] - data['low'])
    indicators['cci'] = (tp - tp_ma) / (0.015 * tp.rolling(n).apply(lambda x: np.std(x)))

    return indicators


def rebalance(unbalanced_data):

    # Separate majority and minority classes
    data_minority = unbalanced_data[unbalanced_data.target==0]
    data_majority = unbalanced_data[unbalanced_data.target==1]

    # Upsample minority class
    n_samples = len(data_majority)
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=n_samples, random_state=5)

    # Combine majority class with upsampled minority class
    data_upsampled = pd.concat([data_majority, data_minority_upsampled])

    data_upsampled.sort_index(inplace=True)

    # Display new class counts
    data_upsampled.target.value_counts()

    return data_upsampled


def normalize(x):

    scaler = StandardScaler()
    x_norm = scaler.fit_transform(x.values)
    x_norm = pd.DataFrame(x_norm, index=x.index, columns=x.columns)

    return x_norm


def scores(models, X, y):

    for model in models:
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        print("Accuracy Score: {0:0.2f} %".format(acc * 100))
        print("F1 Score: {0:0.4f}".format(f1))
        print("Area Under ROC Curve Score: {0:0.4f}".format(auc))


symbol = 'SPX'  # S&P500
data = get_data(symbol)
data.tail(10)
data.describe()
ax = data['close'].plot(figsize=(9, 5))
ax.set_ylabel("Price ($)")
ax.set_xlabel("Time")
plt.show()

data_train = data['2011-01-01':'2017-01-01']
data_train = rebalance(data_train)
y = data_train.target
X = data_train.drop('target', axis=1)
X = normalize(X)
data_val = data['2017-01-01':]
y_val = data_val.target
X_val = data_val.drop('target', axis=1)
X_val = normalize(X_val)

# Machine Learning

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/6, shuffle=False)

models = [GaussianNB(),
          SVC(random_state=5),
          RandomForestClassifier(random_state=5),
          MLPClassifier(random_state=5)]

for model in models:
    model.fit(X_train, y_train)

scores(models, X_test, y_test)

# Grid search
grid_data = [[{'kernel': ['poly'], 'degree': [1, 2, 3, 4], 'C': [0.1, 1, 10, 100], 'random_state': [5]},
              {'kernel': ['rbf', 'sigmoid'], 'C': [0.1, 1, 10, 100], 'random_state': [5]}],
              {'n_estimators': [10, 50, 100],
               'criterion': ['gini', 'entropy'],
               'max_depth': [None, 10, 50, 100],
               'min_samples_split': [2, 5, 10],
               'random_state': [5]},
              {'hidden_layer_sizes': [10, 50, 100],
               'activation': ['identity', 'logistic', 'tanh', 'relu'],
               'solver': ['lbfgs', 'sgd', 'adam'],
               'learning_rate': ['constant', 'invscaling', 'adaptive'],
               'max_iter': [200, 400, 800],
               'random_state': [5]}]
models_grid = list()
for i in range(3):
    grid = GridSearchCV(models[i], grid_data[i], scoring='f1').fit(X_train, y_train)
    print(grid.best_params_)
    model = grid.best_estimator_
    models_grid.append(model)
scores(models_grid, X_test, y_test)

# Validation data
scores(models_grid, X_val, y_val)

# Trading system
rf_model = models_grid[0]
y_pred = rf_model.predict(X_val)
mask = y_pred.copy()
np.place(mask, y_pred==0, -1)
mask = np.roll(mask, 1)
data_returns = data['close'].diff()
data_returns = data_returns[X_val.index]
model_returns = mask * data_returns
model_cum = model_returns.cumsum()
equity = model_returns.sum()
start_close = data["close"][X_val.index[0]]
performance = equity / start_close * 100
ax = model_returns.plot(figsize=(9, 5))
ax.set_ylabel("Returns ($)")
ax.set_xlabel("Time")
plt.show()
ax = model_cum.plot(figsize=(9, 5))
ax.set_ylabel("Cummulative returns ($)")
ax.set_xlabel("Time")
plt.show()
