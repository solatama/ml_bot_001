import os
import time
import logging
import numpy as np
import pandas as pd
from math import factorial
import matplotlib.pyplot as plt
import talib
import yfinance as yf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import ttest_1samp
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import optuna
from optuna.pruners import MedianPruner
from tqdm import tqdm

# === 定数定義 ===
TICKER = '9684.T'
START_DATE = '2024-04-01'
END_DATE = '2025-04-10'
INTERVAL = '1d'
FEATURES = ['close_scaled']
STOP_LOSS = 0.02
TAKE_PROFIT = 0.05
COMMISSION = 0.001
SLIPPAGE = 0.001
SELECTED_MODELS = ['xgboost', 'randomforest', 'catboost', 'lightgbm'] #'lightgbm',
ENSEMBLE_TYPE = 'stacking'  # 'blending', 'stacking', 'voting_hard', 'voting_soft'

# === 1. データ取得関数 ===
def get_data(ticker, start_date, end_date, interval='1d'):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [col.lower() for col in df.columns]
        return df
    except Exception as e:
        print(f"[エラー] データ取得に失敗しました: {e}")
        raise  # エラーをそのまま投げて処理を中断

# === 2. データ前処理関数 ===
def add_cumret(df):
    df['cum_ret'] = df['close'].pct_change().cumsum()
    return df

#=== 3. スケーリング関数 ===
def scale_features(df, features):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled

def basic_preprocessing(df):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df['close_scaled'] = scaler.fit_transform(df[['close']])
    return df

# === 4. 特徴量生成 ===
def calc_features(df):
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"必要な列がデータフレームに存在しません: {missing_columns}")

    # 必要なカラムを取得し、numpy.ndarray 型に変換
    open_col = df['open'].values.astype(float)
    high_col = df['high'].values.astype(float)
    low_col = df['low'].values.astype(float)
    close_col = df['close'].values.astype(float)
    volume = df['volume'].values.astype(float)

    orig_columns = df.columns

    hilo = (df['high'] + df['low']) / 2
    # 価格(hilo または close)を引いた後、価格(close)で割ることで標準化してるものあり

    # ボリンジャーバンド
    df['BBANDS_upperband'], df['BBANDS_middleband'], df['BBANDS_lowerband'] = talib.BBANDS(df['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['BBANDS_upperband'] = (df['BBANDS_upperband'] - hilo) / df['close']
    df['BBANDS_middleband'] = (df['BBANDS_middleband'] - hilo) / df['close']
    df['BBANDS_lowerband'] = (df['BBANDS_lowerband'] - hilo) / df['close']

    # 移動平均
    df['DEMA'] = (talib.DEMA(close_col, timeperiod=30) - hilo) / close_col
    df['EMA'] = (talib.EMA(close_col, timeperiod=30) - hilo) / close_col
    df['EMA_short'] = (talib.EMA(close_col, timeperiod=5) - hilo) / close_col
    df['EMA_middle'] = (talib.EMA(close_col, timeperiod=20) - hilo) / close_col
    df['EMA_long'] = (talib.EMA(close_col, timeperiod=40) - hilo) / close_col
    df['HT_TRENDLINE'] = (talib.HT_TRENDLINE(close_col) - hilo) / close_col
    df['KAMA'] = (talib.KAMA(close_col, timeperiod=30) - hilo) / close_col
    df['MA'] = (talib.MA(close_col, timeperiod=30, matype=0) - hilo) / close_col
    df['MIDPOINT'] = (talib.MIDPOINT(close_col, timeperiod=14) - hilo) / close_col
    df['SMA'] = (talib.SMA(close_col, timeperiod=30) - hilo) / close_col
    df['T3'] = (talib.T3(close_col, timeperiod=5, vfactor=0) - hilo) / close_col
    df['HMA'] = talib.WMA(close_col, timeperiod=30)
    df['TEMA'] = (talib.TEMA(close_col, timeperiod=30) - hilo) / close_col
    df['TRIMA'] = (talib.TRIMA(close_col, timeperiod=30) - hilo) / close_col
    df['WMA'] = (talib.WMA(close_col, timeperiod=30) - hilo) / close_col

    # MACD
    df['MACD_macd'], df['MACD_macdsignal'], df['MACD_macdhist'] = talib.MACD(close_col, fastperiod=12, slowperiod=26, signalperiod=9) # Use close_col instead of close
    df['MACD_macd'] /= close_col # Use close_col instead of close
    df['MACD_macdsignal'] /= close_col # Use close_col instead of close
    df['MACD_macdhist'] /= close_col # Use close_col instead of close
    df['MACD_EXT'], df['MACD_SIGNAL_EXT'], df['MACD_HIST_EXT'] = talib.MACDEXT(close_col, fastperiod=12, slowperiod=26, signalperiod=9, fastmatype=0, slowmatype=0, signalmatype=0) # Use close_col instead of close

    # 線形回帰系
    df['LINEARREG'] = (talib.LINEARREG(close_col, timeperiod=14) - close_col) / close_col # Use close_col instead of close
    df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close_col, timeperiod=14) / close_col # Use close_col instead of close
    df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close_col, timeperiod=14) # Use close_col instead of close
    df['LINEARREG_INTERCEPT'] = (talib.LINEARREG_INTERCEPT(close_col, timeperiod=14) - close_col) / close_col # Use close_col instead of close

    # AD系
    df['AD'] = talib.AD(high_col, low_col, close_col, volume) / close_col # Use high_col, low_col, close_col instead of high, low, close
    df['ADX'] = talib.ADX(high_col, low_col, close_col, timeperiod=14) # Use high_col, low_col, close_col instead of high, low, close
    df['ADXR'] = talib.ADXR(high_col, low_col, close_col, timeperiod=14) # Use high_col, low_col, close_col instead of high, low, close
    df['ADOSC'] = talib.ADOSC(high_col, low_col, close_col, volume, fastperiod=3, slowperiod=10) / close_col # Use high_col, low_col, close_col instead of high, low, close
    df['OBV'] = talib.OBV(close_col, volume) / close_col # Use close_col instead of close

    # オシレーター系
    df['APO'] = talib.APO(close_col, fastperiod=12, slowperiod=26, matype=0) / close_col  # Changed close to close_col
    df['BOP'] = talib.BOP(open_col, high_col, low_col, close_col)  # Changed open, high, low, close to their respective _col variables
    df['CCI'] = talib.CCI(high_col, low_col, close_col, timeperiod=14)  # Changed high, low, close to their respective _col variables
    df['DX'] = talib.DX(high_col, low_col, close_col, timeperiod=14)  # Changed high, low, close to their respective _col variables
    df['MFI'] = talib.MFI(high_col, low_col, close_col, volume, timeperiod=14)  # Changed high, low, close to their respective _col variables
    df['MINUS_DI'] = talib.MINUS_DI(high_col, low_col, close_col, timeperiod=14)  # Changed high, low, close to their respective _col variables
    df['PLUS_DI'] = talib.PLUS_DI(high_col, low_col, close_col, timeperiod=14)  # Changed high, low, close to their respective _col variables
    df['MOM'] = talib.MOM(close_col, timeperiod=10) / close_col  # Changed close to close_col
    df['RSI'] = talib.RSI(close_col, timeperiod=14)  # Changed close to close_col
    df['TRIX'] = talib.TRIX(close_col, timeperiod=30)  # Changed close to close_col
    df['ULTOSC'] = talib.ULTOSC(high_col, low_col, close_col, timeperiod1=7, timeperiod2=14, timeperiod3=28)  # Changed high, low, close to their respective _col variables
    df['WILLR'] = talib.WILLR(high_col, low_col, close_col, timeperiod=14)  # Changed high, low, close to their respective _col variables
    df['SAR'] = talib.SAR(high_col, low_col, acceleration=0.02, maximum=0.2)  # Changed high, low to their respective _col variables

    # ストキャスティクス
    df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(high_col, low_col, close_col, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0) # Changed high, low, close to high_col, low_col, close_col
    df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(high_col, low_col, close_col, fastk_period=5, fastd_period=3, fastd_matype=0) # Changed high, low, close to high_col, low_col, close_col
    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(close_col, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0) # Changed close to close_col

    # ボラティリティ系
    df['MINUS_DM'] = talib.MINUS_DM(high_col, low_col, timeperiod=14) / close_col # Changed high, low, close to high_col, low_col, close_col
    df['PLUS_DM'] = talib.PLUS_DM(high_col, low_col, timeperiod=14) / close_col # Changed high, low, close to high_col, low_col, close_col
    df['STDDEV'] = talib.STDDEV(close_col, timeperiod=5, nbdev=1) # Changed close to close_col
    df['TRANGE'] = talib.TRANGE(high_col, low_col, close_col) # Changed high, low, close to high_col, low_col, close_col
    df['VAR'] = talib.VAR(close_col, timeperiod=5, nbdev=1) # Changed close to close_col
    df['ATR'] = talib.ATR(high_col, low_col, close_col, timeperiod=14) # Changed high, low, close to high_col, low_col, close_col
    df['NATR'] = talib.NATR(high_col, low_col, close_col, timeperiod=14) # Changed high, low, close to high_col, low_col, close_col
    df['VOLATILITY_index'] = df['ATR'] / df['STDDEV']

    # ヒルベルト変換
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_col) # Changed close to close_col
    df['HT_DCPHASE'] = talib.HT_DCPHASE(close_col) # Changed close to close_col
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close_col) # Changed close to close_col
    df['HT_PHASOR_inphase'] /= close_col # Changed close to close_col
    df['HT_PHASOR_quadrature'] /= close_col # Changed close to close_col
    df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(close_col) # Changed close to close_col
    df['HT_SINE_sine'] /= close_col # Changed close to close_col
    df['HT_SINE_leadsine'] /= close_col # Changed close to close_col
    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close_col) # Changed close to close_col

    # その他
    df['ROC'] = talib.ROC(close_col, timeperiod=10) / close_col # Changed close to close_col
    df['STDDEV'] = talib.STDDEV(close_col, timeperiod=5, nbdev=1) / close_col # Changed close to close_col
    df['TRANGE'] = talib.TRANGE(high_col, low_col, close_col) / close_col # Changed high, low, close to high_col, low_col, close_col
    df['AROON_aroondown'], df['AROON_aroonup'] = talib.AROON(high_col, low_col, timeperiod=14) # Changed high, low to high_col, low_col
    df['AROONOSC'] = talib.AROONOSC(high_col, low_col, timeperiod=14) # Changed high, low to high_col, low_col
    df['BETA'] = talib.BETA(high_col, low_col, timeperiod=5) # Changed high, low to high_col, low_col
    df['CORREL'] = talib.CORREL(high_col, low_col, timeperiod=30) # Changed high, low to high_col, low_col
    df['Price_ratio'] = df['close'] / df['close'].shift(1)  # Changed df[close_col] to df['close']
    df['HIGH_ratio'] = df['high'] / df['high'].shift(1)  # Changed df[high_col] to df['high']
    df['LOW_ratio'] = df['low'] / df['low'].shift(1)  # Changed df[low_col] to df['low']

    # Lag特徴量
    lags = [1, 3, 5, 10, 20]
    for lag in lags:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag)
        df[f'log_return_lag_{lag}'] = np.log(df['close'] / df['close'].shift(lag))
    # 周期性の特徴量
    df['DAY_of_week'] = df.index.dayofweek  # 曜日（0=月曜日, 6=日曜日）
    df['IS_weekend'] = (df['DAY_of_week'] >= 5).astype(int)  # 週末かどうか
    df['MONTH'] = df.index.month  # 月（1〜12）
    df['SIN_day'] = np.sin(2 * np.pi * df['DAY_of_week'] / 7)  # 日周期
    df['COS_day'] = np.cos(2 * np.pi * df['DAY_of_week'] / 7)  # 日周期
    df['SIN_month'] = np.sin(2 * np.pi * df['MONTH'] / 12)  # 月周期
    df['COS_month'] = np.cos(2 * np.pi * df['MONTH'] / 12)  # 月周期

    # リターン系
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
    df['return_20d'] = df['close'].pct_change(20)

    df['return_ma_5'] = df['return_1d'].rolling(window=5).mean()
    df['return_ma_10'] = df['return_1d'].rolling(window=10).mean()

    df['volatility_5'] = df['close'].rolling(window=5).std()
    df['volatility_10'] = df['close'].rolling(window=10).std()

    df['range'] = df['high'] - df['low']

    df['volume_change'] = df['volume'].pct_change(1)

    # トレンド・モメンタム系
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_25'] = df['close'].rolling(window=25).mean()
    df['ma_75'] = df['close'].rolling(window=75).mean()

    df['ma_deviation_5'] = df['close'] / df['ma_5'] - 1
    df['ma_deviation_25'] = df['close'] / df['ma_25'] - 1
    df['ma_deviation_75'] = df['close'] / df['ma_75'] - 1

    # モメンタム（n日前比）
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)

    # --- ローソク足パターン（50種類）---
    # 値は +100: 強気シグナル, -100: 弱気シグナル, 0: シグナルなし
    candlestick_patterns = {
        'CDL2CROWS': talib.CDL2CROWS,
        'CDL3BLACKCROWS': talib.CDL3BLACKCROWS,
        'CDL3INSIDE': talib.CDL3INSIDE,
        'CDL3LINESTRIKE': talib.CDL3LINESTRIKE,
        'CDL3OUTSIDE': talib.CDL3OUTSIDE,
        'CDL3STARSINSOUTH': talib.CDL3STARSINSOUTH,
        'CDL3WHITESOLDIERS': talib.CDL3WHITESOLDIERS,
        'CDLABANDONEDBABY': talib.CDLABANDONEDBABY,
        'CDLADVANCEBLOCK': talib.CDLADVANCEBLOCK,
        'CDLBELTHOLD': talib.CDLBELTHOLD,
        'CDLBREAKAWAY': talib.CDLBREAKAWAY,
        'CDLCLOSINGMARUBOZU': talib.CDLCLOSINGMARUBOZU,
        'CDLCONCEALBABYSWALL': talib.CDLCONCEALBABYSWALL,
        'CDLCOUNTERATTACK': talib.CDLCOUNTERATTACK,
        'CDLDARKCLOUDCOVER': talib.CDLDARKCLOUDCOVER,
        'CDLDOJI': talib.CDLDOJI,
        'CDLDOJISTAR': talib.CDLDOJISTAR,
        'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI,
        'CDLENGULFING': talib.CDLENGULFING,
        'CDLEVENINGDOJISTAR': talib.CDLEVENINGDOJISTAR,
        'CDLEVENINGSTAR': talib.CDLEVENINGSTAR,
        'CDLGAPSIDESIDEWHITE': talib.CDLGAPSIDESIDEWHITE,
        'CDLGRAVESTONEDOJI': talib.CDLGRAVESTONEDOJI,
        'CDLHAMMER': talib.CDLHAMMER,
        'CDLHANGINGMAN': talib.CDLHANGINGMAN,
        'CDLHARAMI': talib.CDLHARAMI,
        'CDLHARAMICROSS': talib.CDLHARAMICROSS,
        'CDLHIGHWAVE': talib.CDLHIGHWAVE,
        'CDLHIKKAKE': talib.CDLHIKKAKE,
        'CDLHIKKAKEMOD': talib.CDLHIKKAKEMOD,
        'CDLHOMINGPIGEON': talib.CDLHOMINGPIGEON,
        'CDLIDENTICAL3CROWS': talib.CDLIDENTICAL3CROWS,
        'CDLINNECK': talib.CDLINNECK,
        'CDLINVERTEDHAMMER': talib.CDLINVERTEDHAMMER,
        'CDLKICKING': talib.CDLKICKING,
        'CDLKICKINGBYLENGTH': talib.CDLKICKINGBYLENGTH,
        'CDLLADDERBOTTOM': talib.CDLLADDERBOTTOM,
        'CDLLONGLEGGEDDOJI': talib.CDLLONGLEGGEDDOJI,
        'CDLLONGLINE': talib.CDLLONGLINE,
        'CDLMARUBOZU': talib.CDLMARUBOZU,
        'CDLMATCHINGLOW': talib.CDLMATCHINGLOW,
        'CDLMATHOLD': talib.CDLMATHOLD,
        'CDLMORNINGDOJISTAR': talib.CDLMORNINGDOJISTAR,
        'CDLMORNINGSTAR': talib.CDLMORNINGSTAR,
        'CDLONNECK': talib.CDLONNECK,
        'CDLPIERCING': talib.CDLPIERCING,
        'CDLRICKSHAWMAN': talib.CDLRICKSHAWMAN,
        'CDLRISEFALL3METHODS': talib.CDLRISEFALL3METHODS,
        'CDLSEPARATINGLINES': talib.CDLSEPARATINGLINES,
        'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
    }

    # パターンを一括で計算して DataFrame に追加
    for name, func in candlestick_patterns.items():
        df[name] = func(open_col, high_col, low_col, close_col)

    # スケーリングされた 'close' 列を追加
    scaler = MinMaxScaler()
    df['close_scaled'] = scaler.fit_transform(df[['close']])

    # 例: 翌日の終値が当日の終値より高ければ1、そうでなければ0
    df['long_target'] = (df['close'].shift(-1) > df['close']).astype(int)

    df.dropna(inplace=True)
    return df

# === 5. 相関係数による特徴量削除関数 ===
def remove_highly_correlated_features(df, threshold=0.9, exclude_columns=None):
    """
    高い相関を持つ特徴量を削除する。
    :param df: 特徴量データフレーム
    :param threshold: 相関係数の閾値
    :param exclude_columns: 削除対象から除外する列のリスト
    :return: 相関が高くない特徴量を持つデータフレーム
    """
    if exclude_columns is None:
        exclude_columns = []

    # 相関行列を計算
    corr_matrix = df.corr().abs()

    # 上三角行列を取得
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 高相関の列を特定
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold) and column not in exclude_columns]

    print(f"🛠️ 削除された高相関特徴量: {to_drop}")

    # 高相関の列を削除
    return df.drop(columns=to_drop)

# === 6. LightGBM Sklearn Wrapper ===
class LightGBMSklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **params):
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# === 7. LightGBM重要度で上位特徴量を選択 ===
def select_top_features_with_lightgbm(df, target, top_n=10):
    """
    LightGBMの重要度を用いて上位の特徴量を選択する。
    :param df: 特徴量データフレーム
    :param target: ターゲット列名
    :param top_n: 選択する特徴量の数
    :return: 上位特徴量のリスト
    """
    X = df.drop(columns=[target])
    y = df[target]

    model = lgb.LGBMClassifier()
    model.fit(X, y)

    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    top_features = feature_importances.head(top_n)['feature'].tolist()
    print(f"🌟 LightGBMで選ばれた上位 {top_n} 特徴量: {top_features}")
    return top_features

# === 8. t検定関数 ===
def perform_t_test(df):
    x = df['cum_ret'].diff(1).dropna()  # 累積リターンの差分を計算
    t, p = ttest_1samp(x, 0)  # t検定を実行
    return t, p

# === 9. p平均法計算関数 ===
def calc_p_mean(x, n):
    ps = []
    for i in range(n):
        x2 = x[i * x.size // n:(i + 1) * x.size // n]
        if np.std(x2) == 0:
            ps.append(1)
        else:
            t, p = ttest_1samp(x2, 0)
            if t > 0:
                ps.append(p)
            else:
                ps.append(1)
    return np.mean(ps)

def calc_p_mean_type1_error_rate(p_mean, n):
    return (p_mean * n) ** n / math.factorial(n)

# === 10. モデル作成 ===
def create_model(model_type, params=None):
    if model_type == 'lightgbm':
        return lgb.LGBMClassifier(verbose=-1, **params)
    elif model_type == 'xgboost':
        return xgb.XGBClassifier(**params)
    elif model_type == 'catboost':
        return CatBoostClassifier(verbose=0, **params)
    elif model_type == 'randomforest':
        return RandomForestClassifier(**params)
    elif model_type == 'mlp':
        return MLPClassifier(max_iter=500, **params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# === 10. モデル作成
def create_base_models(selected_models, best_params_dict):
    return [(name, create_model(name, best_params_dict[name])) for name in selected_models]

# === 11. Optunaによるハイパーパラメータ最適化 ===
def optimize_hyperparameters(df, model_type):
    def objective(trial):
        # モデルごとに異なるハイパーパラメータを設定
        if model_type == 'lightgbm':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
            }
        elif model_type == 'xgboost':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
            }
        elif model_type == 'catboost':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'depth': trial.suggest_int('depth', 3, 10),
                'iterations': trial.suggest_int('iterations', 100, 1000),
            }
        elif model_type == 'randomforest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            }
        else:
            raise ValueError(f"Unsupported model type for optimization: {model_type}")

        # モデルを作成してスコアを計算
        model = create_model(model_type, params)
        scores = []
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, test_idx in tscv.split(df):
            X_train, X_test = df.iloc[train_idx][FEATURES], df.iloc[test_idx][FEATURES]
            y_train, y_test = df.iloc[train_idx]['long_target'], df.iloc[test_idx]['long_target']
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_test)[:, 1]
            scores.append(log_loss(y_test, preds))
        return np.mean(scores)

    # Optunaで最適化
    study = optuna.create_study(direction='minimize', pruner=MedianPruner())
    study.optimize(objective, n_trials=30)
    return study.best_params

# === 12. p平均法によるストラテジー有意性検定 ===
def p_mean_test(returns, period=14, alpha=0.03):
    """
    p平均法によるストラテジー有意性検定
    Parameters:
        returns: リターンの時系列 (1次元 array-like)
        period: 検定に使う期間の長さ（日数）
        alpha: 有意水準 (例: 0.03)
    Returns:
        mean_p: p値の平均
        significant: 有意かどうか（mean_p < alpha）
        error_rate: エラー率の近似値
    """
    returns = np.array(returns)
    n = len(returns) // period
    p_values = []

    for i in range(n):
        chunk = returns[i*period : (i+1)*period]
        if len(chunk) < 2:  # t検定には2点以上必要
            continue
        _, p = ttest_1samp(chunk, popmean=0, alternative='greater')
        p_values.append(p)

    if not p_values:
        raise ValueError("有効な期間がありません。データが少なすぎます。")

    mean_p = np.mean(p_values)
    significant = mean_p < alpha

    # エラー率の近似 (mean(p)*N)^N / N!
    N = len(p_values)
    error_rate = (mean_p * N)**N / factorial(N)

    return mean_p, significant, error_rate

# === 13. バックテスト ===
from scipy.special import softmax

# === 14. バックテスト関数 ===
def run_backtest(df, model, features):
    predictions = model.predict(df[features])
    position = None
    entry_price = 0
    pnl = []
    equity_curve = [1.0]

    for i, pred in enumerate(predictions):
        close_price = df.iloc[i]['close']

        if position is None and pred > 0.5:
            position = 'long'
            entry_price = close_price

        if position == 'long':
            stop_loss = entry_price * (1 - STOP_LOSS)
            take_profit = entry_price * (1 + TAKE_PROFIT)
            if close_price <= stop_loss or close_price >= take_profit:
                profit = close_price - entry_price - (entry_price * COMMISSION + close_price * SLIPPAGE)
                pnl.append(profit)
                position = None
                equity_curve.append(equity_curve[-1] * (1 + profit / entry_price))

    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
    equity_curve_array = np.array(equity_curve)
    drawdown = equity_curve_array / np.maximum.accumulate(equity_curve_array) - 1
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
    total_pnl = np.sum(pnl)
    win_rate = np.mean(np.array(pnl) > 0) if pnl else 0

    return {
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve,
    }

# 可視化関数
def plot_equity_curve(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df['cum_ret'], label='Equity Curve')
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    plt.show()

def plot_feature_heatmap(df, features):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()

# === 15. モデルのトレーニングと評価 ===
def train_and_evaluate_model(df, features, target, model_class, test_size=0.2):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = model_class()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# === 16. ウォークフォワード分析 ===
from sklearn.base import clone

def run_walk_forward_backtest(df, model, features, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []
    train_returns = []
    test_returns = []

    for train_idx, test_idx in tqdm(tscv.split(df)):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

        cloned_model = clone(model)
        cloned_model.fit(train_df[features], train_df['long_target'])

        train_bt = run_backtest(train_df, cloned_model, features)
        train_equity = np.array(train_bt['equity_curve'])
        train_return = train_equity[-1] / train_equity[0] - 1
        train_returns.append(train_return)

        test_bt = run_backtest(test_df, cloned_model, features)
        fold_results.append(test_bt)

        test_equity = np.array(test_bt['equity_curve'])
        test_return = test_equity[-1] / test_equity[0] - 1
        test_returns.append(test_return)

    train_returns = np.array(train_returns)
    test_returns = np.array(test_returns)
    wfe_list = test_returns / train_returns
    wfe = np.mean(wfe_list)

    return fold_results, wfe

# === 17. モンテカルロシミュレーション =====
def monte_carlo_simulation(equity_curve, n_simulations=1000):
    final_returns = []
    max_drawdowns = []

    for _ in range(n_simulations):
        shuffled_returns = np.random.permutation(np.diff(equity_curve) / equity_curve[:-1])
        sim_curve = [1.0]
        for r in shuffled_returns:
            sim_curve.append(sim_curve[-1] * (1 + r))

        sim_curve = np.array(sim_curve)
        final_return = sim_curve[-1] - 1
        drawdown = sim_curve / np.maximum.accumulate(sim_curve) - 1
        max_dd = drawdown.min()

        final_returns.append(final_return)
        max_drawdowns.append(max_dd)

    return np.array(final_returns), np.array(max_drawdowns)

# === 18. アウトオブサンプルテスト =====
def run_oos_test(df, model_class, features, train_ratio=0.8):
    n_train = int(len(df) * train_ratio)
    train_df = df.iloc[:n_train]
    oos_df = df.iloc[n_train:]

    model = model_class()
    model.fit(train_df[features], train_df['long_target'])

    oos_bt = run_backtest(oos_df, model, features)

    return oos_bt

# === 19. ロバスト最適化スコア計算 =====
def calculate_robust_score(fold_results):
    sharpe_ratios = [r['sharpe_ratio'] for r in fold_results]
    mean_sr = np.mean(sharpe_ratios)
    std_sr = np.std(sharpe_ratios)
    robust_score = mean_sr - std_sr
    return robust_score

# === 17. バックテスト結果のプロット ===def plot_capital_curve(backtest_results):
def plot_capital_curve(backtest_results):
    plt.figure(figsize=(10, 5))
    plt.plot(backtest_results['equity_curve'], label='Equity Curve')
    plt.title('資産推移')
    plt.xlabel('期間')
    plt.ylabel('資産')
    plt.legend()
    plt.grid()
    plt.show()

# === 18. バックテスト結果の表示 ===
def display_backtest_results(results):
    print("\nバックテスト結果:")
    print(f"{'項目':<15} {'値':>10}")
    print(f"{'-'*25}")
    for key, value in results.items():
        if isinstance(value, list):  # 値がリストの場合
            print(f"{key:<15} {len(value):>10} (リストの長さ)")
        elif isinstance(value, (int, float)):  # 値が数値の場合
            print(f"{key:<15} {value:>10.2f}")
        else:  # その他の型の場合
            print(f"{key:<15} {str(value):>10}")

# === 19. バックテスト結果の保存 ===
def save_backtest_results(backtest_results, filename="backtest_results.csv"):
    # equity_curve の長さに合わせて他の列を埋める
    equity_curve_length = len(backtest_results["equity_curve"])
    equity_df = pd.DataFrame({
        "equity_curve": backtest_results["equity_curve"],
        "total_pnl": [backtest_results["total_pnl"]] * equity_curve_length,
        "win_rate": [backtest_results["win_rate"]] * equity_curve_length,
        "sharpe_ratio": [backtest_results["sharpe_ratio"]] * equity_curve_length,
        "max_drawdown": [backtest_results["max_drawdown"]] * equity_curve_length
    })
    equity_df.to_csv(filename, index=False)
    print(f"バックテスト結果を {filename} に保存しました。")

# === 20. ウォークフォワード分析結果の保存 ===
def save_walk_forward_results(walk_forward_results, filename="walk_forward_results.csv"):
    # 各期間の結果をデータフレームに変換
    results_list = []
    for i, result in enumerate(walk_forward_results):
        results_list.append({
            "fold": i + 1,
            "total_pnl": result["total_pnl"],
            "win_rate": result["win_rate"],
            "sharpe_ratio": result["sharpe_ratio"],
            "max_drawdown": result["max_drawdown"]
        })
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(filename, index=False)
    print(f"ウォークフォワード分析結果を {filename} に保存しました。")

# Optuna のロガーを取得してログレベルを設定
optuna_logger = optuna.logging.get_logger("optuna")
optuna_logger.setLevel(logging.WARNING)  # INFO ログを非表示にする

# === 21. メイン関数 ===
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

# 必要な自作関数（get_data, basic_preprocessing, calc_features, etc.）は別途importされている前提です

def main():
    start_time = time.time()

    steps = [
        "データ取得",
        "前処理（スケーリング）",
        "特徴量生成",
        "相関係数による特徴量削除",
        "累積リターン計算",
        "特徴量選択",
        "モデル最適化",
        "モデル作成",
        "各モデルのトレーニング",
        "アンサンブル作成",
        "バックテスト実行",
        "バックテスト結果保存",
        "ウォークフォワード分析",
        "ウォークフォワード結果保存",
        "モンテカルロシミュレーション",
        "ロバスト最適化",
        "t検定",
        "p平均法",
        "可視化"
    ]

    with tqdm(total=len(steps), desc="進捗状況") as pbar:
        # データ取得
        df = get_data(TICKER, START_DATE, END_DATE, INTERVAL)
        if df is None or df.empty:
            print("データが取得できませんでした。処理を終了します。")
            return
        pbar.update(1)

        # 前処理
        df = basic_preprocessing(df)
        pbar.update(1)

        # 特徴量生成
        df = calc_features(df)
        pbar.update(1)

        # 相関除去
        df = remove_highly_correlated_features(df, exclude_columns=['close', 'close_scaled'])
        pbar.update(1)

        # スケーリング
        FEATURES = [col for col in df.columns if col not in ['long_target', 'close', 'close_scaled']]
        scaler = MinMaxScaler()
        df[FEATURES] = scaler.fit_transform(df[FEATURES])
        pbar.update(1)

        # 累積リターン計算
        df = add_cumret(df)
        pbar.update(1)

        # 特徴量選択
        top_features = select_top_features_with_lightgbm(df, target='long_target', top_n=10)
        FEATURES = top_features
        print(f"🔍 LightGBM選抜特徴量: {FEATURES}")
        pbar.update(1)

        # モデル最適化
        best_params_dict = {model: optimize_hyperparameters(df, model) for model in SELECTED_MODELS}
        pbar.update(1)

        # モデル作成
        base_models = create_base_models(SELECTED_MODELS, best_params_dict)
        pbar.update(1)

        # 各モデルのトレーニング
        for name, model in base_models:
            model.fit(df[FEATURES], df['long_target'])
        pbar.update(1)

        # アンサンブル作成
        if ENSEMBLE_TYPE == 'stacking':
            ensemble_model = StackingClassifier(estimators=base_models, final_estimator=MLPClassifier(max_iter=1000))
        elif ENSEMBLE_TYPE == 'voting_hard':
            ensemble_model = VotingClassifier(estimators=base_models, voting='hard')
        elif ENSEMBLE_TYPE == 'voting_soft':
            ensemble_model = VotingClassifier(estimators=base_models, voting='soft')
        else:
            raise ValueError(f"Unsupported ensemble type: {ENSEMBLE_TYPE}")
        pbar.update(1)

        # バックテスト実行
        ensemble_model.fit(df[FEATURES], df['long_target'])
        backtest_results = run_backtest(df, ensemble_model, FEATURES)
        pbar.update(1)

        # バックテスト結果保存
        display_backtest_results(backtest_results)
        save_backtest_results(backtest_results, filename="backtest_results.csv")
        pbar.update(1)

        # ウォークフォワード分析
        fold_results, wfe = run_walk_forward_backtest(df, ensemble_model, FEATURES, n_splits=5)
        print(f"🛫 ウォークフォワード効率 (WFE): {wfe:.4f}")
        pbar.update(1)

        # ウォークフォワード結果保存
        save_walk_forward_results(fold_results, filename="walk_forward_results.csv")
        pbar.update(1)

        # モンテカルロシミュレーション
        mc_mean, mc_std = monte_carlo_simulation(df['cum_ret'], n_simulations=1000)
        print(f"🎲 モンテカルロ結果: 平均最終リターン={mc_mean[0]:.4f}, 標準偏差={mc_std[0]:.4f}")
        pbar.update(1)

        # ロバスト最適化
        robustness_score = calculate_robust_score(fold_results)
        print(f"🛡️ ロバストネススコア: {robustness_score:.4f}")
        pbar.update(1)

        # t検定
        t_stat, p_value = perform_t_test(df)
        print(f"🧪 t検定結果: t値={t_stat:.4f}, p値={p_value:.4f}")
        pbar.update(1)

        # p平均法
        x = df['cum_ret'].diff(1).dropna()
        period = 14
        alpha = 0.03
        mean_p, significant, error_rate = p_mean_test(x, period=period, alpha=alpha)
        print(f"📈 p平均法結果: 平均p値={mean_p:.4f}, 有意かどうか={significant}, エラー率={error_rate:.4e}")
        pbar.update(1)

        # 可視化
        plot_equity_curve(df)
        plot_feature_heatmap(df, FEATURES)
        pbar.update(1)

    end_time = time.time()
    print('========================================================')
    print(f"✅ 完了！全体の実行時間: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
