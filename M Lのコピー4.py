import os
import time
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
        print(f"データ取得エラー: {e}")
        return pd.DataFrame()

# === 2. データ前処理関数 ===
def add_cumret(df):
    df['cum_ret'] = df['close'].pct_change().cumsum()
    return df

def scale_features(df, features):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled

# === 3. 特徴量生成 ===
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
    df['CLOSE_lag_1'] = df['close'].shift(1)  # 1日遅れの終値 # Changed df[close_col] to df['close']
    df['CLOSE_lag_5'] = df['close'].shift(5)  # 5日遅れの終値 # Changed df[close_col] to df['close']
    df['MOVIENG_avg_5'] = df['close'].rolling(window=5).mean()  # 5日移動平均 # Changed df[close_col] to df['close']
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

    # 例: 翌日の終値が当日の終値より高ければ1、そうでなければ0
    df['long_target'] = (df['close'].shift(-1) > df['close']).astype(int)

    df.dropna(inplace=True)
    return df

# === 3. 相関係数による特徴量削除関数 ===
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

    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold) and column not in exclude_columns]
    print(f"🛠️ 削除された高相関特徴量: {to_drop}")
    return df.drop(columns=to_drop)

# === LightGBM Sklearn Wrapper ===
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

# === LightGBM重要度で上位特徴量を選択 ===
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

# === モデル作成 ===
def create_model(model_type, params=None):
    if model_type == 'lightgbm':
        return lgb.LGBMClassifier(**params)
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

# === Optunaによるハイパーパラメータ最適化 ===
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

# === バックテスト ===
from scipy.special import softmax

# ストラテジー別にモデルを適用し、Softmax重みで統合した予測を用いたバックテスト
# === バックテスト関数 ===
def run_backtest(df, model, features):
    print("🧪 df[features].shape in backtest:", df[features].shape)
    print("🧪 features list:", features)
    print("🧪 first few rows of df[features]:\n", df[features].head())

    # トレーニング時の特徴量を使用
    predictions = model.predict(df[features])
    position = None
    entry_price = 0
    pnl = []
    trade_log = []
    equity_curve = [1.0]  # 資産曲線（初期資産を1.0とする）

    for i, pred in enumerate(predictions):
        close_price = df.iloc[i]['close']

        if position is None and pred > 0.5:
            position = 'long'
            entry_price = close_price
            trade_log.append(('BUY', close_price))

        if position == 'long':
            stop_loss = entry_price * (1 - STOP_LOSS)
            take_profit = entry_price * (1 + TAKE_PROFIT)
            if close_price <= stop_loss or close_price >= take_profit:
                profit = close_price - entry_price - (close_price * COMMISSION + close_price * SLIPPAGE)
                pnl.append(profit)
                position = None
                trade_log.append(('SELL', close_price))
                # 資産曲線を更新
                equity_curve.append(equity_curve[-1] * (1 + profit / entry_price))

    # シャープレシオの計算
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0

    # 最大ドローダウンの計算
    equity_curve_array = np.array(equity_curve)
    drawdown = equity_curve_array / np.maximum.accumulate(equity_curve_array) - 1
    max_drawdown = drawdown.min()

    # 総損益と勝率
    total_pnl = np.sum(pnl)
    win_rate = np.mean(np.array(pnl) > 0) if pnl else 0

    # 結果を辞書形式で返す
    return {
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve,
    }

def run_walk_forward_backtest(df, model, features, n_splits=5):
    """
    ウォークフォワード分析を実行する。
    :param df: データフレーム
    :param model: モデル
    :param features: 使用する特徴量のリスト
    :param n_splits: ウォークフォワードの分割数
    :return: 各期間のバックテスト結果のリスト
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        print(f"=== ウォークフォワード期間 {fold + 1}/{n_splits} ===")

        # トレーニングデータとテストデータに分割
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        # モデルのトレーニング
        model.fit(train_df[features], train_df['long_target'])

        # テストデータでバックテストを実行
        backtest_result = run_backtest(test_df, model, features)
        results.append(backtest_result)

        # 各期間の結果を出力
        print(f"期間 {fold + 1} の結果:")
        print(f"総損益: {backtest_result['total_pnl']:.2f}")
        print(f"勝率: {backtest_result['win_rate']:.2%}")
        print(f"シャープレシオ: {backtest_result['sharpe_ratio']:.2f}")
        print(f"最大ドローダウン: {backtest_result['max_drawdown']:.2%}")

    return results

# === メイン処理 ===
def main():
    start_time = time.time()

    # データ取得
    df = get_data(TICKER, START_DATE, END_DATE, INTERVAL)
    if df is None:
        print("データが取得できませんでした。処理を終了します。")
        return

    print("[DEBUG] fetch_data 後のデータフレーム:")
    print(df.head())
    print("[DEBUG] fetch_data 後の列名:", df.columns.tolist())

    # 特徴量生成
    df = calc_features(df)

    print("🔍 特徴量生成後の列:", df.columns.tolist())

    # 相関係数による特徴量削除
    df = remove_highly_correlated_features(df, exclude_columns=['close'])

    print("🔍 高相関特徴量削除後の列:", df.columns.tolist())

    # 目的変数を除いた全列名をFEATURESに
    FEATURES = [col for col in df.columns if col != 'long_target']

    # LightGBMで上位特徴量を選択
    top_features = select_top_features_with_lightgbm(df, target='long_target', top_n=10)

    # FEATURES を top_features に設定
    FEATURES = top_features
    print("🔍 使用する特徴量:", FEATURES)

    # スケーリング
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['close_scaled'] = scaler.fit_transform(df[['close']])

    # モデル最適化
    best_params_dict = {model: optimize_hyperparameters(df, model) for model in SELECTED_MODELS}

    # モデル作成
    base_models = [(name, create_model(name, best_params_dict[name])) for name in SELECTED_MODELS]

    # 各モデルのトレーニング
    for name, model in base_models:
        model.fit(df[FEATURES], df['long_target'])

    # アンサンブルモデル
    if ENSEMBLE_TYPE == 'stacking':
        ensemble_model = StackingClassifier(estimators=base_models, final_estimator=MLPClassifier(max_iter=500))
    elif ENSEMBLE_TYPE == 'voting_hard':
        ensemble_model = VotingClassifier(estimators=base_models, voting='hard')
    elif ENSEMBLE_TYPE == 'voting_soft':
        ensemble_model = VotingClassifier(estimators=base_models, voting='soft')
    else:
        raise ValueError(f"Unsupported ensemble type: {ENSEMBLE_TYPE}")

    # バックテスト
    ensemble_model.fit(df[FEATURES], df['long_target'])
    backtest_results = run_backtest(df, ensemble_model, FEATURES)

    # バックテスト結果の出力
    print("\nバックテスト結果:")
    print(f"総損益: {backtest_results['total_pnl']:.2f}")
    print(f"勝率: {backtest_results['win_rate']:.2%}")
    print(f"シャープレシオ: {backtest_results['sharpe_ratio']:.2f}")
    print(f"最大ドローダウン: {backtest_results['max_drawdown']:.2%}")

    end_time = time.time()
    print(f"✅ 全体の実行時間: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
