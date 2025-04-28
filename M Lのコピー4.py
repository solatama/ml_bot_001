# 必要なライブラリのインポート
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import talib
import time
import optuna
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# === 定数定義 ===
TICKER = '9684.T'
START_DATE = '2024-04-01'
END_DATE = '2025-04-10'
INTERVAL = '1d'

FEATURES = ['close_scaled']
STOP_LOSS_MODE = 1  # 1: 固定パーセント損切り, 2: パラボリックSAR損切り, 3: 直近安値更新
STOP_LOSS = 0.02    # 固定パーセント損切りでの損失許容率（例: 2%）
TAKE_PROFIT = 0.04  # 利確幅（例: 4%）
COMMISSION = 0.0005  # 手数料率（必要なら設定）
SLIPPAGE = 0.0005    # スリッページ率（必要なら設定）
SELECTED_MODELS = ['xgboost', 'randomforest', 'catboost', 'lightgbm', 'rnn', 'transformer']
ENSEMBLE_TYPE = 'stacking'  # 'blending', 'stacking', 'voting_hard', 'voting_soft'から選択

# yfinanceを使って株価データを取得
def get_data(ticker, start_date, end_date, interval='1d'):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # MultiIndex の場合にリセット
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 列名を小文字に変換
    df.columns = [col.lower() for col in df.columns]
    return df

# データ取得
df = get_data(TICKER, START_DATE, END_DATE, INTERVAL)
df.head()

# スケーリング（MinMaxScalerを使用）
def preprocess_data(df):
    scaler = MinMaxScaler()
    df['close_scaled'] = scaler.fit_transform(df[['close']])
    return df

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

# === 相関係数による特徴量削除関数 ===
def select_features_pipeline(df, target_column, corr_threshold=0.9, top_n=10):
    # 目的変数と 'close' 列を除いた特徴量データフレーム
    features_df = df.drop(columns=[target_column, 'close'])

    # 相関係数の計算
    corr_matrix = features_df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
    print(f"🛠️ 削除された高相関特徴量: {to_drop}")
    reduced_df = features_df.drop(columns=to_drop)

    # LightGBMによる特徴量重要度の計算
    X = reduced_df
    y = df[target_column]
    model = lgb.LGBMClassifier()
    model.fit(X, y)
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    # 上位特徴量の選択
    top_features = feature_importances.head(top_n)['feature'].tolist()
    print(f"🌟 LightGBMで選ばれた上位 {top_n} 特徴量: {top_features}")

    # 選択された特徴量と 'close' 列、目的変数を含むデータフレームの作成
    selected_df = df[top_features + ['close', target_column]]
    return selected_df

# モデル定義
def create_lgbm_model():
    return lgb.LGBMRegressor()

def create_xgboost_model():
    return xgb.XGBRegressor()

def create_catboost_model():
    return cb.CatBoostRegressor(silent=True)

def create_rf_model():
    return RandomForestRegressor()

# LSTMモデル定義
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def create_rnn_model(input_dim, hidden_dim=64, output_dim=1):
    return RNNModel(input_dim, hidden_dim, output_dim)

# Transformerモデル定義
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=hidden_dim, num_encoder_layers=3)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        transformer_out = self.transformer(x)
        return self.fc(transformer_out[:, -1, :])

def create_transformer_model(input_dim, hidden_dim=64, output_dim=1):
    return TransformerModel(input_dim, hidden_dim, output_dim)

# Optunaでハイパーパラメータチューニング
def optimize_model(trial, model_type, X_train, y_train):
    if model_type == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            'subsample': trial.suggest_float("subsample", 0.6, 1.0),
        }
        model = xgb.XGBRegressor(**params)
    elif model_type == 'lightgbm':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 30, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
        }
        model = lgb.LGBMRegressor(**params)
    elif model_type == 'catboost':
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            'depth': trial.suggest_int('depth', 3, 15),
        }
        model = cb.CatBoostRegressor(**params, silent=True)
    elif model_type == 'randomforest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        }
        model = RandomForestRegressor(**params)

    # クロスバリデーションを使用した評価
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
    return score.mean()

# アンサンブル手法（stacking）
def create_ensemble_model(models, ensemble_type):
    if ensemble_type == 'stacking':
        ensemble_model = StackingRegressor(estimators=models)
    elif ensemble_type == 'blending':
        ensemble_model = VotingRegressor(estimators=models)
    elif ensemble_type == 'voting_hard':
        ensemble_model = VotingRegressor(estimators=models, voting='hard')
    elif ensemble_type == 'voting_soft':
        ensemble_model = VotingRegressor(estimators=models, voting='soft')
    return ensemble_model

# バックテスト関数（仮の例）
# 売買ロジックを含むバックテスト関数
def run_backtest(df, model, features):
    # トレーニング時の特徴量を使用
    predictions = model.predict(df[features])
    position = None
    entry_price = 0
    pnl = []
    trade_log = []
    equity_curve = [1.0]  # 資産曲線（初期資産を1.0とする）

    for i, pred in enumerate(predictions):
        close_price = df.iloc[i]['close']  # 'close' 列が存在することを前提

        # 買いエントリー
        if position is None and pred == 1:  # 予測が1（翌日上昇予測）の場合
            position = 'long'
            entry_price = close_price
            trade_log.append(('BUY', close_price))

        # 売りエグジット
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

def plot_asset_growth(asset_values):
    plt.plot(asset_values)
    plt.title('Asset Growth Over Time')
    plt.xlabel('Time')
    plt.ylabel('Asset Value')
    plt.show()

def generate_report(initial_capital, final_capital):
    total_return = (final_capital - initial_capital) / initial_capital
    print(f"Total Return: {total_return * 100:.2f}%")
    # シャープレシオなど他の評価指標を追加

def main():
    start_time = time.time()

    # 1. データの取得と前処理
    print("データ取得と前処理を開始...")
    df = get_data(TICKER, START_DATE, END_DATE, INTERVAL)
    df = preprocess_data(df)

    # 特徴量生成
    print("特徴量を生成しています...")
    df = calc_features(df)

    # 特徴量選択
    print("特徴量選択を開始...")
    df = select_features_pipeline(df, target_column='long_target', corr_threshold=0.9, top_n=20)

    # 特徴量とターゲットに分割
    FEATURES = [col for col in df.columns if col not in ['long_target']]
    X = df[FEATURES].values
    y = df['long_target'].values

    # 2. データ分割
    print("データをトレーニングセットとテストセットに分割...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 3. Optunaでのハイパーパラメータ最適化
    print("モデルの最適化を開始...")
    def objective(trial):
        model_type = 'xgboost'
        return optimize_model(trial, model_type, X_train, y_train)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)  # n_trialsで試行回数を調整
    print("最適化完了！最適パラメータ：", study.best_params)

    # 4. モデルの作成
    model = create_xgboost_model()
    model.set_params(**study.best_params)

    # 5. モデルの学習
    print("モデルの学習を開始...")
    model.fit(X_train, y_train)

    # 6. 予測とバックテスト
    print("予測とバックテストを開始...")
    capital_results = run_backtest(df, model, FEATURES)

    # 7. 資産推移の可視化
    print("資産推移のグラフを描画...")
    plot_asset_growth(capital_results["equity_curve"])

    # 8. 最終レポート
    print("最終レポートを表示...")
    generate_report(10000, capital_results["equity_curve"][-1])  # 初期資産10000と最終資産値を表示

    end_time = time.time()
    print(f"✅ 全体の実行時間: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
