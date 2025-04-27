# ========================== ライブラリ ==========================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import talib
import yfinance as yf

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as catb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === 定数定義 ===
TICKER = '9684.T'
START_DATE = '2024-04-01'
END_DATE = '2025-04-10'
INTERVAL = '1d'
FEATURES = ['close_scaled']
STOP_LOSS_MODE = 1  # 1:固定パーセント損切り 2:パラボリックSAR損切り 3:直近安値更新
STOP_LOSS = 0.02    # 固定パーセント損切りでの損失許容率（例:2%）
TAKE_PROFIT = 0.04  # 利確幅（例:4%）
COMMISSION = 0.0005  # 手数料率（必要なら設定）
SLIPPAGE = 0.0005    # スリッページ率（必要なら設定）
SELECTED_MODELS = ['xgboost', 'randomforest', 'catboost', 'lightgbm']
ENSEMBLE_TYPE = 'stacking'  # 'blending', 'stacking', 'voting_hard', 'voting_soft'

# === データ取得 ===
def fetch_data(ticker, start_date, end_date, interval):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        print("[DEBUG] ダウンロード直後のカラム:", data.columns.tolist())

        if data.empty:
            raise ValueError("データが取得できませんでした。")

        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in expected_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"[ERROR] 必要な列が存在しません: {missing_cols}")

        data = data[expected_cols].copy()
        data.columns = ['open', 'high', 'low', 'close', 'volume']

        print("[DEBUG] 小文字に変換後のカラム:", data.columns.tolist())
        return data

    except Exception as e:
        print(f"[ERROR] fetch_data 内のエラー: {e}")
        return None

# === 特徴量生成 ===
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

    # ボリンジャーバンドを計算して df に追加
    df['BBANDS_upperband'], df['BBANDS_middleband'], df['BBANDS_lowerband'] = talib.BBANDS(
        df['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0
    )

    new_features = {
        # ボリンジャーバンド
        'BBANDS_upperband': (df['BBANDS_upperband'] - hilo) / df['close'],
        'BBANDS_middleband': (df['BBANDS_middleband'] - hilo) / df['close'],
        'BBANDS_lowerband': (df['BBANDS_lowerband'] - hilo) / df['close'],

        # 移動平均
        'DEMA': (talib.DEMA(close_col, timeperiod=30) - hilo) / close_col,
        'EMA': (talib.EMA(close_col, timeperiod=30) - hilo) / close_col,
        'EMA_short': (talib.EMA(close_col, timeperiod=5) - hilo) / close_col,
        'EMA_middle': (talib.EMA(close_col, timeperiod=20) - hilo) / close_col,
        'EMA_long': (talib.EMA(close_col, timeperiod=40) - hilo) / close_col,
        'HT_TRENDLINE': (talib.HT_TRENDLINE(close_col) - hilo) / close_col,
        'KAMA': (talib.KAMA(close_col, timeperiod=30) - hilo) / close_col,
        'MA': (talib.MA(close_col, timeperiod=30, matype=0) - hilo) / close_col,
        'MIDPOINT': (talib.MIDPOINT(close_col, timeperiod=14) - hilo) / close_col,
        'SMA': (talib.SMA(close_col, timeperiod=30) - hilo) / close_col,
        'T3': (talib.T3(close_col, timeperiod=5, vfactor=0) - hilo) / close_col,
        'HMA': talib.WMA(close_col, timeperiod=30),
        'TEMA': (talib.TEMA(close_col, timeperiod=30) - hilo) / close_col,
        'TRIMA': (talib.TRIMA(close_col, timeperiod=30) - hilo) / close_col,
        'WMA': (talib.WMA(close_col, timeperiod=30) - hilo) / close_col,

        # MACD
        'MACD_macd': talib.MACD(close_col, fastperiod=12, slowperiod=26, signalperiod=9)[0] / close_col,
        'MACD_macdsignal': talib.MACD(close_col, fastperiod=12, slowperiod=26, signalperiod=9)[1] / close_col,
        'MACD_macdhist': talib.MACD(close_col, fastperiod=12, slowperiod=26, signalperiod=9)[2] / close_col,

        # 線形回帰系
        'LINEARREG': (talib.LINEARREG(close_col, timeperiod=14) - close_col) / close_col,
        'LINEARREG_SLOPE': talib.LINEARREG_SLOPE(close_col, timeperiod=14) / close_col,
        'LINEARREG_ANGLE': talib.LINEARREG_ANGLE(close_col, timeperiod=14),
        'LINEARREG_INTERCEPT': (talib.LINEARREG_INTERCEPT(close_col, timeperiod=14) - close_col) / close_col,

        # AD系
        'AD': talib.AD(high_col, low_col, close_col, volume) / close_col,
        'ADX': talib.ADX(high_col, low_col, close_col, timeperiod=14),
        'ADXR': talib.ADXR(high_col, low_col, close_col, timeperiod=14),
        'ADOSC': talib.ADOSC(high_col, low_col, close_col, volume, fastperiod=3, slowperiod=10) / close_col,
        'OBV': talib.OBV(close_col, volume) / close_col,

        # オシレーター系
        'APO': talib.APO(close_col, fastperiod=12, slowperiod=26, matype=0) / close_col,
        'BOP': talib.BOP(open_col, high_col, low_col, close_col),
        'CCI': talib.CCI(high_col, low_col, close_col, timeperiod=14),
        'DX': talib.DX(high_col, low_col, close_col, timeperiod=14),
        'MFI': talib.MFI(high_col, low_col, close_col, volume, timeperiod=14),
        'MINUS_DI': talib.MINUS_DI(high_col, low_col, close_col, timeperiod=14),
        'PLUS_DI': talib.PLUS_DI(high_col, low_col, close_col, timeperiod=14),
        'MOM': talib.MOM(close_col, timeperiod=10) / close_col,
        'RSI': talib.RSI(close_col, timeperiod=14),
        'TRIX': talib.TRIX(close_col, timeperiod=30),
        'ULTOSC': talib.ULTOSC(high_col, low_col, close_col, timeperiod1=7, timeperiod2=14, timeperiod3=28),
        'WILLR': talib.WILLR(high_col, low_col, close_col, timeperiod=14),
        'SAR': talib.SAR(high_col, low_col, acceleration=0.02, maximum=0.2),

        # ボラティリティ系
        'MINUS_DM': talib.MINUS_DM(high_col, low_col, timeperiod=14) / close_col,
        'PLUS_DM': talib.PLUS_DM(high_col, low_col, timeperiod=14) / close_col,
        'STDDEV': talib.STDDEV(close_col, timeperiod=5, nbdev=1),
        'TRANGE': talib.TRANGE(high_col, low_col, close_col),
        'VAR': talib.VAR(close_col, timeperiod=5, nbdev=1),
        'ATR': talib.ATR(high_col, low_col, close_col, timeperiod=14),
        'NATR': talib.NATR(high_col, low_col, close_col, timeperiod=14),
        'VOLATILITY_index': talib.ATR(high_col, low_col, close_col, timeperiod=14) / talib.STDDEV(close_col, timeperiod=5, nbdev=1),

        # ストキャスティクス
        'STOCH_slowk': talib.STOCH(high_col, low_col, close_col, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0],
        'STOCH_slowd': talib.STOCH(high_col, low_col, close_col, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[1],
        'STOCHF_fastk': talib.STOCHF(high_col, low_col, close_col, fastk_period=5, fastd_period=3, fastd_matype=0)[0],
        'STOCHF_fastd': talib.STOCHF(high_col, low_col, close_col, fastk_period=5, fastd_period=3, fastd_matype=0)[1],
        'STOCHRSI_fastk': talib.STOCHRSI(close_col, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[0],
        'STOCHRSI_fastd': talib.STOCHRSI(close_col, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[1],

        # ヒルベルト変換
        'HT_DCPERIOD': talib.HT_DCPERIOD(close_col),
        'HT_DCPHASE': talib.HT_DCPHASE(close_col),
        'HT_PHASOR_inphase': talib.HT_PHASOR(close_col)[0] / close_col,
        'HT_PHASOR_quadrature': talib.HT_PHASOR(close_col)[1] / close_col,
        'HT_SINE_sine': talib.HT_SINE(close_col)[0] / close_col,
        'HT_SINE_leadsine': talib.HT_SINE(close_col)[1] / close_col,
        'HT_TRENDMODE': talib.HT_TRENDMODE(close_col),

        # その他
        'ROC': talib.ROC(close_col, timeperiod=10) / close_col,
        'STDDEV': talib.STDDEV(close_col, timeperiod=5, nbdev=1) / close_col,
        'TRANGE': talib.TRANGE(high_col, low_col, close_col) / close_col,
        'AROON_aroondown': talib.AROON(high_col, low_col, timeperiod=14)[0],
        'AROON_aroonup': talib.AROON(high_col, low_col, timeperiod=14)[1],
        'AROONOSC': talib.AROONOSC(high_col, low_col, timeperiod=14),
        'BETA': talib.BETA(high_col, low_col, timeperiod=5),
        'CORREL': talib.CORREL(high_col, low_col, timeperiod=30),
        'Price_ratio': df['close'] / df['close'].shift(1),
        'HIGH_ratio': df['high'] / df['high'].shift(1),
        'LOW_ratio': df['low'] / df['low'].shift(1),

        # Lag特徴量
        'CLOSE_lag_1': df['close'].shift(1),
        'CLOSE_lag_5': df['close'].shift(5),
        'MOVIENG_avg_5': df['close'].rolling(window=5).mean(),

        # 周期性の特徴量
        'DAY_of_week': df.index.dayofweek,
        'IS_weekend': (df.index.dayofweek >= 5).astype(int),
        'MONTH': df.index.month,
        'SIN_day': np.sin(2 * np.pi * df.index.dayofweek / 7),
        'COS_day': np.cos(2 * np.pi * df.index.dayofweek / 7),
        'SIN_month': np.sin(2 * np.pi * df.index.month / 12),
        'COS_month': np.cos(2 * np.pi * df.index.month / 12),

        # リターン系
        'log_return': np.log(df['close'] / df['close'].shift(1)),
        'return_1d': df['close'].pct_change(1),
        'return_5d': df['close'].pct_change(5),
        'return_10d': df['close'].pct_change(10),
        'return_20d': df['close'].pct_change(20),
        'return_ma_5': df['close'].pct_change(1).rolling(window=5).mean(),
        'return_ma_10': df['close'].pct_change(1).rolling(window=10).mean(),
        'volatility_5': df['close'].rolling(window=5).std(),
        'volatility_10': df['close'].rolling(window=10).std(),
        'range': df['high'] - df['low'],
        'volume_change': df['volume'].pct_change(1),

        # トレンド・モメンタム系
        'ma_5': df['close'].rolling(window=5).mean(),
        'ma_25': df['close'].rolling(window=25).mean(),
        'ma_75': df['close'].rolling(window=75).mean(),
        'ma_deviation_5': df['close'] / df['close'].rolling(window=5).mean() - 1,
        'ma_deviation_25': df['close'] / df['close'].rolling(window=25).mean() - 1,
        'ma_deviation_75': df['close'] / df['close'].rolling(window=75).mean() - 1,
        'momentum_5': df['close'] - df['close'].shift(5),
        'momentum_10': df['close'] - df['close'].shift(10),
        'momentum_20': df['close'] - df['close'].shift(20),
    }

    # ローソク足パターンを辞書に追加
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

    # 一括で DataFrame に追加
    df = pd.concat([df, pd.DataFrame(new_features)], axis=1)

    # 例: 翌日の終値が当日の終値より高ければ1、そうでなければ0
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    df.dropna(inplace=True)
    return df

# ========================== 相関除去関数 ==========================
def remove_highly_correlated_features(df, threshold=0.9, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = ['open', 'high', 'low', 'close', 'volume']
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold) and column not in exclude_columns]
    print(f"🛠️ 削除された高相関特徴量: {to_drop}")
    return df.drop(columns=to_drop)

# ========================== PyTorch用ラッパー ==========================
class BaseTorchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, input_dim, hidden_dim=64, num_layers=2, epochs=20, batch_size=32, lr=0.001, device=None):
        self.model_class = model_class
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.classes_ = None  # クラス属性を初期化

    def fit(self, X, y):
        # クラス属性を設定
        self.classes_ = np.unique(y)

        X = torch.tensor(X.values, dtype=torch.float32) if isinstance(X, pd.DataFrame) else torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1) if isinstance(y, pd.Series) else torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = self.model_class(self.input_dim, self.hidden_dim, self.num_layers).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        X = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            probs = torch.sigmoid(outputs).cpu().numpy()
        return np.hstack([1 - probs, probs])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

# ========================== モデル定義 (RNN/Transformer) ==========================
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SimpleTransformer, self).__init__()

        # nhead を input_dim に合わせて調整
        nhead = 4
        if input_dim % nhead != 0:
            nhead = 1  # 必要に応じて nhead を 1 に設定

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.transformer(x)
        out = self.fc(out[:, -1, :])
        return out

# ========================== Baseモデル作成 ==========================
def get_base_models(FEATURES):
    models = {
        'lgb': lgb.LGBMClassifier(random_state=42),
        'xgb': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'cat': catb.CatBoostClassifier(verbose=0, random_state=42),
        'rf': RandomForestClassifier(random_state=42),
        'rnn': BaseTorchWrapper(SimpleRNN, input_dim=len(FEATURES), epochs=30),
        'transformer': BaseTorchWrapper(SimpleTransformer, input_dim=len(FEATURES), epochs=30)
    }
    return models

# ========================== Stackingモデル作成 ==========================
def build_stacking(models):
    estimators = [(name, model) for name, model in models.items()]
    stack_model = StackingClassifier(estimators=estimators, final_estimator=lgb.LGBMClassifier())
    return stack_model

# ========================== バックテスト（超シンプル版） ==========================
def run_backtest(df, model, FEATURES):
    initial_cash = 10000
    cash = initial_cash
    position = 0

    # 修正: df[FEATURES] をそのまま渡す
    preds = model.predict(df[FEATURES])
    df['preds'] = preds

    for i in range(1, len(df)):
        close_price = df['close'].iloc[i]

        # Buy Signal
        if df['preds'].iloc[i-1] == 1 and cash > 0:
            position = cash / close_price
            cash = 0
            print(f"🔼 Buy at {close_price:.2f}, Position: {position:.4f}")

        # Sell Signal
        elif df['preds'].iloc[i-1] == 0 and position > 0:
            cash = position * close_price
            position = 0
            print(f"🔽 Sell at {close_price:.2f}, Cash: {cash:.2f}")

    # 最終資産を計算
    final_value = cash + position * df['close'].iloc[-1]
    total_return = (final_value / initial_cash) - 1
    print(f"📈 最終資産: {final_value:.2f}円（リターン: {total_return*100:.2f}%）")
    return final_value

# ========================== main ==========================
def main():
    # --- データ取得 ---
    df = fetch_data(TICKER, START_DATE, END_DATE, INTERVAL)
    if df is None:
        print("データが取得できませんでした。処理を終了します。")
        return

    # --- 特徴量生成 ---
    df = calc_features(df)

    # --- 相関除去 ---
    print(f"🔍 相関除去前の特徴量数: {df.shape[1]}列")
    df = remove_highly_correlated_features(df)
    print(f"✅ 相関除去後の特徴量数: {df.shape[1]}列")

    # --- 特徴量定義 ---
    FEATURES = [col for col in df.columns if col not in ['target']]

    # --- 学習・評価 ---
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df['target'], test_size=0.2, random_state=42)

    base_models = get_base_models(FEATURES)
    ensemble_model = build_stacking(base_models)

    ensemble_model.fit(X_train, y_train)
    preds = ensemble_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"🎯 テスト精度: {acc:.4f}")

    # --- バックテスト ---
    run_backtest(df, ensemble_model, FEATURES)

if __name__ == "__main__":
    main()
