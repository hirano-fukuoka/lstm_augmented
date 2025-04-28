import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time  # プログレスバー用

# --- セッションステート初期化 ---
if 'model' not in st.session_state:
    st.session_state.model = None

# --- データ拡張（軽量版：50件） ---
def augment_data(df, num_augments=50, noise_std=0.5, time_scale_range=(0.95, 1.05), temp_shift_range=(-2, 2)):
    aug_dfs = []
    progress = st.progress(0)
    status = st.empty()

    for idx in range(num_augments):
        temp = df.copy()

        temp["T_internal"] += np.random.normal(0, noise_std, size=len(temp))
        scale = np.random.uniform(*time_scale_range)
        temp["time"] = temp["time"] * scale
        shift = np.random.uniform(*temp_shift_range)
        temp["T_internal"] += shift
        temp["T_surface"] += shift

        aug_dfs.append(temp)

        progress.progress((idx + 1) / num_augments)
        status.text(f"データ拡張中... {idx+1}/{num_augments}")

    status.text("✅ データ拡張完了")
    return pd.concat(aug_dfs, ignore_index=True)

# --- LSTMモデル構築 ---
def create_sequences(df, window_size=20):
    X, y = [], []
    for i in range(len(df) - window_size):
        seq_x = df["T_internal"].iloc[i:i+window_size].values
        seq_y = df["T_surface"].iloc[i+window_size]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_with_progress(model, X, y, epochs=10, batch_size=32):
    progress = st.progress(0)
    status = st.empty()
    for epoch in range(epochs):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0)
        progress.progress((epoch + 1) / epochs)
        status.text(f"LSTM学習中... エポック {epoch+1}/{epochs}")
    status.text("✅ LSTM学習完了")
    return model

# --- サイクル検出＆切り出し ---
def extract_cycles(df, start_col, lag_sec, duration_sec, sampling=0.1):
    starts = df[df[start_col].diff() == 1].index
    lag_steps = int(lag_sec / sampling)
    duration_steps = int(duration_sec / sampling)
    segments = []
    for s in starts:
        t_start = s + lag_steps
        t_end = t_start + duration_steps
        if t_end <= len(df):
            segments.append(df.iloc[t_start:t_end].copy())
    return pd.concat(segments, ignore_index=True) if segments else pd.DataFrame()

# --- Streamlitアプリ本体 ---
st.set_page_config(page_title="LSTMによるT_surface予測（軽量版＋セッション保存）", layout="wide")
st.title("🌡️ LSTM版 T_surface 多点予測アプリ（軽量版・学習スキップ対応）")

# --- サイドバー設定 ---
st.sidebar.header("⏱️ 時間設定")
lag_seconds = st.sidebar.number_input("立ち上がりラグ（秒）", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
duration_seconds = st.sidebar.number_input("予測する時間範囲（秒）", min_value=5.0, max_value=120.0, value=55.0, step=1.0)
sampling_rate = 0.1
window_size = 20

# --- 1. 学習用データアップロード ---
st.header("1️⃣ 学習用データアップロード")
train_file = st.file_uploader("T_internal, T_surface, start_signalを含むCSV", type="csv")

if train_file:
    df = pd.read_csv(train_file)
    if set(["T_internal", "T_surface", "start_signal"]).issubset(df.columns):
        base_segment = extract_cycles(df, "start_signal", lag_seconds, duration_seconds, sampling_rate)
        st.subheader("🔄 データ拡張中...")
        aug_train_df = augment_data(base_segment, num_augments=50)

        X, y = create_sequences(aug_train_df, window_size)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = build_lstm_model((window_size, 1))
        st.subheader("🔄 LSTM学習中...")
        model = train_lstm_with_progress(model, X, y, epochs=10)
        st.success("✅ モデル学習完了")

        # ✅ モデルをセッションに保存
        st.session_state.model = model
    else:
        st.error("必要な列が見つかりません。")

# --- 2. 予測データアップロード ---
st.header("2️⃣ 予測用データアップロード")
test_file = st.file_uploader("T_internal1〜5, start_signalを含むCSV", type="csv")

def prepare_predict_sequences(df, window_size=20):
    X = []
    for i in range(len(df) - window_size):
        seq_x = df.iloc[i:i+window_size].values
        X.append(seq_x)
    return np.array(X)

if st.session_state.model and test_file:
    model = st.session_state.model  # ✅ セッションからモデル取得
    df_test = pd.read_csv(test_file)
    
    if "start_signal" not in df_test.columns:
        st.error("start_signal列がありません")
    else:
        internal_cols = [col for col in df_test.columns if col.startswith("T_internal")]
        if not internal_cols:
            st.error("T_internal1〜5列が必要です")
        else:
            progress = st.progress(0)
            status = st.empty()
            all_preds = []

            for idx, col in enumerate(internal_cols):
                temp_df = df_test[["time", col, "start_signal"]].rename(columns={col: "T_internal"})
                segments = extract_cycles(temp_df, "start_signal", lag_seconds, duration_seconds, sampling_rate)
                if not segments.empty:
                    X_pred = prepare_predict_sequences(segments["T_internal"], window_size)
                    X_pred = X_pred.reshape((X_pred.shape[0], X_pred.shape[1], 1))
                    y_pred = model.predict(X_pred)
                    result_df = segments.iloc[window_size:].copy()
                    result_df[f"Predicted_T_surface_{col}"] = y_pred.flatten()
                    all_preds.append(result_df.set_index("time")[[f"Predicted_T_surface_{col}"]])

                progress.progress((idx + 1) / len(internal_cols))
                status.text(f"予測中... {idx+1}/{len(internal_cols)}個完了")

            if all_preds:
                result_df = pd.concat(all_preds, axis=1).reset_index()

                # --- 予測結果表示 ---
                st.subheader("📊 予測結果プレビュー")
                st.dataframe(result_df.head())

                # --- グラフ描画 ---
                st.subheader("📈 入力vs予測グラフ")
                fig, axes = plt.subplots(len(internal_cols), 1, figsize=(10, 2.5 * len(internal_cols)), sharex=True)
                time_vals = result_df["time"]
                for i, col in enumerate(internal_cols):
                    ax = axes[i]
                    pred_col = f"Predicted_T_surface_{col}"
                    original_trimmed = df_test[col][len(df_test) - len(time_vals):].values
                    ax.plot(time_vals, original_trimmed, label=col, color="tab:blue")
                    ax.plot(time_vals, result_df[pred_col], label=pred_col, color="tab:red", linestyle="--")
                    ax.set_ylabel("温度 [°C]")
                    ax.set_title(f"{col} vs 予測")
                    ax.legend()
                axes[-1].set_xlabel("時間 [s]")
                st.pyplot(fig)

                # --- CSV出力 ---
                st.subheader("💾 予測結果CSVダウンロード")
                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 ダウンロード", data=csv_bytes, file_name="predicted_surface_lstm_light_session.csv", mime="text/csv")

            status.text("✅ 予測完了！")
