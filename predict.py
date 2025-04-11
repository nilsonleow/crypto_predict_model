import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from binance import Client
import argparse
from datetime import datetime, timedelta

# Функция для вычисления RSI
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Функция для загрузки данных с Binance
def get_historical_data(client, symbol, interval, start_str, end_str):
    try:
        print(f"Запрашиваем данные для {symbol} с {start_str} по {end_str}...")
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        if not klines:
            print(f"Нет данных для {symbol}")
            return None
        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                             'close_time', 'quote_asset_volume', 'trades', 
                                             'taker_buy_base', 'taker_buy_quote', 'ignored'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data = data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
        print(f"Получено {len(data)} записей для {symbol}")
        return data
    except Exception as e:
        print(f"Ошибка при сборе данных для {symbol}: {e}")
        return None

# Функция для подготовки данных
def prepare_data(data, seq_length):
    try:
        data['rsi'] = calculate_rsi(data['close'], periods=14)
        data = data.dropna()
        features = data[['close', 'rsi']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(features)
        return scaled_features, scaler
    except Exception as e:
        print(f"Ошибка при подготовке данных: {e}")
        return None, None

# Создание последовательностей
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length + 1):
        X.append(data[i:i + seq_length])
    return np.array(X)

# Парсинг аргументов командной строки
parser = argparse.ArgumentParser(description='Predict cryptocurrency prices using LSTM models.')
parser.add_argument('--powerful', action='store_true', help='Use the powerful model (36 pairs, 4 LSTM layers with 150 units)')
args = parser.parse_args()

# Выбор модели
model_file = 'multi_crypto_lstm_model_powerful.keras' if args.powerful else 'multi_crypto_lstm_model.keras'
print(f"Используем модель: {model_file}")

# Загрузка модели
try:
    model = load_model(model_file)
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    exit(1)

# Инициализация клиента Binance
try:
    client = Client()
    print("Клиент Binance успешно инициализирован")
except Exception as e:
    print(f"Ошибка при инициализации клиента Binance: {e}")
    exit(1)

# Параметры
seq_length = 30
interval = Client.KLINE_INTERVAL_1DAY

# Запрос тикера и периода у пользователя
user_input = input("Введите тикер и период (например, 'BTC 3d', по умолчанию 1 день): ").strip()
ticker, *period = user_input.split()
ticker = ticker.upper()
symbol = f"{ticker}USDT"

# Парсинг периода
if period:
    period_str = period[0].lower()
    if period_str.endswith('d'):
        days = int(period_str[:-1])
    else:
        days = int(period_str)
else:
    days = 1  # По умолчанию 1 день

print(f"Предсказываем для {symbol} на {days} день(дня) вперед")

# Загрузка данных с Binance
end_date = datetime.now()
start_date = end_date - timedelta(days=seq_length + 14)  # Дополнительно 14 дней для RSI
start_str = start_date.strftime('%d %b, %Y')
end_str = end_date.strftime('%d %b, %Y')

data = get_historical_data(client, symbol, interval, start_str, end_str)
if data is None:
    print(f"Не могу получить данные для {symbol}. Возможно, эта пара не торгуется на Binance.")
    exit(1)

# Подготовка данных
scaled_features, scaler = prepare_data(data, seq_length)
if scaled_features is None:
    print("Не удалось подготовить данные для предсказания")
    exit(1)

# Подготовка последней последовательности
last_sequence = scaled_features[-seq_length:]
last_sequence = np.expand_dims(last_sequence, axis=0)

# Многократное предсказание на указанный период
predictions = []
current_sequence = last_sequence.copy()

for day in range(days):
    # Предсказание на следующий день
    next_day_pred = model.predict(current_sequence)
    next_day_pred_denorm = scaler.inverse_transform(
        np.concatenate((next_day_pred, np.zeros((len(next_day_pred), 1))), axis=1)
    )[:, 0]
    predicted_price = next_day_pred_denorm[0]
    predictions.append(predicted_price)
    
    # Обновление последовательности для следующего предсказания
    if day < days - 1:  # Если это не последнее предсказание
        # Создаем новый элемент для последовательности
        # Для RSI используем среднее значение из последних данных (упрощение)
        last_rsi = current_sequence[0, -1, 1]  # Последнее значение RSI
        new_entry = np.array([[next_day_pred[0], last_rsi]])
        new_entry_scaled = scaler.transform(new_entry)
        
        # Обновляем последовательность
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1] = new_entry_scaled[0]

# Вывод предсказаний
for i, price in enumerate(predictions, 1):
    pred_date = (end_date + timedelta(days=i)).strftime('%Y-%m-%d')
    print(f"Предсказанная цена {symbol} на {pred_date}: {price:.2f} USDT")
