import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_sequences(data, look_back):
    X, Y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        Y.append(data[i, 0])
    return np.array(X), np.array(Y)

def main():
    try:
        logging.basicConfig(filename='lstm.log', level=logging.INFO,
                            format='%(asctime)s:%(levelname)s:%(message)s')
        ticker = 'VIVT3.SA'
        data_atual = datetime.now().strftime('%Y-%m-%d')
        print(f"Baixando dados históricos para {ticker} de 2023-01-01 até {data_atual}...")
        dados = yf.download(ticker, start='2023-01-01', end=data_atual)
        print(f"Dados baixados com sucesso. Total de registros: {len(dados)}")
        print("Colunas disponíveis:", dados.columns.tolist())
        print("\nPrimeiras linhas dos dados:")
        print(dados.head())
        logging.info('Dados históricos baixados com sucesso.')
        if dados.empty:
            print("Erro: Nenhum dado foi baixado para o período solicitado.")
            logging.error('Nenhum dado foi baixado para o período solicitado.')
            return
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.get_level_values(0)
            print("Colunas normalizadas:", dados.columns.tolist())
        if 'Close' not in dados.columns:
            print("Erro: Coluna 'Close' não encontrada nos dados baixados.")
            logging.error("Coluna 'Close' não encontrada nos dados baixados.")
            return
        data_base = dados.index[-1]
        print(f'Usando último dia disponível como data base: {data_base.date()}')
        preco_base = dados['Close'].loc[data_base]
        print(f'Preço base ({data_base.date()}): R$ {preco_base:.2f}')
        if len(dados) < 70:
            print("Erro: Não há dados suficientes para treinar o modelo.")
            logging.error("Não há dados suficientes para treinar o modelo.")
            return
        close_prices = dados['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)
        split = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:split], scaled_data[split:]
        look_back = 60
        X_train, Y_train = create_sequences(train_data, look_back)
        X_test, Y_test = create_sequences(test_data, look_back)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        logging.info('Dados preparados corretamente para treino e teste.')
        modelo_lstm = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        modelo_lstm.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        modelo_lstm.fit(X_train, Y_train, epochs=50, batch_size=32, 
                        validation_data=(X_test, Y_test), callbacks=[early_stopping], verbose=1)
        logging.info('Treinamento do modelo concluído.')
        pred_test = modelo_lstm.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, pred_test))
        print(f'RMSE do Modelo: {rmse:.4f}')
        ult_data = dados.index[-1]
        target_date = ult_data + pd.DateOffset(months=12)
        datas_futuras = pd.date_range(start=ult_data, end=target_date, freq='B')[1:]
        forecast_steps = len(datas_futuras)
        print(f"\nProjetando preços de {ult_data.date()} até {target_date.date()} ({forecast_steps} dias úteis)")
        previsoes = []
        entrada = scaled_data[-look_back:].tolist()
        for _ in range(forecast_steps):
            seq = np.array(entrada[-look_back:]).reshape((1, look_back, 1))
            predicao = modelo_lstm.predict(seq, verbose=0)[0,0]
            previsoes.append(predicao)
            entrada.append([predicao])
        previsoes = scaler.inverse_transform(np.array(previsoes).reshape(-1, 1))
        plt.figure(figsize=(12,6))
        plt.plot(dados.index, dados['Close'], label='Preço Histórico')
        plt.plot(datas_futuras, previsoes, label='Previsão 12 meses', color='red', linestyle='--')
        plt.axvline(x=dados.index[-1], color='green', linestyle=':', label='Data Atual')
        plt.title(f'Previsão do Preço da Ação VIVA3 - Projeção de 12 meses')
        plt.xlabel('Data')
        plt.ylabel('Preço (R$)')
        plt.legend()
        plt.grid(True)
        plt.savefig('previsao_lstm.png')
        plt.show()
        preco_projetado = previsoes[-1][0]
        print(f'Preço Projetado em {target_date.date()}: R$ {preco_projetado:.2f}')
        variacao = ((preco_projetado - preco_base) / preco_base) * 100
        print(f'Variação percentual esperada de {data_base.date()} até {target_date.date()}: {variacao:.2f}%')
        logging.info(f'Preço base ({data_base.date()}): R$ {preco_base:.2f}')
        logging.info(f'Preço projetado em {target_date.date()}: R$ {preco_projetado:.2f}')
        logging.info(f'Variação percentual esperada de {data_base.date()} até {target_date.date()}: {variacao:.2f}%')
    except Exception as e:
        logging.error(f'Erro no script: {e}')
        print(f'Ocorreu um erro: {e}')

if __name__ == "__main__":
    main()