# LSTM Previsão de Preços - VIVT3.SA

Este projeto utiliza uma rede neural LSTM para prever o preço da ação VIVT3.SA (Vivo) usando dados históricos da B3.

## Como usar

1. **Pré-requisitos:**
   - Python 3.8 ou superior (recomendado Python 3.10)
   - Instale as dependências:
     ```sh
     pip install numpy pandas yfinance matplotlib tensorflow scikit-learn
     ```

2. **Execução:**
   - Execute o script principal:
     ```sh
     python lstm65.py
     ```
   - O script irá:
     - Baixar os dados históricos da ação VIVT3.SA desde 2023
     - Treinar um modelo LSTM
     - Fazer previsões para os próximos 12 meses
     - Gerar e salvar o gráfico `previsao_lstm.png`
     - Exibir métricas e variação percentual esperada

3. **Saídas:**
   - Gráfico salvo: `previsao_lstm.png`
   - Log de execução: `lstm.log`
   - Resultados impressos no terminal

## Sobre o gráfico

- **Linha azul:** Preço histórico da ação
- **Linha vermelha tracejada:** Previsão do preço para os próximos 12 meses
- **Linha verde pontilhada:** Data base da última cotação



## Orientações

- O script exige pelo menos 70 registros diários para treinar o modelo.
- Se não houver dados suficientes, o script irá avisar e encerrar.
- O modelo é apenas uma simulação e não deve ser usado para decisões financeiras reais.
- Para usar com outros ativos, altere o valor da variável `ticker` no início do script.

## Autor
- Adaptado por arthur1535
- Baseado em exemplos de LSTM para séries temporais financeiras
