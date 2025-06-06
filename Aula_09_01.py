import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

print('\n---- OBTENDO DADOS ----')

endereco_dados = 'https://www.ispdados.rj.gov.br/Arquivos/BaseDPEvolucaoMensalCisp.csv'

# Criando o DataFrame Ocorrencias
df_ocorrencias = pd.read_csv(endereco_dados,sep=';',encoding='iso-8859-1')

# Criando o DataFrame Roubo de Veiculo
df_roubo_veiculo = df_ocorrencias[['cisp','roubo_veiculo']]
df_roubo_veiculo = df_roubo_veiculo.groupby(['cisp']).sum(['roubo_veiculo']).reset_index()

# Criando o DataFrame Furto de Veiculo
df_furto_veiculo = df_ocorrencias[['cisp','furto_veiculos']]
df_furto_veiculo = df_furto_veiculo.groupby(['cisp']).sum(['furto_veiculos']).reset_index()

# Criando o DataFrame Recuperação de Veiculo
df_recuperacao_veiculo = df_ocorrencias[['cisp','recuperacao_veiculos']]
df_recuperacao_veiculo = df_recuperacao_veiculo.groupby(['cisp']).sum(['recuperacao_veiculos']).reset_index()

# Criando o DataFrame Roubo e Furto de Veiculo
df_roubo_furto_veiculo = df_ocorrencias[['cisp','roubo_veiculo','furto_veiculos']]
df_roubo_furto_veiculo = df_roubo_furto_veiculo.groupby(['cisp']).sum(['roubo_veiculo','furto_veiculos']).reset_index()

# Criando o DataFrame Roubo e Recuperação de Veiculo
df_roubo_recuperacao_veiculo = df_ocorrencias[['cisp','roubo_veiculo','recuperacao_veiculos']]
df_roubo_recuperacao_veiculo = df_roubo_recuperacao_veiculo.groupby(['cisp']).sum(['roubo_veiculo','recuperacao_veiculos']).reset_index()

# Criando o DataFrame Furto e Recuperação de Veiculo
df_furto_recuperacao_veiculo = df_ocorrencias[['cisp','furto_veiculos','recuperacao_veiculos']]
df_furto_recuperacao_veiculo = df_furto_recuperacao_veiculo.groupby(['cisp']).sum(['furto_veiculos','recuperacao_veiculos']).reset_index()

# Criando o Array dos Roubos de Veiculos
array_roubo_veiculo = np.array(df_roubo_veiculo["roubo_veiculo"])

# Criando o Array dos Furto de Veiculos
array_furto_veiculo = np.array(df_furto_veiculo["furto_veiculos"])

# Criando o Array das Recuperação de Veiculos
array_recuperacao_veiculo = np.array(df_recuperacao_veiculo["recuperacao_veiculos"])

# IQR (Intervalo interquartil)
    # q3 - q1
    # É a amplitude do intervalo dos 50% dos dados centrais
    # Ela ignora os valores extremos. Max e Min que estão fora do IQR
    # Não sofre a interferência dos valores extremos
    # quanto mais próximo de zero, mais homogêneo são os dados
    # quanto mais próximo do q3, mais heterogêneo são os dados

# Obtendo os Quartis dos Roubos de Veiculos - Método weibull
q1_roubo_veiculo = np.quantile(array_roubo_veiculo, 0.25, method='weibull')
q2_roubo_veiculo = np.quantile(array_roubo_veiculo, 0.50, method='weibull')
q3_roubo_veiculo = np.quantile(array_roubo_veiculo, 0.75, method='weibull')
iqr_roubo_veiculo = q3_roubo_veiculo - q1_roubo_veiculo

# Obtendo os Quartis dos Furtos de Veiculos - Método weibull
q1_furto_veiculo = np.quantile(array_furto_veiculo, 0.25, method='weibull')
q2_furto_veiculo = np.quantile(array_furto_veiculo, 0.50, method='weibull')
q3_furto_veiculo = np.quantile(array_furto_veiculo, 0.75, method='weibull')
iqr_furto_veiculo = q3_furto_veiculo - q1_furto_veiculo

# Obtendo os Quartis das Recuperações de Veiculos - Método weibull
q1_recuperacao_veiculo = np.quantile(array_recuperacao_veiculo, 0.25, method='weibull')
q2_recuperacao_veiculo = np.quantile(array_recuperacao_veiculo, 0.50, method='weibull')
q3_recuperacao_veiculo = np.quantile(array_recuperacao_veiculo, 0.75, method='weibull')
iqr_recuperacao_veiculo = q3_recuperacao_veiculo - q1_recuperacao_veiculo

# Identificando os outliers superiores e inferiores dos Roubos de Veiculos
limite_superior_roubo_veiculo = q3_roubo_veiculo + (1.5 * iqr_roubo_veiculo)
limite_inferior_roubo_veiculo = q1_roubo_veiculo - (1.5 * iqr_roubo_veiculo)

# Identificando os outliers superiores e inferiores dos Furtos de Veiculos
limite_superior_furto_veiculo = q3_furto_veiculo + (1.5 * iqr_furto_veiculo)
limite_inferior_furto_veiculo = q1_furto_veiculo - (1.5 * iqr_furto_veiculo)

# Identificando os outliers superiores e inferiores das Recuperação de Veiculos
limite_superior_recuperacao_veiculo = q3_recuperacao_veiculo + (1.5 * iqr_recuperacao_veiculo)
limite_inferior_recuperacao_veiculo = q1_recuperacao_veiculo - (1.5 * iqr_recuperacao_veiculo)

# Filtrando o DataFrame Roubos de Veiculos
df_roubo_veiculo_outliers_superiores = df_roubo_veiculo[df_roubo_veiculo['roubo_veiculo'] > limite_superior_roubo_veiculo]
df_roubo_veiculo_outliers_inferiores = df_roubo_veiculo[df_roubo_veiculo['roubo_veiculo'] < limite_inferior_roubo_veiculo]

# Filtrando o DataFrame Furtos de Veiculos
df_furto_veiculo_outliers_superiores = df_furto_veiculo[df_furto_veiculo['furto_veiculos'] > limite_superior_furto_veiculo]
df_furto_veiculo_outliers_inferiores = df_furto_veiculo[df_furto_veiculo['furto_veiculos'] < limite_inferior_furto_veiculo]

# Filtrando o DataFrame Recuperação de Veiculos
df_recuperacao_veiculo_outliers_superiores = df_recuperacao_veiculo[df_recuperacao_veiculo['recuperacao_veiculos'] > limite_superior_recuperacao_veiculo]
df_recuperacao_veiculo_outliers_inferiores = df_recuperacao_veiculo[df_recuperacao_veiculo['recuperacao_veiculos'] < limite_inferior_recuperacao_veiculo]

# Verificando a existência de Outliers Inferiores para Roubo de Veiculos
print('\n- Verificando a existência de outliers inferiores - Roubo de Veículos')
if len(df_roubo_veiculo_outliers_inferiores) == 0:
    print("Não existem outliers inferiores")
else:
    print(df_roubo_veiculo_outliers_inferiores)

# Verificando a existência de Outliers Inferiores para Furtos de Veiculos
print('\n- Verificando a existência de outliers inferiores - Furtos de Veículos')
if len(df_furto_veiculo_outliers_inferiores) == 0:
    print("Não existem outliers inferiores")
else:
    print(df_furto_veiculo_outliers_inferiores)

# Verificando a existência de Outliers Inferiores para Recuperação de Veiculos
print('\n- Verificando a existência de outliers inferiores - Recuperação de Veículos')
if len(df_recuperacao_veiculo_outliers_inferiores) == 0:
    print("Não existem outliers inferiores")
else:
    print(df_recuperacao_veiculo_outliers_inferiores)   

# Verificando a existência de Outliers Superiores para Roubo de Veiculos
print('\n- Verificando a existência de outliers superiores - Roubo de Veículos')
if len(df_roubo_veiculo_outliers_superiores) == 0:
    print("Não existem outliers superiores")
else:
    print(df_roubo_veiculo_outliers_superiores)

# Verificando a existência de Outliers Superiores para Furtos de Veiculos
print('\n- Verificando a existência de outliers superiores - Furtos de Veículos')
if len(df_furto_veiculo_outliers_superiores) == 0:
    print("Não existem outliers superiores")
else:
    print(df_furto_veiculo_outliers_superiores)

# Verificando a existência de Outliers Superiores para Recuperação de Veiculos
print('\n- Verificando a existência de outliers superiores - Recuperação de Veículos')
if len(df_recuperacao_veiculo_outliers_superiores) == 0:
    print("Não existem outliers superiores")
else:
    print(df_recuperacao_veiculo_outliers_superiores)

# Correlação entre Roubo de Veiculo e Furto de Veiculo
# 0.9 a 1.0 (positiva ou negativa) = Muito forte correlação
# 0.7 a 0.9 (positiva ou negativa) = Forte correlação
# 0.5 a 0.7 (positiva ou negativa) = Moderada correlação
# 0.3 a 0.5 (positiva ou negativa) = Fraca correlação
# 0.0 a 0.3 (positiva ou negativa) = Sem correlação

corr_roubo_furto_veiculo = np.corrcoef(df_roubo_furto_veiculo['roubo_veiculo'], df_roubo_furto_veiculo['furto_veiculos'])[0,1]
corr_roubo_recuperacao_veiculo = np.corrcoef(df_roubo_recuperacao_veiculo['roubo_veiculo'], df_roubo_recuperacao_veiculo['recuperacao_veiculos'])[0,1]
corr_furto_recuperacao_veiculo = np.corrcoef(df_furto_recuperacao_veiculo['furto_veiculos'], df_furto_recuperacao_veiculo['recuperacao_veiculos'])[0,1]

print(f"Correlação entre Roubo e Furto de Veiculo: {corr_roubo_furto_veiculo:.3f}")
print(f"Correlação entre Roubo e Recuperação de Veiculo: {corr_roubo_recuperacao_veiculo:.3f}")
print(f"Correlação entre Furto e Recuperação de Veiculo: {corr_furto_recuperacao_veiculo:.3f}")

# Regressão Linear Simples
# Coeficiente angular: 
# Também chamado de inclinação da reta ou coeficiente de regressão, esse valor indica o quanto a variável dependente (por exemplo, vendas, temperatura, lucro etc.) aumenta ou diminui a cada unidade de aumento da variável independente (por exemplo, tempo, investimento, idade etc.).
# Intercepto:
# É o ponto onde a reta de regressão cruza o eixo Y. Representa o valor estimado da variável dependente quando a variável independente é zero.
# R²:
# Representa a proporção de variabilidade da variável dependente que pode ser explicada pela variável independente.
# É o coeficiente de determinação, que mede o quão bem o modelo explica a variação dos dados.
# Varia de 0 a 1:
# 0: o modelo não explica nada.
# 1: o modelo explica perfeitamente.

x_roubo_veiculo = df_roubo_recuperacao_veiculo[['roubo_veiculo']]
y_recuperacao_veiculo = df_roubo_recuperacao_veiculo[['recuperacao_veiculos']]
modelo_simples_roubo_recuperacao = LinearRegression()
modelo_simples_roubo_recuperacao.fit(x_roubo_veiculo, y_recuperacao_veiculo)
print("\nRegressão Linear Simples - Roubo de Veículos e Recuperação de Veículos")
print(f"Coeficiente angular: {modelo_simples_roubo_recuperacao.coef_[0][0]:.2f}")
print(f"Intercepto: {modelo_simples_roubo_recuperacao.intercept_[0]:.2f}")
print(f"R²: {modelo_simples_roubo_recuperacao.score(x_roubo_veiculo, y_recuperacao_veiculo):.3f}")

x_roubo_veiculo = df_roubo_furto_veiculo[['roubo_veiculo']]
y_furto_veiculo = df_roubo_furto_veiculo[['furto_veiculos']]
modelo_simples_roubo_furto = LinearRegression()
modelo_simples_roubo_furto.fit(x_roubo_veiculo, y_furto_veiculo)
print("\nRegressão Linear Simples - Roubo de Veículos e Furto de Veículos")
print(f"Coeficiente angular: {modelo_simples_roubo_furto.coef_[0][0]:.2f}")
print(f"Intercepto: {modelo_simples_roubo_furto.intercept_[0]:.2f}")
print(f"R²: {modelo_simples_roubo_furto.score(x_roubo_veiculo, y_furto_veiculo):.3f}")

x_furto_veiculo = df_furto_recuperacao_veiculo[['furto_veiculos']]
y_recuperacao_veiculo = df_furto_recuperacao_veiculo[['recuperacao_veiculos']]
modelo_simples_furto_recuperacao = LinearRegression()
modelo_simples_furto_recuperacao.fit(x_furto_veiculo, y_recuperacao_veiculo)
print("\nRegressão Linear Simples - Furto de Veículos e Recuperação de Veículos")
print(f"Coeficiente angular: {modelo_simples_furto_recuperacao.coef_[0][0]:.2f}")
print(f"Intercepto: {modelo_simples_furto_recuperacao.intercept_[0]:.2f}")
print(f"R²: {modelo_simples_furto_recuperacao.score(x_furto_veiculo, y_recuperacao_veiculo):.3f}")

# Visualizando os Dados Analisados
print('\n- Visualizando os Dados Analisados -')
plt.subplots(3,3,figsize=(16,10))
plt.suptitle('Análise dos Dados - Roubos, Furtos e Recuperações de Veículos',fontsize=16)

# Posição 01: Gráfico dos Outliers dos Roubos de Veículos
df_roubo_veiculo_outliers_superiores_order = df_roubo_veiculo_outliers_superiores.sort_values(by='roubo_veiculo',ascending=True)
plt.subplot(3,3,1)
plt.title('Gráfico dos Outliers dos Roubos de Veículos')
plt.barh(df_roubo_veiculo_outliers_superiores_order['cisp'].astype(str),df_roubo_veiculo_outliers_superiores_order['roubo_veiculo'])

# Posição 02: Gráfico dos Outliers dos Furtos de Veículos
df_furto_veiculo_outliers_superiores_order = df_furto_veiculo_outliers_superiores.sort_values(by='furto_veiculos',ascending=True)
plt.subplot(3,3,2)
plt.title('Gráfico dos Outliers dos Furtos de Veículos')
plt.barh(df_furto_veiculo_outliers_superiores_order['cisp'].astype(str),df_furto_veiculo_outliers_superiores_order['furto_veiculos'])

# Posição 03: Gráfico dos Outliers das Recuperações de Veículos
df_recuperacao_veiculo_outliers_superiores = df_recuperacao_veiculo_outliers_superiores.sort_values(by='recuperacao_veiculos',ascending=True)
plt.subplot(3,3,3)
plt.title('Gráfico dos Outliers das Recuperações de Veículos')
plt.barh(df_recuperacao_veiculo_outliers_superiores['cisp'].astype(str),df_recuperacao_veiculo_outliers_superiores['recuperacao_veiculos'])

# Posição 04: Gráfico de Correlação dos Roubos e Recuperações de Veículos
plt.subplot(3,3,4)
plt.title('Correlação dos Roubos e Recuperações de Veículos')
plt.scatter(df_roubo_recuperacao_veiculo['roubo_veiculo'],df_roubo_recuperacao_veiculo['recuperacao_veiculos'])
plt.xlabel('Roubos de Veículos')
plt.ylabel('Recuperação de Veículos')

# Posição 05: Gráfico de Correlação dos Furtos e Recuperações de Veículos
plt.subplot(3,3,5)
plt.title('Correlação dos Furtos e Recuperações de Veículos')
plt.scatter(df_furto_recuperacao_veiculo['furto_veiculos'],df_furto_recuperacao_veiculo['recuperacao_veiculos'])
plt.xlabel('Furtos de Veículos')
plt.ylabel('Recuperação de Veículos')   

# Posição 06: Gráfico de Correlação dos Roubos e Furtos de Veículos
plt.subplot(3,3,6)
plt.title('Correlação dos Roubos e Furtos de Veículos')
plt.scatter(df_roubo_furto_veiculo['roubo_veiculo'],df_roubo_furto_veiculo['furto_veiculos'])
plt.xlabel('Roubos de Veículos')
plt.ylabel('Furtos de Veículos')

# Posição 07: Gráfico de Regressão Linear dos Roubos e Recuperações de Veículos
plt.subplot(3,3,7)
plt.title('Regressão Linear: Roubos e Recuperações de Veículos')
sns.regplot(x="roubo_veiculo", y="recuperacao_veiculos",data=df_roubo_recuperacao_veiculo,fit_reg=True)

# Posição 08: Gráfico de Regressão Linear dos Furtos e Recuperações de Veículos
plt.subplot(3,3,8)
plt.title('Regressão Linear: Furtos e Recuperações de Veículos')
sns.regplot(x="furto_veiculos", y="recuperacao_veiculos",data=df_furto_recuperacao_veiculo,fit_reg=True)

# Posição 09: Gráfico de Regressão Linear dos Roubos e Furtos de Veículos
plt.subplot(3,3,9)
plt.title('Regressão Linear: Roubos e Furtos de Veículos')
sns.regplot(x="roubo_veiculo", y="furto_veiculos",data=df_roubo_furto_veiculo,fit_reg=True)

plt.savefig("relatório_roubos_furtos_recuperação_veículos.png")
plt.show()