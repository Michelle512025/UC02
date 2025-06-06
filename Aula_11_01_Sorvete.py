# Objetivo:
# Analisar como a temperatura média diária influencia a produção de sorvete em uma sorveteria localizada no Rio de Janeiro, durante o mês de janeiro de 2024.
# Hipóteses a serem testadas: 
# Hipótese 1 (H1): Existe uma correlação positiva entre a temperatura média e a produção de sorvete.
# Hipótese 2 (H2): A temperatura média é um bom preditor da produção de sorvete (modelo de regressão linear simples).

# Importando a Biblioteca
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

print('\n---- OBTENDO DADOS ----')

endereco_dados = 'BASES/base_dados_sorvete_clima.csv'

df_sorvete = pd.read_csv(endereco_dados,sep=',',encoding='utf-8')
df_sorvete = df_sorvete[['Data','Producao_Sorvete','Temperatura_Media']]   # Data,Producao_Sorvete,Temperatura_Media,Umidade

print('\n---- EXIBINDO A BASE DE DADOS ----')
print(df_sorvete)

array_producao_sorvete = np.array(df_sorvete['Producao_Sorvete'])
array_temperatura_media = np.array(df_sorvete['Temperatura_Media'])

# Obtendo as métricas solicitadas
media_producao_sorvete = np.mean(array_producao_sorvete)
max_producao_sorvete = np.max(array_producao_sorvete)
min_producao_sorvete = np.min(array_producao_sorvete)
nome_maiorsorvete = df_sorvete[df_sorvete['Producao_Sorvete'] == max_producao_sorvete]['Data']
amplitude_producao_sorvete = max_producao_sorvete - min_producao_sorvete
mediana_producao_sorvete = np.median(array_producao_sorvete)
distancia_producao_sorvete = abs((media_producao_sorvete - mediana_producao_sorvete)/mediana_producao_sorvete)*100

media_temperatura_media = np.mean(array_temperatura_media)
max_temperatura_media = np.max(array_temperatura_media)
min_temperatura_media = np.min(array_temperatura_media)
nome_maiortemperatura_media = df_sorvete[df_sorvete['Temperatura_Media'] == max_temperatura_media]['Data']
amplitude_temperatura_media = max_temperatura_media - min_temperatura_media
mediana_temperatura_media = np.median(array_temperatura_media)
distancia_temperatura_media = abs((media_temperatura_media - mediana_temperatura_media)/mediana_temperatura_media)*100

# Correlação entre Temperatura Média e Produção de Sorvete:
# 0.9 a 1.0 (positiva ou negativa) = Muito forte correlação
# 0.7 a 0.9 (positiva ou negativa) = Forte correlação
# 0.5 a 0.7 (positiva ou negativa) = Moderada correlação
# 0.3 a 0.5 (positiva ou negativa) = Fraca correlação
# 0.0 a 0.3 (positiva ou negativa) = Sem correlação

corr_tempmedia_prodsorvete = np.corrcoef(df_sorvete['Producao_Sorvete'], df_sorvete['Temperatura_Media'])[0,1]

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

x_temperatura_media = df_sorvete[['Temperatura_Media']]
y_producao_sorvete = df_sorvete[['Producao_Sorvete']]
modelo_simples_temperatura_media_producao_sorvete = LinearRegression()
modelo_simples_temperatura_media_producao_sorvete.fit(x_temperatura_media, y_producao_sorvete)
print('\nRegressão Linear Simples - Temperatura Média e Produção de Sorvete')
print(f"Coeficiente angular: {modelo_simples_temperatura_media_producao_sorvete.coef_[0][0]:.2f}")
print(f"Intercepto: {modelo_simples_temperatura_media_producao_sorvete.intercept_[0]:.2f}")
print(f"R²: {modelo_simples_temperatura_media_producao_sorvete.score(x_temperatura_media, y_producao_sorvete):.3f}")

print('\n---- MEDIDAS DESCRITIVAS -----')
print('Média da Produção de Sorvete: ',media_producao_sorvete)
print('Média da Temperatura Media: ',media_temperatura_media)
print(f"Correlação entre Temperatura Média e Produção de Sorvete: {corr_tempmedia_prodsorvete:.3f}")
print(f"Amplitude da Produção de Sorvete: {amplitude_producao_sorvete:.0f}")
print(f"Amplitude da Temperatura Media: {amplitude_temperatura_media:.1f}")
print(f"Mediana da Produção de Sorvete: {mediana_producao_sorvete:.0f}")
print(f"Mediana da Temperatura Media: {mediana_temperatura_media:.1f}")
print(f"Distância da Produção de Sorvete: {distancia_producao_sorvete:.1f}%")
print(f"Distância da Temperatura Media: {distancia_temperatura_media:.1f}%")
print(f"Nome do dia com maior Produção de Sorvete: {nome_maiorsorvete.values[0]}")
print()

print('\n---- GRAFICO -----')
# Visualizando os Dados Analisados
print('\n- Visualizando os Dados Analisados -')
plt.subplots(1,3,figsize=(16,4))
plt.suptitle('Análise dos Dados - Produção de Sorvetes e Temperatura Média',fontsize=12)

# Posição 01: Gráfico da produção de sorvetes no mês
plt.subplot(1,3,1)
plt.title('Gráfico da Produção de Sorvetes no Mês')
plt.bar(df_sorvete['Data'],df_sorvete['Producao_Sorvete'])

# Posição 02: Gráfico da temperatura média no mês
plt.subplot(1,3,2)
plt.title('Gráfico da Temperatura Média no Mês')
plt.bar(df_sorvete['Data'],df_sorvete['Temperatura_Media'])

# Posição 04: Gráfico de Regressão Linear da produção de sorvetes e temperatura média
plt.subplot(1,3,3)
plt.title('Produção de Sorvete x Temperatura Média')
plt.xlabel('Temperatura Média')
plt.ylabel('Produção de Sorvete')
plt.scatter(df_sorvete['Temperatura_Media'], df_sorvete['Producao_Sorvete'])
plt.plot(df_sorvete['Temperatura_Media'], modelo_simples_temperatura_media_producao_sorvete.predict(x_temperatura_media))
plt.show()

print('\n---- FIM DO PROGRAMA ----')
print()