#!/usr/bin/env python
# coding: utf-8

# # Análise de dados com Python
# ## Esse treinamento apresenta algumas das principais ferramentas disponíveis na linguagem python para análise de dados. A busca dos entendimentos se passa a partir dos eixos:
# ### 1 - Compreensão dos dados
# ### 2 - Análise estatística dos dados
# ### 3 - Operações em DataFrames
# ### 4 - Análises de negócio
# ### 5 - SVM como classificador para o problema da predição
# 
# ##### Fonte: IBGE / DETRAN

# ## 1 - Compreensão dos dados

# In[1]:


import pandas as pd
df_frota = pd.read_csv('frotarj.csv',sep=';')


# In[2]:


df_frota


# In[3]:


df_frota.describe() # Análise descritiva da amostra 


# In[4]:


df_frota.loc[df_frota.MUNICIPIO == "RIO DE JANEIRO"]


# In[5]:


df_frota.loc[:,"MUNICIPIO"]


# In[6]:


df_frota.loc[df_frota.AUTOMOVEL > 100000]


# In[7]:


df_frota_auto_bus = df_frota.loc[:,["MUNICIPIO","AUTOMOVEL","ONIBUS"]]


# In[8]:


df_frota_auto_bus["RELATION"] = df_frota_auto_bus["AUTOMOVEL"] / df_frota_auto_bus["ONIBUS"]


# In[10]:


df_frota_auto_bus


# ## 2 - Análises estatísticas dos dados

# In[57]:


df_frota_auto_bus["RELATION"].describe() # Análise descritiva da variavel RELATION na amostra composta


# In[58]:


import matplotlib # Biblioteca para plot dos dados, facilitadores para análise estatística


# In[16]:


df_frota_auto_bus.plot.scatter('AUTOMOVEL','RELATION',s=10);


# In[20]:


df_frota_auto_bus.plot.scatter('RELATION','AUTOMOVEL',s=10);


# In[26]:


df_frota_auto_bus.AUTOMOVEL.plot.hist();


# In[27]:


df_frota_auto_bus.ONIBUS.plot.hist();


# ### O histograma abaixo apresenta a distribuição nos valores da relação entre total de automóveis pelo total de ônibus nas cidades do estado do Rio de Janeiro

# In[29]:


df_frota_auto_bus.RELATION.plot.hist();


# In[30]:


df_frota_auto_bus.RELATION.plot.kde();


# In[31]:


df_frota_auto_bus.RELATION.var()


# O calculo abaixo representa a medida do desvio padrão dos dados em relação à média amostral

# In[36]:


df_frota_auto_bus.RELATION.std()


# In[37]:


df_frota_auto_bus.RELATION.mean()


# Calculo da correlacao entre automovel e onibus

# In[61]:


df_frota_auto_bus.AUTOMOVEL.corr(df_frota_auto_bus.ONIBUS) # correlacao linear positiva


# In[62]:


df_frota_auto_bus.corr() # correlacao entre os atributos do dataset


# In[63]:


df_frota.corr() # correlacao entre todas as variaveis do problema


# ## 3 - Operações em DataFrames

# In[70]:


df_cidades = pd.read_csv('cidaderj.csv',sep=';')


# In[74]:


df_cidades


# In[77]:


df_cidades.HABITANTES.corr(df_cidades.PIB_CAPITA_1) # Baixa correlação entre HABITANTES / PIB


# In[81]:


df_cidades.loc[df_cidades.MUNICIPIO == 'RIO DE JANEIRO']


# In[82]:


df_cidades.loc[df_cidades.MUNICIPIO == 'NITEROI']


# In[86]:


df_cidades.loc[df_cidades.PIB_CAPITA_1 == df_cidades.PIB_CAPITA_1.max()] # Filter higher PIB


# In[87]:


df_cidades.loc[df_cidades.PIB_CAPITA_1 == df_cidades.PIB_CAPITA_1.min()] # Filter lower PIB


# In[89]:


df_cidades.sort_values(by=['PIB_CAPITA_1'], ascending=True)[0:5] # Ordem decrescente por PIB


# In[90]:


df_cidades.sort_values(by=['PIB_CAPITA_1'], ascending=False)[0:5] # Ordem decrescente por PIB


# In[92]:


df_cidades.PIB_CAPITA_1.plot.hist(); # So por curiosidade - agrupamento do PIB


# In[93]:


df_cidades.PIB_CAPITA_1.plot.kde(); # So por curiosidade - distribuição do PIB


# Vamos combinar os DataFrames para extrair alguns insights

# In[104]:


df_cidade_frota = pd.merge(df_frota, df_cidades, on='MUNICIPIO') # Combine data based on MUNICIPIO


# In[105]:


df_cidade_frota


# ## 4 - Análise de negócio
# #### Algumas análises de negócio
# 
# ##### 1 - Apresente a relação entre número de automóveis por habitante nas 5 cidades com maior PIB
# ##### 2 - Uma famosa empresa fabricante de pneus para caminhões deseja montar uma fábrica no estado do RJ. Apresente aos gestores uma cidade candidata à abertura da fábrica que atenda pelo menos 3 cidades com baixo custo operacional de deslocamento. (Este é um exercício de estratégia de negócio, utilize outras fontes de consulta para avaliar a sua decisão)

# 1 - Relação entre número de automóveis por habitante nas 5 cidades com maior PIB

# In[128]:


df_pib = df_cidades.sort_values(by=['PIB_CAPITA_1'], ascending=False)[0:5] # Filtra os 5 melhores PIB


# In[135]:


df_pib_auto = pd.merge(df_frota.loc[:,['MUNICIPIO','AUTOMOVEL']], df_pib.loc[:,['MUNICIPIO', 'PIB_CAPITA_1','HABITANTES']],on='MUNICIPIO') # Merge nos DFs


# In[136]:


df_pib_auto['RELATION_HAB_AUTO'] = df_pib_auto.HABITANTES / df_pib_auto.AUTOMOVEL


# In[137]:


df_pib_auto = df_pib_auto.sort_values(['RELATION_HAB_AUTO'])
df_pib_auto


# ## 5 - SVM como classificador para predição

# In[138]:


df_cidades.describe()


# In[151]:


svm_data = df_cidades.loc[:,['VL_BRUTO_AGROP_1000','VL_BRUTO_INDUSTRIA_1000','VL_BRUTO_SERV_1000','VL_BRUTO_ADMDEF_1000','VL_BRUTO_PRECCORR_1000','IMPOSTOS_PROD_1000','PIB_PRECCORR_1000','HABITANTES','PIB_CAPITA_1']]
svm_data_mean = svm_data.PIB_CAPITA_1.mean()
svm_data.PIB_CAPITA_1 = svm_data.PIB_CAPITA_1.apply(lambda x: 0 if x <= svm_data_mean else 1)
svm_data # Conditions base on PIB numbers to have a "good" PIB_PER_CAPITA


# In[157]:


from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
# get labels - y
labels   = np.array(svm_data['PIB_CAPITA_1'])
# get features values - x1,x2,...
features = svm_data.drop('PIB_CAPITA_1', axis = 1)
# name of columns
feature_list = list(features.columns)
# convert to numpy array
features = np.array(svm_data)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)


# In[170]:


clf = svm.SVC(gamma='scale',random_state=0)


# In[171]:


clf.fit(train_features, train_labels) # Executa o aprendizado


# In[172]:


predictions = clf.predict(test_features) # Faz a predição


# In[173]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_labels, predictions)
accuracy # accuracia de 69%


# Vamos rodar o mesmo exemplo com outro classificador - floresta aleatória

# In[175]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)


# In[176]:


clf.fit(train_features, train_labels) # Executa o aprendizado
predictions = clf.predict(test_features) # Faz a predição


# In[177]:


accuracy = accuracy_score(test_labels, predictions)
accuracy # accuracia de 86%

