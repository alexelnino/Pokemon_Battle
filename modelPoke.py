import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

dataPokemon=pd.read_csv('pokemon.csv')
dataCombat=pd.read_csv('combats.csv')
# print(dataPokemon.head(5))
# print(dataCombat.values[0])
# print(len(dataCombat.values))

dataCocok=pd.DataFrame(
    columns=['HP1','Attack1','Defense1','Special Attack1','Special Defense1',
    'Speed1','HP2','Attack2','Defense2','Special Attack2','Special Defense2','Speed2']
)

b=0
for a in dataCombat.values:
    dataCocok.loc[b]=(dataPokemon[dataPokemon['#']==a[0]]['HP'].values[0],dataPokemon[dataPokemon['#']==a[0]]['Attack'].values[0],dataPokemon[dataPokemon['#']==a[0]]['Defense'].values[0],dataPokemon[dataPokemon['#']==a[0]]['Sp. Atk'].values[0],dataPokemon[dataPokemon['#']==a[0]]['Sp. Def'].values[0],dataPokemon[dataPokemon['#']==a[0]]['Speed'].values[0],dataPokemon[dataPokemon['#']==a[1]]['HP'].values[0],dataPokemon[dataPokemon['#']==a[1]]['Attack'].values[0],dataPokemon[dataPokemon['#']==a[1]]['Defense'].values[0],dataPokemon[dataPokemon['#']==a[1]]['Sp. Atk'].values[0],dataPokemon[dataPokemon['#']==a[1]]['Sp. Def'].values[0],dataPokemon[dataPokemon['#']==a[1]]['Speed'].values[0])
    b+=1
    if b==1000:
        break
    # print(dataCocok.iloc[0])
    # break

b=0
for a in dataCombat['Winner']:
    if a==dataCombat.iloc[b]['First_pokemon']:
        dataCombat.iloc[b]['Winner']=1
        b+=1
    else:
        dataCombat.iloc[b]['Winner']=0
        b+=1
target=dataCombat['Winner'].iloc[0:1000]

# print(target)
# print(dataCocok)
# print(np.matrix(dataCocok))

# random forest classifier
from sklearn.ensemble import RandomForestClassifier
modelRFC = RandomForestClassifier(n_estimators=50)
modelRFC.fit(np.matrix(dataCocok),target)

model = LogisticRegression(solver = 'lbfgs')
model.fit(np.matrix(dataCocok),target)

poke1='Charizard'
poke2='Pikachu'
indexpoke1=dataPokemon[dataPokemon['Name']==poke1]['#'].values[0]
indexpoke2=dataPokemon[dataPokemon['Name']==poke2]['#'].values[0]
statpoke=[]
statpoke.append([dataPokemon[dataPokemon['#']==indexpoke1]['HP'].values[0],dataPokemon[dataPokemon['#']==indexpoke1]['Attack'].values[0],dataPokemon[dataPokemon['#']==indexpoke1]['Defense'].values[0],dataPokemon[dataPokemon['#']==indexpoke1]['Sp. Atk'].values[0],dataPokemon[dataPokemon['#']==indexpoke1]['Sp. Def'].values[0],dataPokemon[dataPokemon['#']==indexpoke1]['Speed'].values[0],dataPokemon[dataPokemon['#']==indexpoke2]['HP'].values[0],dataPokemon[dataPokemon['#']==indexpoke2]['Attack'].values[0],dataPokemon[dataPokemon['#']==indexpoke2]['Defense'].values[0],dataPokemon[dataPokemon['#']==indexpoke2]['Sp. Atk'].values[0],dataPokemon[dataPokemon['#']==indexpoke2]['Sp. Def'].values[0],dataPokemon[dataPokemon['#']==indexpoke2]['Speed'].values[0]])
print(statpoke)

print(model.predict_proba(statpoke))
print(model.predict(statpoke))
proba=modelRFC.predict_proba(statpoke)
print('score')
print('score',(model.score(np.matrix(dataCocok),target))*100 ,'%')
print('score RFC',(modelRFC.score(np.matrix(dataCocok),target))*100 ,'%')       
if proba[0][0]>=proba[0][1]:
    print(str(proba[0][0]*100)+'%',str(poke2),'Wins!')
else:
    print(str(proba[0][1]*100)+'%',str(poke2),'Wins!')

from sklearn.externals import joblib as jb
jb.dump(model,'modelPoke')