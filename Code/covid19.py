import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.deterministic import DeterministicProcess

def count_incidence(df, date, entity):
    if entity == 0:
        ci = len(df.loc[df.FECHA_SINTOMAS == date])
    else:
        ci = len(df.loc[(df.FECHA_SINTOMAS == str(date)) & (df.ENTIDAD_RES == entity)])
    return ci

os.system("wget https://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip")
os.system("unzip datos_abiertos_covid19.zip")
os.system("rm datos_abiertos_covid19.zip")

file = (pd.to_datetime('today') - pd.Timedelta('1 days')).strftime('%y%m%d')+"COVID19MEXICO.csv" #Database file
Lag = 72 #10 weeks into the past
Date_Range = pd.date_range(end=pd.to_datetime('today').date(), periods=Lag)

df = pd.read_csv(file, engine="python")
columns = list(df.columns)
columns.remove('ENTIDAD_RES')
columns.remove('FECHA_SINTOMAS')
columns.remove('CLASIFICACION_FINAL')
df = df.drop(columns=columns)
df = df.drop(df[pd.to_datetime(df.FECHA_SINTOMAS) < Date_Range[0]].index)
df = df.drop(df[df.CLASIFICACION_FINAL > 3].index)
df = df.drop(columns=['CLASIFICACION_FINAL'])

df_Entities = pd.read_csv("../Data/Entidades.csv")

df_Incidence = pd.DataFrame({'Date':Date_Range})
for entity in range(33):
    Incidence = []
    for date in range(Lag):
        Incidence.append(count_incidence(df, str(Date_Range[date].date()), entity) )
    df_Incidence[df_Entities.iloc[entity, 0]] = Incidence

Entities = list(df_Entities.Entidad)
Incidences = []
Rates = []
trst = 17
for entity in Entities:
    dp = DeterministicProcess(df_Incidence[entity][:Lag-trst].index, constant=False, period=7, fourier=3)
    model_fit = AutoReg(df_Incidence[entity][:Lag-trst], lags=5, trend='n', seasonal=False, deterministic=dp).fit()
    predictions = model_fit.predict(start=Lag-trst, end=Lag-1)
    Incidences.append(sum(predictions[-12:-5]))
    Rates.append(sum(predictions[-12:-5])/sum(predictions[-19:-12]) - 1)

df_Entities['Incidencia Semanal'] = Incidences
df_Entities['Incidencia Semanal Normalizada'] = df_Entities['Incidencia Semanal']/df_Entities['Población']*1e5
df_Entities['Tasa de Cambio'] = Rates


fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
df_Entities.plot.bar(x='Entidad', y='Incidencia Semanal Normalizada', color='C3', ax=axs[0])
axs[0].set_title('Última actualización: '+pd.to_datetime('today').strftime('%d-%m-%y'), fontsize=13)
axs[0].set_xlabel('')
axs[0].set_xticks([])
axs[0].yaxis.set_tick_params(labelsize=13)
axs[0].legend(fontsize=13)
df_Entities.plot.bar(x='Entidad', y='Tasa de Cambio', color='C3', ax=axs[1])
axs[1].set_ylim([-1, 1])
axs[1].set_xlabel('')
axs[1].xaxis.set_tick_params(labelsize=13)
axs[1].yaxis.set_tick_params(labelsize=13)
axs[1].legend(fontsize=13)
fig.tight_layout()
fig.savefig('../docs/Fig01.png', transparent=True)



fig1 = px.bar(df_Entities, x='Entidad', y='Incidencia Semanal Normalizada')
fig2 = px.bar(df_Entities, x='Entidad', y='Tasa de Cambio')

fig = make_subplots(rows=2, cols=1)
fig.add_trace(fig1['data'][0], row=1, col=1)
fig.add_trace(fig2['data'][0], row=2, col=1)
fig.update_layout(title_text="Útlima actialización: "+pd.to_datetime('today') .strftime('%d-%m-%y'),
    height=600
    )
fig.update_xaxes(showticklabels=False, title=None, row=1, col=1)
fig.update_yaxes(title="Incidencia Semanal Normalizada", row=1, col=1)
fig.update_yaxes(title="Tasa de Cambio", row=2, col=1)

fig.show()
fig.write_html("../docs/Fig01.html")

os.system("rm "+file)

