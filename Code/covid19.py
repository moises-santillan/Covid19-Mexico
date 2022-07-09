import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from scipy.stats import norm


def count_incidence(df, date, entity):
    if entity == 0:
        ci = len(df.loc[df.FECHA_SINTOMAS == date])
    else:
        ci = len(df.loc[(df.FECHA_SINTOMAS == str(date)) & (df.ENTIDAD_RES == entity)])
    return ci



date = pd.to_datetime('today') - pd.Timedelta('1 days')
path = "https://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip"
#os.system("curl " + path + " -o datos_abiertos_covid19.zip")
os.system("wget --no-check-certificate -O datos_abiertos_covid19.zip " + path)
os.system("unzip datos_abiertos_covid19.zip")
os.system("rm datos_abiertos_covid19.zip")
file = (pd.to_datetime('today') - pd.Timedelta('1 days')).strftime('%y%m%d')+"COVID19MEXICO.csv" #Database file
df = pd.read_csv(file, engine="python")
os.system("rm "+file)



Lag = 90
Date_Range = pd.date_range(end=pd.to_datetime('today').date(), periods=Lag)
columns = list(df.columns)
columns.remove('ENTIDAD_RES')
columns.remove('FECHA_SINTOMAS')
columns.remove('FECHA_INGRESO')
columns.remove('CLASIFICACION_FINAL')
df = df.drop(columns=columns)
df = df.drop(df[pd.to_datetime(df.FECHA_SINTOMAS) < Date_Range[0]].index)
df = df.drop(df[df.CLASIFICACION_FINAL > 3].index)
df = df.drop(columns=['CLASIFICACION_FINAL'])


df_Entities = pd.read_csv("Data/Entidades.csv")
df_Incidence = pd.DataFrame({'Date':Date_Range})
for entity in range(33):
    Incidence = []
    for date in range(Lag):
        Incidence.append(count_incidence(df, str(Date_Range[date].date()), entity) )
    df_Incidence[df_Entities.iloc[entity, 0]] = Incidence
df_Incidence.to_csv("Data/Incidence.csv", index=False)


Entities = list(df_Entities.Entidad)
df_Nowcasting = pd.DataFrame()
df_Nowcasting["Date"] = df_Incidence["Date"]
trst = 12
loc = (pd.to_datetime(df['FECHA_INGRESO'])-pd.to_datetime(df['FECHA_SINTOMAS'])).mean().total_seconds()/3600/24 + 3
scale = (pd.to_datetime(df['FECHA_INGRESO'])-pd.to_datetime(df['FECHA_SINTOMAS'])).std().total_seconds()/3600/24 + 1
for entity in Entities:
    aux = df_Incidence[entity].rolling(3, center=True).mean()
    aux[[0, Lag-1]] = df_Incidence[entity][[0, Lag-1]]
    aux /= norm.cdf(range(Lag-1, -1, -1), loc=loc, scale=scale)
    df_Nowcasting[entity] = aux
df_Nowcasting.to_csv("Data/Nowcasting.csv", index=False)


Incidences = []
Rates = []
for entity in Entities:
    Incidences.append(df_Nowcasting[entity][Lag-18:-4].sum())
    Rates.append(df_Nowcasting[entity][Lag-18:-4].sum() / df_Nowcasting[entity][Lag-32:Lag-18].sum() - 1)
df_Entities["Incidencia Semanal"] = Incidences
df_Entities['Tasa de Cambio'] = Rates
df_Entities["Incidencia Semanal Normalizada"] = df_Entities["Incidencia Semanal"] / df_Entities["Población"] * 1e5


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
fig.savefig('docs/Fig01.png')



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
fig.write_html("docs/Fig01.html")

os.system('Git "update"')
