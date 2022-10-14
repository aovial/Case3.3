#!/usr/bin/env python
# coding: utf-8

# <b> Groep 22: Bente, Noah & Lukas. <b>

# PACKAGES.

# In[1]:


#Streamlit & dash installeren. 
#pip install streamlit
#pip install streamlit_folium
#pip install seaborn


# In[2]:


#importeren van packages.
import pandas as pd
import numpy as np
import plotly.express as px 
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import streamlit as st
import geopandas as gpd
import requests
import streamlit
import folium 
import json
import streamlit_folium as st_folium
from streamlit_folium import folium_static


# DATA.

# In[3]:


#Dataset laadpaaldata inladen. 
laadpaal = pd.read_csv('laadpaaldata.csv')
lp = pd.DataFrame(laadpaal)
lph = laadpaal.head()
   #print(laadpaal)


# In[4]:


#Datasets van RDW inladen. 
#De eerste csv bevat personenauto's met een eerste tenaamstelling > 01-01-2022 & eerste toelating > 01-01-2022
#Link naar data: https://opendata.rdw.nl/Voertuigen/M1-tenaamstelling-toelating-01-01-2022-/qn6h-6bru .
rdw = pd.read_csv('rdw.csv')
#Data filteren en opslaan onder zelfde naam. 
#rdw = rdw.filter(['Kenteken', 'Voertuigsoort', 'Merk', 'Handelsbenaming', 'Datum tenaamstelling',
                                      #'Eerste kleur', 'Aantal Cilinders', 'Cilinderinhoud','Datum eerste toelating', 
                                       #'Datum eerste tenaamstelling in Nederland', 'Catalogusprijs'])
    #rdw.to_csv('rdw.csv')
#De tweede csv is de eerste dataset maar er is gefilterd op alle auto's die geen cilinders hebben. 
rdwel = pd.read_csv('rdwel.csv')
#Data filteren en opslaan onder zelfde naam.
#rdwel = rdwel.filter(['Kenteken', 'Voertuigsoort', 'Merk', 'Handelsbenaming', 'Datum tenaamstelling',
                                      #'Eerste kleur', 'Aantal Cilinders', 'Cilinderinhoud','Datum eerste toelating', 
                                       #'Datum eerste tenaamstelling in Nederland', 'Catalogusprijs'])
    #rdw.to_csv('rdw.csv')
    
rdwh = rdw.head()
rdwelh = rdwel.head()


# In[5]:


#Datums overzetten naar dt, en nieuwe kolom aanmaken met maanden van toelating. 
rdw['Maand_et'] = pd.to_datetime(rdw['Datum eerste toelating'].astype(str), format='%Y%m%d')
rdw['Maand_et'] = pd.DatetimeIndex(rdw['Maand_et']).month
#Maanden op juiste volgorde zetten, en index vervangen voor naam van de maand geschreven in het Nederlands.
rdw['Maand_et'].value_counts().sort_index(ascending=True)
md = rdw['Maand_et'].value_counts().sort_index(ascending=True).to_frame()
maanden = ['Januari', 'Februari', 'Maart', 'April', 'Mei', 'Juni', 'Juli', 'Augustus', 'September', 'Oktober']
md['Maanden'] = maanden
md.set_index('Maanden', inplace=True)

#Datums overzetten naar dt, en nieuwe kolom aanmaken met maanden van toelating. 
rdwel['Maand_et_el'] = pd.to_datetime(rdwel['Datum eerste toelating'].astype(str), format='%Y%m%d')
rdwel['Maand_et_el'] = pd.DatetimeIndex(rdwel['Maand_et_el']).month
#Maanden op juiste volgorde zetten, en index vervangen voor naam van de maand geschreven in het Nederlands.
rdwel['Maand_et_el'].value_counts().sort_index(ascending=True)
mdel = rdwel['Maand_et_el'].value_counts().sort_index(ascending=True).to_frame()
maanden = ['Januari', 'Februari', 'Maart', 'April', 'Mei', 'Juni', 'Juli', 'Augustus', 'September', 'Oktober']
mdel['Maanden'] = maanden
mdel.set_index('Maanden', inplace=True)

#De twee gemaakte dataframes samenvoegen voor het maken van plots. 
mx = pd.merge(md, mdel, how='inner', on='Maanden')


# In[6]:


#API inladen
response = requests.get("https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=8000&compact=true&verbose=false&key=a4790509-3683-4a91-9797-e7c2bb144fe4").json()
   #print(response)


# In[7]:


#API omzetten naar dataframe. 
laadpalen = pd.json_normalize(response)
df4 = pd.json_normalize(laadpalen.Connections)
df5 = pd.json_normalize(df4[0])
   #df5.head()
   #print(df5)
df5h = laadpalen.head()


# In[8]:


#Samenvoegen van data. 
laadpalen = pd.concat([laadpalen, df5], axis=1)
   #laadpalen.head()
   #Describe.
laadpalen_describe = laadpalen.describe()
   #print(laadpalen_describe)


# In[9]:


# Dataframe verkleinen naar de kolommen die wij nodig hebben 
laadpalen_filter = laadpalen[['DateLastVerified', 'ID', 'DataProviderID', 'OperatorID', 'UsageTypeID', 'DateCreated', 
                             'AddressInfo.ID', 'AddressInfo.Title', 'AddressInfo.Postcode', 'AddressInfo.StateOrProvince', 'AddressInfo.Latitude', 'AddressInfo.Longitude']]
   #laadpalen_filter


# In[10]:


#Een nieuw dataframe gemaakt van alle rows die een provincie missen.
missing_province = laadpalen_filter[laadpalen_filter['AddressInfo.StateOrProvince'].isna()]
   #missing_province


# In[11]:


#Wij hebben een dataset gevonden waarin alle postcodes en provincies zijn gekoppeld. 
#Via deze manier willen wij de missende provincies toevoegen aan de dataset. 
#Hiervoor hebben wij van de postcodes alleen de cijfers nodig. 
#Hiervoor hebben wij bij beide datasets nieuwe column aangemaakt. Deze merge we vervolgens naar 1 dataset.
missing_province['AddressInfo.Postcode'] = missing_province['AddressInfo.Postcode'].astype(str)
missing_province['AddressInfo.Postcode_1'] = missing_province['AddressInfo.Postcode'].str[:4]

data_postcode = pd.read_csv('https://raw.githubusercontent.com/EdwinTh/pc4_prov/master/pc4_provincie_mapping.txt')
postcode = pd.DataFrame(data_postcode)
postcode['pc4'] = postcode['pc4'].astype(str)
postcode.rename(columns = {'pc4':'AddressInfo.Postcode_1'}, inplace=True)


# In[12]:


data_postcode = pd.read_csv('https://raw.githubusercontent.com/EdwinTh/pc4_prov/master/pc4_provincie_mapping.txt')


# In[ ]:





# In[13]:


#Mergen van de sets.
missing_province = pd.merge(missing_province, postcode, how='inner', on='AddressInfo.Postcode_1')


# In[14]:


#Wij vervangen de province door de provincie van de andere dataset. 
#Als deze is aangevuld laten wij de toegevoegde kolommen weer vallen. Zodat onze dataset compact blijft.
missing_province['AddressInfo.StateOrProvince'] = missing_province['provincie']
missing_province['AddressInfo.StateOrProvince'] = missing_province['AddressInfo.StateOrProvince']
missing_province.drop(labels = ['AddressInfo.Postcode_1', 'provincie'], axis=1, inplace=True)


# In[15]:


#Daarna filteren wij de data met provincies uit de dataset. Om met_province en missing_province hierna weer samen te voegen.
met_province = laadpalen_filter[~ laadpalen_filter['AddressInfo.StateOrProvince'].isna()]
  #met_province.head()

alle_palen = pd.concat([met_province, missing_province], ignore_index=True)
  #alle_palen


# In[16]:


#Wij gaan bij alle_palen kijken welke unique waarden er zijn voor provincie. Om zo te achterhalen of alle provincies hetzelfde zijn geschreven. 
#Het blijkt dat dit niet zo is. Hierdoor gaan wij alle waardes herschrijven naar dezelfde waardes als in onze geojson file.
   #alle_palen['AddressInfo.StateOrProvince'].unique()
    
alle_palen['AddressInfo.StateOrProvince']=alle_palen['AddressInfo.StateOrProvince'].astype(str)
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].str.lower()


# In[17]:


#Toevoegen van geojson file met provincies. 
map_provincies = gpd.read_file('provinces.geojson')
   #map_provincies['name'].unique()


# In[18]:


alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['north holland', 'nh', 'north-holland', 'nordholland', 'noord holand', 'holandia północna', 'noord holland', 'noord-hooland', 'stadsregio amsterdam', 'noord-holland'], 'Noord-Holland')
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['north brabant', 'noord brabant', 'nordbrabant', 'noord brabant ', 'noord-brabant', 'samenwerkingsverband regio eindhoven'], 'Noord-Brabant')
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['ut', 'UTRECHT', 'utrecht'], 'Utrecht')
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['south holland', 'zh', 'zuid-holland ', 'stellendam', 'zuid holland', 'mrdh', 'stadsgewest haaglanden', 'stadsregio rotterdam', 'zuid-holland'], value='Zuid-Holland')
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['frl', 'friesland'], 'Friesland (Fryslân)')
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['seeland', 'zeeland'], 'Zeeland')
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['stadsregio arnhem nijmegen', 'gld', 'gelderland ', 'gelderland'], 'Gelderland')
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['overijsel', 'overijsel', 'regio twente', 'regio zwolle', 'overijssel'], 'Overijssel')
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['drente', 'drenthe'], 'Drenthe')
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['flevolaan', 'flevoland'], 'Flevoland')
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['groningen'], 'Groningen')
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['limburg', 'stadsregio arnoord-hollandem nijmegen'], 'Limburg')
alle_palen['AddressInfo.StateOrProvince'] = alle_palen['AddressInfo.StateOrProvince'].replace(['', ' '], np.nan)

   #alle_palen['AddressInfo.StateOrProvince'].unique()


# In[19]:


#Alle waardes met nan laten wij als laatst nog vallen. Dit zodat elke laadpaal aan een provincie is gekoppeld.
alle_palen.dropna(subset=['AddressInfo.StateOrProvince'])


# Daarna merge wij de map_provincies met alle_laadpalen, zodat alle laadpalen een geo data bevatten. Hierna kunnen wij beginnen met het plotten van een map. Om de informatie weer te geven moesten wij int gemaakt worden van name. Daarnaast is er een totale aantal toegevoegd om zo een mooi beeld te krijgen van alle laadpalen per provincie.

# In[20]:


#Merge map provincies en alle palen

laadpaal_map_compleet = map_provincies.merge(alle_palen, how='inner', 
                                 right_on='AddressInfo.StateOrProvince', left_on='name')

   #laadpaal_map_compleet.head()
    
laadpaal_map_compleet['count'] = 1

  #laadpaal_map_compleet.head()
    
laadpaal_map_compleet['name'] = laadpaal_map_compleet['name'].astype(str)


# In[21]:


totaal_per_provincie = laadpaal_map_compleet.name.value_counts().reset_index(name='totale_laadpalen')

   #totaal_per_provincie


# In[22]:


laadpaal_map_compleet2 = laadpaal_map_compleet.merge(totaal_per_provincie, how='inner', 
                                 right_on='index', left_on='name')
#laadpaal_map_compleet2


# TITEL/OPMAAK APP.

# In[23]:


st.set_page_config(page_title="Dashboard Groep 22", page_icon="🚗", layout = "wide", initial_sidebar_state="expanded")


# In[24]:


#Title voor de app, weergegeven boven elke pagina.  
st.title('Dashboard Elektrische Mobiliteit en Laadpalen')


# In[25]:


#Opmaak van sidebar in de app. 
st.sidebar.title('Navigatie')


# PLOTS.

# In[26]:


lp24 = lp[~(lp['ChargeTime'] < 0) & ~(lp['ChargeTime'] > 24)] 
#alp24 = lp24[~(lp['ConnectedTime'] < 0) & ~(lp24['ConnectedTime'] > 24)] 
lp12 = lp[~(lp['ChargeTime'] < 0) & ~(lp['ChargeTime'] > 12)]
#alp12 = lp12[~(lp['ConnectedTime'] < 0) & ~(lp12['ConnectedTime'] > 24)] 
lp8 = lp[~(lp['ChargeTime'] < 0) & ~(lp['ChargeTime'] > 8)]
#alp8 = lp8[~(lp['ConnectedTime'] < 0) & ~(lp8['ConnectedTime'] > 24)] 


# In[27]:


x1 = lp['ChargeTime'].mean().round(3)
#x1


# In[28]:


x2 = lp['ChargeTime'].median()
#x2


# In[29]:


fig10 = go.Figure(px.histogram(lp12, x='ChargeTime', nbins=30))
fig10.update_layout(title= 'Hoeveelheid Laadtijd tot 12 uur lang',
                  xaxis_title="Laadtijd (uren)", yaxis_title="hoeveelheid")
fig10.add_annotation(text='mediaan = 1.77<br>gemiddelde = 2.33 ', 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=1,
                    y=0.5,
                    bordercolor='black',
                    borderwidth=1) 
#fig10


# In[30]:


fig11 = go.Figure(px.histogram(lp8, x='ChargeTime', nbins=30))
fig11.update_layout(title= 'Hoeveelheid Laadtijd tot 8 uur lang',
                  xaxis_title="Laadtijd (uren)", yaxis_title="hoeveelheid")

fig11.add_annotation(text='mediaan = 1.77<br>gemiddelde = 2.33 ', 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=1,
                    y=0.5,
                    bordercolor='black',
                    borderwidth=1)         
#fig11


# In[31]:


#fig22 = go.Figure()
#fig22.add_trace(go.Histogram(x=lp242['ChargeTime']))
#fig22.add_trace(go.Histogram(x=lp242['ConnectedTime']))
#fig22


# In[32]:


#Staafdiagram die het aantal toelatingen in 2022 laat zien. 
fig1 = go.Figure(px.bar(md, x=md.index, y='Maand_et', color=md.index, title='Toelatingen per maand 2022', text_auto=True))
fig1.update_layout(showlegend=True,
                  xaxis_title="Maanden", yaxis_title="Aantal toelatingen")
#fig1.show()


# In[33]:


#Staafdiagram die het aantal toelatingen van elektrische auto's in 2022 laat zien. 
fig2 = px.bar(mdel, x=md.index, y='Maand_et_el', color=mdel.index, title="Toelatingen Elektrische Auto's per maand 2022", text_auto=True,
            labels={'Maand_et_el':'Aantal Toelatingen', 'x':'Maanden'})
fig2.update_layout(showlegend=False)
#fig2.show()


# In[34]:


#lijndiagram die de toelatingen t.o.v. elkaar laat zien. 
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=mx.index, y=mx["Maand_et"], name="Toelatingen Totaal ", mode="lines+markers"))
fig3.add_trace(go.Scatter(x=mx.index, y=mx["Maand_et_el"], name="Toelatingen Elektrisch", mode="lines+markers"))
fig3.update_layout(title="Toelatingen Elektrische Auto's ten opzichte van Totale Toelatingen",
                 xaxis_title="Maanden", yaxis_title="Aantal toelatingen",)
#fig3.show()


# In[35]:


#m = folium.Map(location= [52.1326, 5.2913], zoom_start=5,
              # tiles='Cartodb Positron')

#folium.Choropleth(geo_data=laadpaal_map_compleet2,
                 #name = 'geometry',
                 #data=laadpaal_map_compleet2,
                 #columns=['name', 'totale_laadpalen'],
                 #key_on='feature.properties.name',
                 #fill_color='YlGn',
                #fill_opacity=0.7,
                #line_opacity=0.2,
                 #legend_name="Totaal aantal laadpalen per provincie, 2022").add_to(m)

#folium.LayerControl().add_to(m)
#m


# In[36]:


from folium.plugins import MarkerCluster

m = folium.Map(location= [52.1326, 5.2913], zoom_start=7)

cluster = MarkerCluster().add_to(m)

for index, row in laadpaal_map_compleet.iterrows():
    marker=(folium.CircleMarker(location=[row['AddressInfo.Latitude'], row['AddressInfo.Longitude']],
                        popup= row['AddressInfo.Title'],
                         fill=True,
                               radius=0.2))
    marker.add_to(cluster)

m
#figkaart


# OPMAAK APP.

# In[37]:


#Opmaak Dashboard tussen 'knoppen' bij bijbehorende pagina. 

#Radioknoppen in de sidebar die navigatie over de pagina mogelijk maken. 
pages = st.sidebar.radio('paginas', options=['Home','Datasets', 'Visualisaties', 'Einde'], label_visibility='hidden')

if pages == 'Home':
    st.markdown("Welkom op het dashboard van groep 22. Gebruik de knoppen in de sidebar om tussen de verschillende paginas te navigeren. ")
    st.image("hva.png", width=None ,output_format='auto')
elif pages == 'Datasets':
    st.subheader('Gebruikte Datasets.')
    st.markdown("Hieronder wordt de dataset met data over het gebruik van de laadpaal weergegeven.")
    st.dataframe(data=lph, use_container_width=False)
    st.subheader('Dataset RDW.')
    st.markdown("Dataset met kentekeninformatie. ")
    st.dataframe(data=rdwh, use_container_width=False)
    st.markdown("Bron Dataset: https://opendata.rdw.nl/Voertuigen/M1-tenaamstelling-toelating-01-01-2022-/qn6h-6bru")
    st.subheader('Dataset Open Charge Map')
    st.markdown("Dataset met locaties van laadpalen, gebruikt om een kaart van laadpalen in Nederland te maken. Als API ingeladen van open charge map.")
    st.dataframe(data=df5h, use_container_width=False)
    st.markdown("Bron Dataset: https://openchargemap.org/site/develop#api")
elif pages == 'Visualisaties':
    st.subheader("Hier worden de visualisaties weergegeven die wij hebben opgesteld."), st.image("map.png", width=None ,output_format='auto'), folium_static(m), st.image("12uur.png", width=None, output_format='auto'), st.image("8uur.png", width=None, output_format='auto'), st.plotly_chart(fig1), st.plotly_chart(fig2), st.plotly_chart(fig3) 
elif pages == 'Einde':
    st.markdown('Bedankt voor het bezoeken.')
    st.markdown('Groep 22: Bente van Hameren, Noah Wijnheimer, Lukas Čović. ')


# In[ ]:





# Test/werk
