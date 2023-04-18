import streamlit as st
import requests
import pandas
st.set_page_config(page_title='Kobe shot predict', layout='wide')

def perform_inference(data: pandas.Series, url = 'http://localhost:5001/invocations'):
    print(data.drop(columns=['shot_made_flag']))
    print(data.drop(columns=['shot_made_flag']).to_dict(orient='records'))
    data = {
        "dataframe_records": 
            data.drop(columns=['shot_made_flag']).to_dict(orient='records'),      
    }

    response = requests.post(url, json=data)
    results = response.json()
    pred = results['predictions']
    return pred


intro = """
# Exemplo de kobe bryant shot selection

Este exemplo usará um modelo treinado como dataset [Kobe Bryant Shot Selection
](https://www.kaggle.com/competitions/kobe-bryant-shot-selection).
"""

st.write(intro)

    
lat = st.text_input('latitude')
lon = st.text_input('longitude')
minutes_remaining = st.text_input('Minutos faltantes')
period = st.text_input('periodo')
playoffs = st.text_input('playoffs')
shot_distance = st.text_input('Distância')


# Quando o usuário clicar em um botão de submit, salva o novo registro no dataset
if st.button('Adicionar registro'):
    df = pandas.DataFrame({
        'lat': [float(lat)],
        'lon': [float(lon)],
        'minutes_remaining': [int(minutes_remaining)],
        'period': [int(period)],
        'playoffs': [int(playoffs)],
        'shot_distance': [int(shot_distance)],
        'shot_made_flag': [None] # valor indefinido, pois ainda não sabemos se será um acerto ou erro
    })
     
    st.write(df.head())
    pred = perform_inference(df)
    st.write(f"A predição é {pred}")
