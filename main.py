import pandas as pd
import streamlit as st

st.title("Partidas Ranqueadas - LOL")
st.write('')

df = pd.read_csv('.\MatchTimelinesFirst15.csv')


st.write("Tabela de dados completa:")
df

st.write('')
check=st.checkbox('Mostrar Tabela Filtrada')

if check:
    st.write('')
    st.write("Tabela filtrada com o Gold do time azul acima de Trinta mil:")
    df[df['blueGold'] >= 30000.00]

st.set_option('deprecation.showPyplotGlobalUse', False)
X = df.drop(['Unnamed: 0', 'matchId', 'blue_win'], 1)
y = df['blue_win']

st.write('')
check2= st.checkbox('Mostrar Gráficos de Gold, Minions da Jungle e Minions Abatidos do time Azul')

st.write('')
if check2:
    df = pd.DataFrame(df[:200], columns=['blueGold', 'blueJungleMinionsKilled', 'blueMinionsKilled'])
    df.hist()
    st.pyplot()


check3= st.checkbox('Mostrar Regressão Logística')


entrada_test_size = st.sidebar.slider("Test-Size", 0.01, 0.99, 0.01)
entrada_random_size= st.sidebar.slider("Random-State", 0, 99, 20)

st.write('')
if check3:
    st.write("Regressão Logística:")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=entrada_test_size, random_state=entrada_random_size)
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression(max_iter=48649)
    logreg.fit(X_train, y_train)
    prediction = logreg.predict(X_test)
    from sklearn.metrics import classification_report
    st.write(classification_report(y_test, prediction))