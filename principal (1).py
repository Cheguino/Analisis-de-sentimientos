from cleantext import cleantext
import streamlit as st  
from textblob import TextBlob
import pandas as pd
import altair as alt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentiment_analysis_spanish import sentiment_analysis




# Fxn
def convert_to_df(sentiment):
    sentiment_dict = {'Polaridad': sentiment.polarity, 'Subjetividad': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['Metrica', 'Valor'])
    return sentiment_df

def analyze_token_sentiment(docx):
    clf = sentiment_analysis.SentimentAnalysisSpanish() # creacion del analizador de sentimientos

    pos_list = []
    neg_list = []
    neu_list = []

    for i in docx.split():
        resultado = clf.sentiment(i)
        # Análisis de sentimientos y clasificación:
        if resultado > 0.6: #mayor a 0.6
            pos_list.append(i)
            pos_list.append(resultado)

        elif resultado < 0.4: #menor 
            neg_list.append(i)
            neg_list.append(resultado)
        else:
            neu_list.append(i)
            neu_list.append(resultado)

    result = {'Positivos': pos_list, 'Negativos': neg_list, 'Neutrales': neu_list}
    return result
 




		






def main():
    #st.title("Análisis de ingreso")
    #st.subheader("U4 Proyecto")

    with st.form(key='nlpForm'):
        texto = st.text_area("Ingrese texto")
        submit_button = st.form_submit_button(label='Analizar')

    # layout
    col1, col2 = st.columns(2)
    if submit_button:
        with col1:
            st.info("Resultados")
            limpio = cleantext.clean(texto, clean_all= False, extra_spaces=True , stemming=False, 
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True, stp_lang='spanish')
            #st.write(limpio)
            sentiment = TextBlob(limpio).translate(from_lang='es', to='en').sentiment
            #st.write(sentiment)

            # Emoji
            if sentiment.polarity > 0:
                st.markdown("Sentiment:: Positive :😃")
            elif sentiment.polarity < 0:
                st.markdown("Sentiment:: Negative :😟 ")
            else:
                st.markdown("Sentiment:: Neutral 😐 ")

            # Dataframe
            result_df = convert_to_df(sentiment)
            st.dataframe(result_df)

            # Visualization
            c = alt.Chart(result_df).mark_bar().encode(
                x='Metrica',
                y='Valor',
                color='Metrica')
            st.altair_chart(c, use_container_width=True)

        with col2:
            st.info("Sentimiento")

            token_sentiments = analyze_token_sentiment(limpio)
            st.write(token_sentiments)


if __name__ == '__main__':
    main()