import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
import twint
import string 
from textblob import TextBlob
import re 
from wordcloud import WordCloud,STOPWORDS
import pickle
stopwords=set(STOPWORDS)


# Set page name and favicon
st.set_page_config(page_title='Twitter scraper',page_icon=':iphone:')


st.image('twii2.jpeg',width=40)
st.subheader("""
Tweep Stalker App :smile:
""")

# customize form
with st.form(key='Twitter_form'):
    username= st.text_input('Who do you want to know more about?')
    limit = st.slider('How many tweets do you want to analysed?', 0, 3000, step=100)
    
    submit_button = st.form_submit_button(label='Search')

    if submit_button:
        # configure twint
        c = twint.Config()
        c.Limit=1000
        c.Username=username
        c.Store_object=True
        c.Pandas = True
        c.Hide_output=True
        twint.run.Search(c)
        
        df = twint.storage.panda.Tweets_df
        
        
        tweets=pd.DataFrame(df['tweet'])

        def remove_content(text):
            text = re.sub(r"http\S+", "", text) #remove urls
            text=re.sub(r'\S+\.com\S+','',text) #remove urls
            text=re.sub(r'\@\w+','',text) #remove mentions
            text =re.sub(r'\#\w+','',text) #remove hashtags
            return text
        def process_text(text, stem=False): #clean text
            text=remove_content(text)
            text = re.sub('[^A-Za-z]', ' ', text.lower()) #remove non-alphabets
            tokenized_text = text.split()
            stop_words =  set(stopwords)
            clean_text = [
                 word for word in tokenized_text
                 if word not in stop_words
            ]
            if stem:
                clean_text=[stemmer.stem(word) for word in clean_text]
            return ' '.join(clean_text)

        df['cleaned_tweets']=df['tweet'].apply(lambda x: process_text(x))
        df['tweet']=df['tweet'].apply(lambda x: remove_content(x))


        emp=[]
        full=[]
        for i in df['reply_to'].values:
            if len(i)==0:
                emp.append(i)
            if len(i)>0:
                full.append(i)

        no_rep=len(full)
        no_tweet=len(emp)

        ta=pd.DataFrame([no_rep,no_tweet],index=['Replies','Tweets'])
        labels=['Replies','Tweets']
        fig01=px.pie(ta,values=0,labels=labels,names=['Replies','Tweets'],
                      hole=0.5,title='Tweet vs Replies', width=550, height=450.5)
        st.plotly_chart(fig01)


        if no_rep>no_tweet:
            st.write('This person definitely loves gist')
        else:
            st.write('This is my space to share myself')

        from wordcloud import WordCloud, STOPWORDS

        temp=' '.join(df['cleaned_tweets'].tolist())
        mask = np.array(Image.open("/Users/datalab/Documents/personal/twii2.jpeg"))
        wordcloud = WordCloud( background_color="black", max_words=1000, max_font_size = 20,mask=mask).generate(temp)

        
        plt.figure(figsize=(10,10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title('Cloud Of Frequently Used Words')
    
        

        st.pyplot(plt)

        
        st.subheader("Sentiment Spread")
        polarity= []

        for i in df['cleaned_tweets']:
            polarity.append(TextBlob(i).sentiment.polarity)
        df['Polarity']=polarity
        a = np.array(df['Polarity'])

        y=sum(np.array(a)>0)
        z=sum(np.array(a)<0)
        x=sum(np.array(a)==0)
        p=z+y+x
        w=pd.DataFrame([x,y,z],index=['Positive','Negative','Neutral'])

        labels=['Positive', 'Negative','Neutral']
        fig0=px.pie(w,values=0,labels=labels,names=['Positive', 'Negative','Neutral'],
                      hole=0.5,width=550, height=450.5)
        st.plotly_chart(fig0)
        
        st.subheader("Average Number Of Tweets By Time Of Day")
        tweet_time=df.groupby('hour')['tweet'].count().to_frame()
        tweet_time['percent']=round((tweet_time.tweet/tweet_time.tweet.sum())*100,2)
        tweet_time=tweet_time.reset_index()
        fig = px.line(tweet_time, x='hour', y='tweet')
        fig.update_layout(xaxis=dict(showgrid=False),
              yaxis=dict(showgrid=False))
        fig.update_yaxes(visible=False, showticklabels=False)
        st.plotly_chart(fig)

        st.subheader("Here Are These Tweep's Most Frequent Gist Partners, Know any of them? ")
        reply=[]
        for e in full:
            for i in range(len(e)):
                u=(e[i].get('screen_name'))
                reply.append(u)
        rp=pd.DataFrame({'Reply':reply})
        rp=rp.value_counts().to_frame()
        rp=rp.reset_index()
        rp.columns=['name','reply']
        rp=rp.head(10)
        for i in rp['name']:
            st.write(i)
        #rp['percent']=(rp['reply']/rp['reply'].sum())*100
        #rp=rp.head(10)
        #fig2 = px.bar(rp, y='name', x='percent', title='Preffered Gist Partners (%)',text_auto='.2s',orientation='h')
        #fig2.update_traces(textfont_size=12, textposition="outside", cliponaxis=False,)
        #fig2.update_layout(xaxis=dict(showgrid=False),
              #yaxis=dict(showgrid=False))

        #st.plotly_chart(fig2)

        loaded_model = tf.keras.models.load_model('/Users/datalab/Documents/personal/tw_model')

        pkl_file = open('/Users/datalab/Documents/personal/encoder.pkl', 'rb')
        le = pickle.load(pkl_file) 
        pkl_file.close()
        predictions =loaded_model.predict(df['cleaned_tweets'])
        predict_results = predictions.argmax(axis=1)
        df['topic']=le.inverse_transform(predict_results)
        topics=df['topic'].value_counts().to_frame()
        topics=topics.head(5)
        topics=topics.reset_index()
        topics.columns=['topic','count']

        topics['percent']=topics['count'].apply(lambda x:(x/topics['count'].sum())*100)
        fig3 = px.bar(topics, x='topic', y='percent',title='Most relatable topics (%)', text_auto='.2s')
        fig3.update_layout(xaxis=dict(showgrid=False),
              yaxis=dict(showgrid=False))
        st.plotly_chart(fig3)
        lik=int(df['nlikes'].mean())
        st.write(f'Average number of likes:{lik}')
#         if output_csv == 'Yes':
#             st.markdown(get_csv_download_link(data, file_name), unsafe_allow_html=True)
