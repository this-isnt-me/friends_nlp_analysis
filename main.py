# Import Python Libraries

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
from PIL import Image

# SET WIDTH AS WIDE
st.set_page_config(layout="wide")


# Declare Functions

@st.cache_data
def load_data(input_url):
    data = pd.read_csv(input_url)
    data['lda_topic'] = data['lda_topic'].astype(str)
    data['nmf_topic'] = data['nmf_topic'].astype(str)
    return data


# possible datasource urls
friends_url = r'data\friends_ross_rachael_dataset_processed_sentiment.csv'

# Load in
loaded_data = load_data(friends_url)

# Main Page Headers
st.title("Analysis of Ross and Rachel's Relationship")
st.markdown(
    '### _This Application is a Dashboard for the Analysis of Verbal Exchange\''
    's between Ross and Rachael in Friends  S1 to 10_')
st.write("---")
st.markdown('###')

# Sidebar Headers
image_file_path = r'gallery/deepminer_logo.png'
image_inv_file_path = r'gallery/deepminer_logo_inverse.png'
logo = Image.open(image_inv_file_path)
st.sidebar.image(logo, width=250)
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown('### _These controls allow you to adjust the outputs_')

# Random Tweet Form
form1 = st.sidebar.form(key="1")
form1.subheader("Display Random Speech")
random_text = form1.radio('Sentiment', ('STRONGLY POSITIVE', 'POSITIVE', 'NEUTRAL', 'NEGATIVE', 'STRONGLY NEGATIVE'))
# form1.markdown(loaded_data.query('sentiment_class == @random_text')[['text']].sample(n=1).iat[0, 0])
form1.form_submit_button("Display Random Text")

st.markdown(f"### Random Text - {random_text}")
st.markdown(loaded_data.query('sentiment_class == @random_text')[['text']].sample(n=1).iat[0, 0])
st.write("---")

# Tweets by Sentiment Form
form2 = st.sidebar.form(key="2")
form2.markdown('### Number of Texts by Sentiment - Total')
select = form2.selectbox('Visualisation type', ['Histogram', 'Pie Chart'], key=1)
sentiment_count = loaded_data['sentiment_class'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment': sentiment_count.index, 'Text': sentiment_count.values})
form2.form_submit_button("Update Total Sentiment Chart")

if not st.sidebar.checkbox("Hide Total Chart", True, key='cb1'):
    st.markdown('### Number of Texts by Sentiment')
    if select == 'Histogram':
        fig = px.bar(sentiment_count, y='Sentiment', x='Text',
                     color='Sentiment', height=500, width=1200, orientation="h",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig)
    elif select == 'Pie Chart':
        fig = px.pie(sentiment_count, values='Text', names='Sentiment',  height=800, width=800,
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, theme=None)
    st.write("---")


form3 = st.sidebar.form(key="3")
form3.markdown('### Number of Tweets by Sentiment - Character')
choice = form3.multiselect('Select Character', ('Rachel Green', 'Ross Geller'), key=2)
form3.form_submit_button("Sentiment by Character Chart")
if not st.sidebar.checkbox("Hide By Character Chart", True, key='cb2') and len(choice) > 0:
    st.markdown('### Number of Conversations by Sentiment and Character')
    choice_data = loaded_data[loaded_data.character.isin(choice)]
    fig_choice = px.histogram(choice_data, x='character', y='sentiment_class', histfunc='count',
                              color='character', facet_col='sentiment_class',
                              labels={'character': ""}, height=600, width=1000,
                              color_discrete_sequence=px.colors.sequential.Viridis)
    fig_choice.for_each_annotation(lambda a: a.update(text=a.text.replace("sentiment_class=", "")))
    # fig_choice.for_each_annotation(lambda a: a.update(text=a.text.replace("sentiment_class=", "")))
    st.plotly_chart(fig_choice)
    st.write("---")

# Tweets by Sentiment Form LDA
form4 = st.sidebar.form(key="4")
form4.markdown('### Number of Conversations by LDA Topic')
lda_select = form4.selectbox('Visualisation type', ['Histogram', 'Pie Chart'], key=3)
lda_count = loaded_data['lda_topic'].value_counts()
lda_count = pd.DataFrame({'Topic': lda_count.index, 'Count': lda_count.values})
lda_count = lda_count.sort_values(by='Topic', ascending=True)
form4.form_submit_button("Update LDA Topic Chart")

if not st.sidebar.checkbox("Hide LDA Chart", True, key='cb3'):
    st.markdown('### Number of Conversations by LDA Topic')
    if lda_select == 'Histogram':
        fig = px.bar(lda_count, y='Topic', x='Count',
                     color='Topic', height=500, width=1200, orientation="h",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig)
    elif lda_select == 'Pie Chart':
        fig = px.pie(lda_count, values='Topic', names='Count',  height=800, width=800,
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, theme=None)
    lda_df = loaded_data[["lda_topic", "lda_keywords"]]
    lda_df = lda_df.sort_values("lda_topic").drop_duplicates()
    lda_topic_list = list(lda_df["lda_topic"])
    lda_keyword_list = list(lda_df["lda_keywords"])
    for index, entry in enumerate(lda_topic_list):
        st.markdown(f'##### Topic Number: {entry} - Keywords: {lda_keyword_list[index]}')
    st.write("---")

# Tweets by Sentiment Form NMF
form5 = st.sidebar.form(key="6")
form5.markdown('### Number of Conversations by NMF Topic')
nmf_select = form5.selectbox('Visualisation type', ['Histogram', 'Pie Chart'], key=4)
nmf_count = loaded_data['nmf_topic'].value_counts()
nmf_count = pd.DataFrame({'Topic': nmf_count.index, 'Count': nmf_count.values})
nmf_count = nmf_count.sort_values(by='Topic', ascending=True)
form5.form_submit_button("Update NMF Topic Chart")

if not st.sidebar.checkbox("Hide NMF Chart", True, key='cb4'):
    st.markdown('### Number of Conversations by NMF Topic')
    if nmf_select == 'Histogram':
        fig = px.bar(nmf_count, y='Topic', x='Count',
                     color='Topic', height=500, width=1200, orientation="h",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig)
    elif nmf_select == 'Pie Chart':
        fig = px.pie(nmf_count, values='Topic', names='Count',  height=800, width=800,
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, theme=None)
    nmf_df = loaded_data[["nmf_topic", "nmf_keywords"]]
    nmf_df = nmf_df.sort_values("nmf_topic").drop_duplicates()
    nmf_topic_list = list(nmf_df["nmf_topic"])
    nmf_keyword_list = list(nmf_df["nmf_keywords"])
    for index, entry in enumerate(nmf_topic_list):
        st.markdown(f'##### Topic Number: {entry} - Keywords: {nmf_keyword_list[index]}')
    st.write("---")

# Sentiment by Season
st.sidebar.markdown('### Sentiment over Time')
form6 = st.sidebar.form(key="7")
form6.markdown('### Select Season')
sent_season = form6.selectbox('Season Number', ['All', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key=7)
form6.form_submit_button("Update Season")

if not st.sidebar.checkbox("Hide Sentiment over Time Chart", True, key='cb5'):
    st.markdown('### Sentiment Analysis over Time')
    if sent_season == 'All':
        fig = px.line(loaded_data, y='sentiment', x='scene_inc', markers=True,
                     color='character', height=1200, width=1200,
                     color_discrete_sequence=['#238a8d', '#fde725'],
                     title="All Seasons",
                     labels={
                         "sentiment": "sentiment (negative -1 to positive 1)",
                         "scene_inc" : "Interaction number"
                     })
        st.plotly_chart(fig)
    else:
        sub_dataframe = loaded_data[loaded_data["season"] == sent_season]
        fig = px.line(sub_dataframe, y='sentiment', x='scene_inc', markers=True,
                     color='character', height=1200, width=1200,
                     color_discrete_sequence=['#238a8d', '#fde725'],
                     title=f"Season {sent_season}",
                     labels={
                         "sentiment": "sentiment (negative -1 to positive 1)",
                         "scene_inc" : "Interaction number"
                     })
        st.plotly_chart(fig)
    st.write("---")

# Objectivity by Season
form7 = st.sidebar.form(key="8")
form7.markdown('### Select Season')
obj_season = form7.selectbox('Season Number', ['All', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key=8)
form7.form_submit_button("Update Season")

if not st.sidebar.checkbox("Hide Objectivity over Time Chart", True, key='cb8'):
    st.markdown('### Objectivity Analysis over Time')
    if obj_season == 'All':
        fig = px.line(loaded_data, y='objectivity', x='scene_inc', markers=True,
                     color='character', height=1200, width=1200,
                     color_discrete_sequence=['#238a8d', '#fde725'],
                     title="All Seasons",
                     labels={
                         "objectivity": "objectivity (objective 0 to subjective 1)",
                         "scene_inc" : "Interaction number"
                     })
        st.plotly_chart(fig)
    else:
        sub_dataframe = loaded_data[loaded_data["season"] == obj_season]
        fig = px.line(sub_dataframe, y='objectivity', x='scene_inc', markers=True,
                     color='character', height=1200, width=1200,
                     color_discrete_sequence=['#238a8d', '#fde725'],
                     title=f"Season {obj_season}",
                     labels={
                         "objectivity": "objectivity (objective 0 to subjective 1)",
                         "scene_inc" : "Interaction number"
                     })
        st.plotly_chart(fig)
    st.write("---")

### UMAP Plotting
st.sidebar.markdown('### 3d Scatter Chart Word Sentence Vectors')
form8 = st.sidebar.form(key="9")
form8.markdown('### Select Season')
umap_season = form8.selectbox('Season Number', ['All', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key=9)
umap_color_season = form8.selectbox('Select Color Filter Option', ['Character', 'Sentiment Class', 'Objectivity Class', 'LDA Topic', 'NMF Topic'], key=10)
form8.form_submit_button("Update Season")

if not st.sidebar.checkbox("Hide 3d Scatter Chart", True, key='cb7'):
    map_option_dict = {
        'Character' : 'character',
        'Sentiment Class' : 'sentiment_class',
        'Objectivity Class' : 'objectivity_class',
        'LDA Topic' : 'lda_topic',
        'NMF Topic' : 'nmf_topic'
    }
    color_option_dict = {
        'Character' : ['#3B528B', '#FDE725'],
        'Sentiment Class' : ['#440154', '#3B528B', '#24868E', '#35B779', '#FDE725'],
        'Objectivity Class' : ['#440154', '#3B528B', '#24868E', '#35B779', '#FDE725'],
        'LDA Topic' : ['#440154', '#472D7B', '#3B528B', '#31688E', '#24868E', '#1F9A8A', '#35B779', '#75D054',
                      '#AADC32', '#FDE725'],
        'NMF Topic' : ['#440154', '#472D7B', '#3B528B', '#31688E', '#24868E', '#1F9A8A', '#35B779', '#75D054',
                      '#AADC32', '#FDE725']
    }
    color_split = map_option_dict[umap_color_season]
    color_sequence = color_option_dict[umap_color_season]
    st.markdown('### 3d Scatter Chart Word Sentence Vectors')
    if umap_season == 'All':
        fig = px.scatter_3d(loaded_data, x='x_coord', y='y_coord', z='z_coord',
                            color=color_split, height=1200, width=1200,
                            title="All Seasons",
                            color_discrete_sequence=color_sequence)
        st.plotly_chart(fig)
    else:
        sub_dataframe = loaded_data[loaded_data["season"] == umap_season]
        fig = px.scatter_3d(sub_dataframe, x='x_coord', y='y_coord', z='z_coord',
                            color=color_split, height=1200, width=1200,
                            title=f"Season {umap_season}",
                            color_discrete_sequence=color_sequence)
        st.plotly_chart(fig)
    st.write("---")