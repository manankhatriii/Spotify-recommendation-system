import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

features = ['valence', 'year', 'acousticness', 'danceability', 'energy', 'explicit', 
            'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 
            'speechiness', 'tempo', 'seconds']

def find_similar_songs(user_input, df, features, scaler, pca, kmeans):
    scaled_input = scaler.transform([list(user_input.values())])
    pca_input = pca.transform(scaled_input)
    
    cluster = kmeans.predict(pca_input)[0]
    st.write(f'This song belongs to cluster: {cluster}')
    
    cluster_songs = df[df['cluster'] == cluster]
    
    similarity = cosine_similarity(cluster_songs[features], np.array([list(user_input.values())]))
    cluster_songs = cluster_songs.copy()
    cluster_songs['similarity'] = similarity[:, 0]
    
    top_5_songs = cluster_songs.nlargest(5, 'similarity')
    return top_5_songs[['similarity', 'id', 'name', 'artists'] + features]

df = pd.read_csv('data.csv')
df['seconds'] = df['duration_ms'] / 1000
df['decade'] = (df['year'] // 10) * 10

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

pca = PCA(n_components=3)
pca_features = pca.fit_transform(scaled_features)

kmeans = KMeans(n_clusters=11, random_state=42)
df['cluster'] = kmeans.fit_predict(pca_features)

st.title("Song Recommendation System")

song_name = st.text_input("Enter the name of a song:")

if song_name:
    song_name_lower = song_name.lower()
    matching_song = df[df['name'].str.lower() == song_name_lower]
    
    if matching_song.empty:
        st.write("Song not found in the dataset.")
    else:
        user_input = matching_song.iloc[0][features].to_dict()
        
        similar_songs = find_similar_songs(user_input, df, features, scaler, pca, kmeans)

        st.write("Top 5 recommended songs based on your input:")
        st.dataframe(similar_songs)