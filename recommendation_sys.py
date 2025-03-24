import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def format_genres(genre):
    return genre.replace("|", " ").replace("-", "")
data = pd.read_csv("movies.csv", encoding="latin-1", sep="\t", usecols=["title", "genres"])
data['genres'] = data['genres'].apply(format_genres)
# or use lambda data['genres'] = data['genres'].apply(lambda genre: genre.replace("|", " "))
# how can we compare the relation among genres, more or less?
# digitalize them, tf/idf
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["genres"])
tfidf_dense = pd.DataFrame(tfidf_matrix.todense(), index=data['title'], columns=vectorizer.get_feature_names_out())
# why in this, we don't split train and test set ? => because we don't have any labels
# print(vectorizer.vocabulary_)
# print(tfidf_matrix.shape)

#{'animation': 2, 'children': 3, 'comedy': 4, 'adventure': 1, 'fantasy': 8, 'romance': 15, 'drama': 7, 'action': 0,
# 'crime': 5, 'thriller': 17, 'horror': 11, 'sci': 16, 'fi': 9, 'documentary': 6, 'war': 18,
# 'musical': 12, 'mystery': 13, 'film': 10, 'noir': 14, 'western': 19}
# error is 'sci': 16, 'fi': 9, sci-fi is split into 2 types  : .replace("-", "")
# (3883, 20) => (3883, 18)

# cosine similarity, to find the relation between two vectors
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_dense = pd.DataFrame(cosine_sim, index=data['title'], columns=data['title'])
# print(cosine_sim.shape)

input_movie = "Jumanji (1995)"
top_k = 20
result = cosine_dense.loc[input_movie, :]
result = result.sort_values(ascending=False)[:top_k].to_frame(name="score").reset_index()
