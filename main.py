import numpy as np
import pandas as pd

def get_cosine(item_data):
    num_movies = item_data.shape[1]
    cosine_matrix = np.zeros((num_movies, num_movies), dtype=np.double)


    for i in range(num_movies):
        for j in range(num_movies):
            movie_pair = item_data.iloc[:,[i,j]]
            movie_pair = movie_pair.dropna()
            movie_i = movie_pair.iloc[:,0]
            movie_j = movie_pair.iloc[:, 1]
            cosine_matrix[i,j] = np.dot(movie_i, movie_j)/(np.linalg.norm(movie_i)*np.linalg.norm(movie_j))


    return cosine_matrix

df = pd.read_csv("MovieRecommender.csv")
item_data = df.iloc[:,1:].copy()
cm = get_cosine(item_data)
pd.DataFrame(np.round(cm, decimals=3)).to_csv("similar.csv")
num_movies = item_data.shape[1]

X= df.iloc[:,1:].to_numpy()

for user in range(X.shape[0]):
    has_rating=~np.isnan(X[user])
    for movie in range(X.shape[1]):
        if not has_rating[movie]:
            X[user, movie] = np.dot(X[user, has_rating], cm[movie, has_rating])/np.sum(np.abs(cm[movie, has_rating]))
    X[user, has_rating]= -10.0

#make code values 1-2 = -1 and 3-5 = 1

df.iloc[:,1:] = np.round(X, decimals= 3)
df.to_csv("recommend.csv")
