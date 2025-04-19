from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# Tạo Flask app
app = Flask(__name__)

# Load model và mapping
model = load_model("/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/notebooks/ncf_model.h5")

with open("/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/notebooks/user_mapping.pkl", "rb") as f:
    user_mapping = pickle.load(f)

with open("/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/notebooks/movie_mapping.pkl", "rb") as f:
    movie_mapping = pickle.load(f)

# Đọc movies.dat
movies = pd.read_csv("/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/movies.dat", sep="::", engine="python", 
                     names=["movieId", "title", "genres"], encoding="latin1")

# Đọc ratings.dat và đổi tên cột MovieID -> ItemID
ratings_1m = pd.read_csv("/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/ratings.dat", sep="::", engine="python", 
                      names=['userId', 'movieId', 'rating', 'timestamp'])

reverse_movie_mapping = {v: k for k, v in movie_mapping.items()}
user_embeddings = model.get_layer(index=2).get_weights()[0]
movie_embeddings = model.get_layer(index=3).get_weights()[0]

# Tạo ma trận người dùng - phim
user_item_matrix = ratings_1m.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# Tính toán độ tương đồng giữa các người dùng
user_similarity = cosine_similarity(user_item_matrix)

def collaborative_filtering(user_id, top_k=10):
    user_idx = user_id - 1  # vì chỉ số bắt đầu từ 0 trong ma trận
    similar_users = user_similarity[user_idx]
    
    # Tìm ra top_k người dùng tương đồng nhất
    similar_users_idx = np.argsort(similar_users)[::-1][1:top_k+1]
    
    # Dự đoán các phim mà người dùng có thể thích dựa trên các người dùng tương tự
    recommended_movies = []
    for idx in similar_users_idx:
        similar_user_ratings = user_item_matrix.iloc[idx]
        recommended_movies.extend(similar_user_ratings[similar_user_ratings > 0].index)
    
    recommended_movies = list(set(recommended_movies))
    
    # Lọc ra top_k phim
    return movies[movies["movieId"].isin(recommended_movies[:top_k])]

# Hàm gợi ý cho user mới
def recommend_for_new_user(user_ratings_100k, top_k=10):
    mapped_movies = []
    for movie_id, rating in user_ratings_100k:
        if movie_id in movie_mapping:
            mapped_id = movie_mapping[movie_id]
            mapped_movies.append((mapped_id, rating))

    if not mapped_movies:
        return []

    movie_ids, ratings = zip(*mapped_movies)
    movie_vecs = movie_embeddings[list(movie_ids)]
    avg_embedding = np.average(movie_vecs, axis=0)

    similarities = cosine_similarity([avg_embedding], user_embeddings)[0]
    nearest_user_idx = np.argmax(similarities)

    return recommend_movies_by_index(nearest_user_idx, top_k)

# Hàm recommend cho user đã có ID trong 1M
def recommend_movies_by_index(mapped_user_id, top_k=10):
    num_movies = len(movie_mapping)
    all_movie_ids = np.arange(num_movies)

    user_input = np.full(len(all_movie_ids), mapped_user_id).reshape(-1, 1)
    movie_input = all_movie_ids.reshape(-1, 1)

    predicted_ratings = model.predict([user_input, movie_input], verbose=0)
    top_indices = predicted_ratings.flatten().argsort()[-top_k:][::-1]
    top_movie_ids = movie_input[top_indices].flatten()

    top_movie_ids_original = [reverse_movie_mapping[mid] for mid in top_movie_ids]
    recommended_movies = movies[movies["movieId"].isin(top_movie_ids_original)]

    return recommended_movies[["title", "genres"]]

# Định nghĩa route cho trang chính
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    if request.method == "POST":
        user_id = int(request.form.get("user_id"))

        # Lấy lịch sử rating của user từ MovieLens 1M
        user_rated = ratings_1m[ratings_1m["userId"] == user_id][["movieId", "rating"]]
        rated_movies = list(user_rated.itertuples(index=False, name=None))

        if rated_movies:
            # Sử dụng NCF để gợi ý
            recommendations = recommend_for_new_user(rated_movies, top_k=10)
            recommendations = recommendations.to_dict(orient="records")
        else:
            # Nếu không có rating, sử dụng CF
            recommendations = collaborative_filtering(user_id, top_k=10)
            recommendations = recommendations.to_dict(orient="records")

    return render_template("index.html", recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)
