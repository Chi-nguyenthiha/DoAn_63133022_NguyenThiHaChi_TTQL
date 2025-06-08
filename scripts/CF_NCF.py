from flask import Flask, render_template, request
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sys

sys.path.append('/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/scripts/ncf-utils')
from utils import Utils, EarlyStopping, cols_dict
from model import NCF

app = Flask(__name__)

# Load dataset
ratings_data_path = '/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/ratings.dat'
users_data_path = '/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/users.dat'
items_data_path = '/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/movies.dat'

ratings_data = pd.read_csv(ratings_data_path, sep='::', names=cols_dict['ratings'], engine='python')
users_data = pd.read_csv(users_data_path, sep='::', names=cols_dict['users'], engine='python')
items_data = pd.read_csv(items_data_path, sep='::', names=cols_dict['items'], encoding='latin-1', engine='python')

users_data_og = users_data.copy()
items_data_og = items_data.copy()
ratings_data_og = ratings_data.copy()

# Preprocessing
users_data = Utils.one_hot_encode(users_data, ['occupation', 'gender', 'age'])
items_data = Utils.multi_hot_encode(items_data, 'genre')
users_data = Utils.extract_category_avg_ratings(users_data, items_data, ratings_data)
items_data = Utils.extract_year(items_data)
users_data = Utils.move_column(users_data, ['gender_M', 'gender_F'], 0)
users_data, items_data = Utils.extend_users_items(users_data, items_data, ratings_data)
ratings_data = ratings_data.drop(['timestamp'], axis=1)
users_data = users_data.drop(['user_id', 'zip_code'], axis=1)
items_data = items_data.drop(['movie_id', 'title'], axis=1)
ratings_data['rating'] = ratings_data['rating'] / 5.0
items_data['year'] = items_data['year'] / items_data['year'].max()
users_data.iloc[:, -18:] = users_data.iloc[:, -18:] / users_data.iloc[:, -18:].max().max()

# Split
X_users_train, X_users_test = train_test_split(users_data, test_size=0.2, random_state=42)
X_users_val, X_users_test = train_test_split(X_users_test, test_size=0.5, random_state=42)
X_items_train, X_items_test = train_test_split(items_data, test_size=0.2, random_state=42)
X_items_val, X_items_test = train_test_split(X_items_test, test_size=0.5, random_state=42)
y_ratings_train, y_ratings_test = train_test_split(ratings_data, test_size=0.2, random_state=42)
y_ratings_val, y_ratings_test = train_test_split(y_ratings_test, test_size=0.5, random_state=42)

X_users_train, X_users_val, X_users_test = X_users_train.values, X_users_val.values, X_users_test.values
X_items_train, X_items_val, X_items_test = X_items_train.values, X_items_val.values, X_items_test.values
y_ratings_train, y_ratings_val, y_ratings_test = y_ratings_train.values, y_ratings_val.values, y_ratings_test.values

# Model init
user_dim = users_data.shape[1]
item_dim = items_data.shape[1]
num_users = users_data_og['user_id'].max()
num_items = items_data_og['movie_id'].max()

model = NCF(
    num_users=num_users,
    num_items=num_items,
    user_dim=user_dim,
    item_dim=item_dim,
    num_factors=32,
    mode='explicit',
    criterion=torch.nn.MSELoss(),
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-5,
    verbose=True,
    gpu=True
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/weightss/explicit.pth', map_location=device))
model.eval()
from sklearn.metrics.pairwise import cosine_similarity
def recommend_cf(user_id, ratings_data=ratings_data_og, top_k=10):
    """
    Gợi ý phim dựa trên Collaborative Filtering đơn giản (Item-based CF):
    - Tính điểm rating trung bình cho mỗi phim.
    - Lọc bỏ phim user đã xem.
    - Trả về top_k phim có điểm rating trung bình cao nhất.

    Args:
        user_id (int): ID người dùng.
        ratings_data (DataFrame): Dữ liệu rating gốc.
        top_k (int): Số lượng phim đề xuất.

    Returns:
        DataFrame gồm 2 cột: ['movie_id', 'pred_rating'] sắp xếp theo pred_rating giảm dần.
    """
    # Lấy phim user đã xem
    watched_movies = ratings_data[ratings_data['user_id'] == user_id]['movie_id'].unique()

    # Tính điểm rating trung bình cho mỗi phim
    avg_ratings = ratings_data.groupby('movie_id')['rating'].mean().reset_index()

    # Lọc bỏ phim user đã xem
    recs = avg_ratings[~avg_ratings['movie_id'].isin(watched_movies)]

    # Sắp xếp theo điểm rating trung bình giảm dần
    recs = recs.sort_values(by='rating', ascending=False).head(top_k)
    recs.rename(columns={'rating': 'pred_rating'}, inplace=True)
    return recs.reset_index(drop=True)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model(true_ratings, pred_ratings):
    """
    Tính toán RMSE và MAE giữa rating thật và rating dự đoán.

    Args:
        true_ratings (np.array): Mảng rating thật.
        pred_ratings (np.array): Mảng rating dự đoán.

    Returns:
        tuple (rmse, mae)
    """
    rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
    mae = mean_absolute_error(true_ratings, pred_ratings)
    return rmse, mae
def get_true_ratings(user_id, movie_ids):
    """
    Lấy rating thật của user cho các movie_id đã cho.

    Args:
        user_id (int): ID người dùng.
        movie_ids (list or np.array): Danh sách movie_id.

    Returns:
        np.array rating thật, nếu user chưa đánh giá movie thì trả NaN.
    """
    ratings = ratings_data_og[(ratings_data_og['user_id'] == user_id) & (ratings_data_og['movie_id'].isin(movie_ids))]
    movie_to_rating = dict(zip(ratings['movie_id'], ratings['rating']))
    return np.array([movie_to_rating.get(mid, np.nan) for mid in movie_ids])
def recommend_movies_for_existing_user(user_id_int, model, users_data, items_data_og, ratings_data_og, top_k=10, k_retrieve=300):
    import torch

    old_user = {'id': user_id_int}

    user_id, user_tensor, _, _ = Utils.preprocess_user(
        user=old_user,
        num_items=items_data_og.shape[0],
        users=users_data.drop_duplicates(inplace=False).values,
        topk=3,
        verbose=False
    )

    items_tensor = Utils.preprocess_items(items_data_og)

    user_id, user_tensor = user_id.to(model.device), user_tensor.to(model.device)

    movies = Utils.retrieve(
        movies=items_tensor,
        user=user_tensor.detach().cpu().numpy(),
        k=k_retrieve,
        random_state=0
    )

    movie_ids, movies = Utils.filter(
        movies=movies,
        ratings=ratings_data_og,
        user_id=user_id_int
    )
    movie_ids, movies = movie_ids.to(model.device), movies.to(model.device)

    y_pred = model(
        user_id[:len(movies)],
        movie_ids,
        user_tensor[:len(movies)],
        movies
    ).cpu().detach().numpy()

    movies_retrieved = items_data_og[items_data_og['movie_id'].isin(movie_ids.cpu().numpy())]
    movies_retrieved = movies_retrieved.sort_values(
        by='movie_id',
        key=lambda x: pd.Categorical(x, categories=movie_ids.cpu().numpy(), ordered=True)
    )

    top_recommendations = Utils.order(y_pred, movies_retrieved, mode='explicit', top_k=top_k)

    # Rename cột nếu cần
    if 'pred_rating' not in top_recommendations.columns:
        for col in top_recommendations.columns:
            if col != 'movie_id':
                top_recommendations = top_recommendations.rename(columns={col: 'pred_rating'})
                break


    return top_recommendations


@app.route('/compare', methods=['GET', 'POST'])
def compare_models():
    if request.method == 'POST':
        try:
            user_id = int(request.form['user_id'])

            # CF recommend
            cf_recs = recommend_cf(user_id)
            cf_pred = cf_recs['pred_rating'].values
            cf_true = get_true_ratings(user_id, cf_recs['movie_id'].values)

            # NCF recommend
            ncf_recs = recommend_movies_for_existing_user(
                user_id, model, users_data, items_data_og, ratings_data_og, top_k=10)
            ncf_pred = ncf_recs['pred_rating'].values
            ncf_true = get_true_ratings(user_id, ncf_recs['movie_id'].values)

            # Lọc bỏ rating nan (không có rating thật)
            mask_cf = ~np.isnan(cf_true)
            mask_ncf = ~np.isnan(ncf_true)

            cf_rmse, cf_mae = evaluate_model(cf_true[mask_cf], cf_pred[mask_cf]) if mask_cf.any() else (None, None)
            ncf_rmse, ncf_mae = evaluate_model(ncf_true[mask_ncf], ncf_pred[mask_ncf]) if mask_ncf.any() else (None, None)

            return render_template('compare_result.html',
                                   user_id=user_id,
                                   cf_recommendations=cf_recs.to_html(classes='data', index=False),
                                   ncf_recommendations=ncf_recs.to_html(classes='data', index=False),
                                   cf_rmse=cf_rmse, cf_mae=cf_mae,
                                   ncf_rmse=ncf_rmse, ncf_mae=ncf_mae)
        except Exception as e:
            return f"Lỗi: {str(e)}"
    return render_template('compare_form.html')



if __name__ == '__main__':
    app.run(debug=True)