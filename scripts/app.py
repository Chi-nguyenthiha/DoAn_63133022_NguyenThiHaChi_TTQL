# movie_app.py
# File xây dựng ứng dụng Flask để phục vụ mô hình NCF và CF
from flask import Flask, render_template, request, redirect, url_for, session, flash
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
import sys
from sklearn.metrics.pairwise import cosine_similarity

# Import model and utility
sys.path.append('/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/scripts/ncf-utils')
from utils import Utils, EarlyStopping, cols_dict
from model import NCF
import time

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # để dùng session


# Đường dẫn dữ liệu #đổi đường dẫn này theo thư mục của bạn
ratings_data_path = '/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/ratings.dat'
users_data_path = '/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/users.dat'
items_data_path = '/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/movies.dat'

ratings_data = pd.read_csv(ratings_data_path, sep='::', names=cols_dict['ratings'], engine='python')
users_data = pd.read_csv(users_data_path, sep='::', names=cols_dict['users'], engine='python')
items_data = pd.read_csv(items_data_path, sep='::', names=cols_dict['items'], encoding='latin-1', engine='python')

users_data_og = users_data.copy()
items_data_og = items_data.copy()
ratings_data_og = ratings_data.copy()

# --- Pivot ratings matrix ---
ratings_matrix = ratings_data.pivot(index='user_id', columns='movie_id', values='rating')

# --- Fill NA with 0 (for similarity calc) ---
rating_matrix_filled = ratings_matrix.fillna(0)

# --- Compute similarity matrices ---
user_similarity = cosine_similarity(rating_matrix_filled)
item_similarity = cosine_similarity(rating_matrix_filled.T)

user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix_filled.index, columns=rating_matrix_filled.index)
item_similarity_df = pd.DataFrame(item_similarity, index=rating_matrix_filled.columns, columns=rating_matrix_filled.columns)

# --- Split into train, val, test ---
train_data, temp_data = train_test_split(ratings_data, test_size=0.2, random_state=42)         # 80% train, 20% temp
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42) #10% val + 10% test

# --- Pivot full matrix from train only ---
ratings_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating')
rating_matrix_filled = ratings_matrix.fillna(0)

# --- Recompute similarities using train set ---
user_similarity = cosine_similarity(rating_matrix_filled)
item_similarity = cosine_similarity(rating_matrix_filled.T)

user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix_filled.index, columns=rating_matrix_filled.index)
item_similarity_df = pd.DataFrame(item_similarity, index=rating_matrix_filled.columns, columns=rating_matrix_filled.columns)

# Mã hóa và tiền xử lý dữ liệu
users_data = Utils.one_hot_encode(users_data, ['occupation', 'gender', 'age'])
items_data = Utils.multi_hot_encode(items_data, 'genre')
users_data = Utils.extract_category_avg_ratings(users_data, items_data, ratings_data)
items_data = Utils.extract_year(items_data)
users_data = Utils.move_column(users_data, ['gender_M', 'gender_F'], 0)
users_data, items_data = Utils.extend_users_items(users_data, items_data, ratings_data)

# --- DROP UNUSED COLUMNS ---
ratings_data = ratings_data.drop(['timestamp'], axis=1)
users_data = users_data.drop(['user_id', 'zip_code'], axis=1)
items_data = items_data.drop(['movie_id', 'title'], axis=1)


# --- NORMALIZATION ---
ratings_data['rating'] = ratings_data['rating'] / 5.0
items_data['year'] = items_data['year'] / items_data['year'].max()
users_data.iloc[:, -18:] = users_data.iloc[:, -18:] / users_data.iloc[:, -18:].max().max()

# --- SPLIT DATA ---
X_users_train, X_users_test = train_test_split(users_data, test_size=0.2, random_state=42)
X_users_val, X_users_test = train_test_split(X_users_test, test_size=0.5, random_state=42)

X_items_train, X_items_test = train_test_split(items_data, test_size=0.2, random_state=42)
X_items_val, X_items_test = train_test_split(X_items_test, test_size=0.5, random_state=42)

y_ratings_train, y_ratings_test = train_test_split(ratings_data, test_size=0.2, random_state=42)
y_ratings_val, y_ratings_test = train_test_split(y_ratings_test, test_size=0.5, random_state=42)

# --- CONVERT TO NUMPY ---
X_users_train, X_users_val, X_users_test = X_users_train.values, X_users_val.values, X_users_test.values
X_items_train, X_items_val, X_items_test = X_items_train.values, X_items_val.values, X_items_test.values
y_ratings_train, y_ratings_val, y_ratings_test = y_ratings_train.values, y_ratings_val.values, y_ratings_test.values

# Khởi tạo mô hình NCF
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

# Tải mô hình đã huấn luyện
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/weightss/explicit.pth', map_location=device))
model.eval()
# NCF Predict function
def recommend_movies_for_existing_user(user_id_int, model, users_data, items_data_og, ratings_data_og, top_k=10, k_retrieve=300):
    """
    Trả về danh sách top_k bộ phim đề xuất cho người dùng đã tồn tại dựa trên lịch sử tương tác.
    
    Args:
        user_id_int (int): ID người dùng đã tồn tại trong tập dữ liệu.
        model (NCF): Mô hình đã huấn luyện.
        users_data (DataFrame): Dữ liệu người dùng đã xử lý.
        items_data_og (DataFrame): Dữ liệu phim gốc.
        ratings_data_og (DataFrame): Dữ liệu ratings gốc.
        top_k (int): Số lượng phim đề xuất.
        k_retrieve (int): Số lượng phim được lấy trong bước Retrieval.

    Returns:
        DataFrame: Danh sách phim đề xuất cùng điểm dự đoán.
    """
    import torch

    old_user = {'id': user_id_int}

    # Bước 1: Tiền xử lý user
    user_id, user_tensor, _, _ = Utils.preprocess_user(
        user=old_user,
        num_items=items_data_og.shape[0],
        users=users_data.drop_duplicates(inplace=False).values,
        topk=3,
        verbose=False
    )

    # Bước 2: Tiền xử lý item
    items_tensor = Utils.preprocess_items(items_data_og)

    # Chuyển tensor sang thiết bị phù hợp
    user_id, user_tensor = user_id.to(model.device), user_tensor.to(model.device)

    # Bước 3: Retrieval
    movies = Utils.retrieve(
        movies=items_tensor,
        user=user_tensor.detach().cpu().numpy(),
        k=k_retrieve,
        random_state=0
    )

    # Bước 4: Filter phim đã xem
    movie_ids, movies = Utils.filter(
        movies=movies,
        ratings=ratings_data_og,
        user_id=user_id_int
    )
    movie_ids, movies = movie_ids.to(model.device), movies.to(model.device)

    # Bước 5: Dự đoán rating
    y_pred = model(
        user_id[:len(movies)],
        movie_ids,
        user_tensor[:len(movies)],
        movies
    ).cpu().detach().numpy()
    y_pred = np.clip(np.round(y_pred * 5), 1, 5)  # Scale từ [0,1] về [1,5] và làm tròn
    # Bước 6: Sắp xếp phim
    movies_retrieved = items_data_og[items_data_og['movie_id'].isin(movie_ids.cpu().numpy())]
    movies_retrieved = movies_retrieved.sort_values(
        by='movie_id',
        key=lambda x: pd.Categorical(x, categories=movie_ids.cpu().numpy(), ordered=True)
    )

    top_recommendations = Utils.order(y_pred, movies_retrieved, mode='explicit', top_k=top_k)
    return top_recommendations

# --- CF Predict function ---
def predict_user_cf(user_id, item_id, k=20):
    if user_id not in user_similarity_df.index or item_id not in ratings_matrix.columns:
        return np.nan
    sim_users = user_similarity_df[user_id].drop(index=user_id).nlargest(k)
    ratings = ratings_matrix.loc[sim_users.index, item_id]
    weighted_ratings = sim_users * ratings.fillna(0)
    return weighted_ratings.sum() / (np.abs(sim_users[ratings.notna()]).sum() + 1e-8)

def predict_item_cf(user_id, item_id, k=20):
    if item_id not in item_similarity_df.index or user_id not in ratings_matrix.index:
        return np.nan
    sim_items = item_similarity_df[item_id].drop(index=item_id).nlargest(k)
    ratings = ratings_matrix.loc[user_id, sim_items.index]
    weighted_ratings = sim_items * ratings.fillna(0)
    return weighted_ratings.sum() / (np.abs(sim_items[ratings.notna()]).sum() + 1e-8)
def evaluate_ncf(model, user_ids, item_ids, true_ratings):
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        user_ids_tensor = torch.LongTensor(user_ids).to(model.device)
        item_ids_tensor = torch.LongTensor(item_ids).to(model.device)
        user_features = torch.tensor(users_data.values[user_ids], dtype=torch.float32).to(model.device)
        item_features = torch.tensor(items_data.values[item_ids], dtype=torch.float32).to(model.device)
        preds = model(user_ids_tensor, item_ids_tensor, user_features, item_features).cpu().numpy()
        preds = np.clip(np.round(preds * 5), 1, 5)  # Scale từ [0,1] về [1,5] và làm tròn
    end_time = time.time()
    mae = np.mean(np.abs(true_ratings - preds))
    rmse = np.sqrt(mean_squared_error(true_ratings, preds))
    latency = (end_time - start_time) / len(true_ratings)  # thời gian trung bình 1 dự đoán
    
    return mae, rmse, latency

def evaluate_cf(predict_function, data, top_n=100):
    y_true, y_pred = [], []
    start_time = time.time()

    for _, row in data.iterrows():
        uid, iid, true_rating = row['user_id'], row['movie_id'], row['rating']
        pred = predict_function(uid, iid)
        if not np.isnan(pred):
            y_true.append(true_rating)
            y_pred.append(pred)
        if len(y_true) >= top_n:
            break

    end_time = time.time()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    latency = (end_time - start_time) / len(y_true)
    return mae, rmse, latency


def user_cf_recommend(user_id, movies_df, k=20, top_n=10):
    """
    Trả về top_n phim gợi ý sử dụng User-based CF.
    """
    # Lấy danh sách tất cả movie_id đã xuất hiện trong tập train
    movie_ids = ratings_matrix.columns

    # Lọc ra các phim mà user này chưa đánh giá
    rated_movies = ratings_matrix.loc[user_id].dropna().index.tolist() if user_id in ratings_matrix.index else []
    candidate_movies = [movie_id for movie_id in movie_ids if movie_id not in rated_movies]

    # Dự đoán rating cho từng phim chưa xem
    predictions = []
    for item_id in candidate_movies:
        pred = predict_user_cf(user_id, item_id, k=k)
        if not np.isnan(pred):
            predictions.append((item_id, pred))

    # Sắp xếp theo rating dự đoán
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [movie_id for movie_id, _ in predictions[:top_n]]

    # Trả về thông tin phim
    return movies_df[movies_df['movie_id'].isin(top_movie_ids)]


def item_cf_recommend(user_id, movies_df, k=20, top_n=10):
    """
    Trả về top_n phim gợi ý sử dụng Item-based CF.
    """
    # Lấy danh sách tất cả movie_id đã xuất hiện trong tập train
    movie_ids = ratings_matrix.columns

    # Lọc ra các phim mà user này chưa đánh giá
    rated_movies = ratings_matrix.loc[user_id].dropna().index.tolist() if user_id in ratings_matrix.index else []
    candidate_movies = [movie_id for movie_id in movie_ids if movie_id not in rated_movies]

    # Dự đoán rating cho từng phim chưa xem
    predictions = []
    for item_id in candidate_movies:
        pred = predict_item_cf(user_id, item_id, k=k)
        if not np.isnan(pred):
            predictions.append((item_id, pred))

    # Sắp xếp theo rating dự đoán
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [movie_id for movie_id, _ in predictions[:top_n]]

    # Trả về thông tin phim
    return movies_df[movies_df['movie_id'].isin(top_movie_ids)]


# Fake "đăng nhập" chỉ bằng user_id
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        if user_id in ratings_matrix.index:
            session['user_id'] = user_id
            flash(f'Đăng nhập thành công với user_id: {user_id}')
            return redirect(url_for('home'))
        else:
            flash('User ID không tồn tại!')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Đã đăng xuất.')
    return redirect(url_for('login'))

# Trang chủ: Hiển thị phim gợi ý theo thể loại user đã xem, lựa chọn giữa NCF và CF
@app.route('/', methods=['GET', 'POST'])
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']

    method = request.args.get('method', 'ncf')  # 'ncf', 'usercf', 'itemcf'
    
    if method == 'ncf':
        recommended_movies = recommend_movies_for_existing_user(
            user_id, model, users_data, items_data_og, ratings_data_og, top_k=20
        )
        # Evaluate
        sample = test_data.sample(n=100, random_state=42)
        mae, rmse, latency = evaluate_ncf(
            model,
            sample['user_id'].values,
            sample['movie_id'].values,
            sample['rating'].values
        )
    elif method == 'usercf':
        recommended_movies = user_cf_recommend(user_id, items_data_og, k=20, top_n=20)
        mae, rmse, latency = evaluate_cf(predict_user_cf, test_data, top_n=100)
    else:  # 'itemcf'
        recommended_movies = item_cf_recommend(user_id, items_data_og, k=20, top_n=20)
        mae, rmse, latency = evaluate_cf(predict_item_cf, test_data, top_n=100)

    watched_movies_ids = ratings_data_og[ratings_data_og['user_id'] == user_id]['movie_id'].unique()
    watched_genres = items_data_og[items_data_og['movie_id'].isin(watched_movies_ids)]['genre'].str.cat(sep='|').split('|')
    from collections import Counter
    top_genres = [genre for genre, _ in Counter(watched_genres).most_common(5)]

    return render_template(
        'home.html',
        user_id=user_id,
        movies=recommended_movies.to_dict(orient='records'),
        method=method,
        top_genres=top_genres,
        mae=mae,
        rmse=rmse,
        latency=latency
    )
    

# Tìm phim theo từ khóa (title)
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    if query:
        result = items_data_og[items_data_og['title'].str.contains(query, case=False, na=False)]
    else:
        result = pd.DataFrame()
    return render_template('search.html', query=query, movies=result.to_dict(orient='records'))

# Đánh giá phim (thêm rating mới)
@app.route('/rate', methods=['POST'])
def rate_movie():
    if 'user_id' not in session:
        flash('Vui lòng đăng nhập để đánh giá phim')
        return redirect(url_for('login'))
    user_id = session['user_id']
    movie_id = int(request.form['movie_id'])
    rating = float(request.form['rating'])
    
    # Cập nhật ratings_data_og (giả định lưu tạm trong bộ nhớ)
    global ratings_data_og
    new_rating = pd.DataFrame({'user_id':[user_id], 'movie_id':[movie_id], 'rating':[rating]})
    ratings_data_og = pd.concat([ratings_data_og, new_rating], ignore_index=True)

    flash(f'Cảm ơn bạn đã đánh giá phim {movie_id} với điểm {rating}')
    return redirect(request.referrer or url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

# hiện các tiêu chí : phần trăm chính xác, thời gian dự đoán, số lượng phim đề xuất.