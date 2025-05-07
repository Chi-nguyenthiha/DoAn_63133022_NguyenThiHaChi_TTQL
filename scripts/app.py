from flask import Flask, render_template, request
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sys

# Thêm thư mục chứa utils vào sys.path
sys.path.append('/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/scripts/ncf-utils')

from utils import Utils, EarlyStopping, cols_dict
from model import NCF

app = Flask(__name__)


# Đường dẫn dữ liệu
ratings_data_path = '/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/ratings.dat'
users_data_path = '/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/users.dat'
items_data_path = '/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/movies.dat'

ratings_data = pd.read_csv(ratings_data_path, sep='::', names=cols_dict['ratings'], engine='python')
users_data = pd.read_csv(users_data_path, sep='::', names=cols_dict['users'], engine='python')
items_data = pd.read_csv(items_data_path, sep='::', names=cols_dict['items'], encoding='latin-1', engine='python')

users_data_og = users_data.copy()
items_data_og = items_data.copy()
ratings_data_og = ratings_data.copy()

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
model.load_state_dict(torch.load('/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/notebooks/weights/explicit.pth', map_location=device))
model.eval()

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

    # Bước 6: Sắp xếp phim
    movies_retrieved = items_data_og[items_data_og['movie_id'].isin(movie_ids.cpu().numpy())]
    movies_retrieved = movies_retrieved.sort_values(
        by='movie_id',
        key=lambda x: pd.Categorical(x, categories=movie_ids.cpu().numpy(), ordered=True)
    )

    top_recommendations = Utils.order(y_pred, movies_retrieved, mode='explicit', top_k=top_k)
    return top_recommendations


@app.route('/recommend_existing', methods=['GET', 'POST'])
def recommend_existing():
    if request.method == 'POST':
        try:
            user_id = int(request.form['user_id'])  # Lấy user_id từ form
            recommended_df = recommend_movies_for_existing_user(
                user_id_int=user_id,
                model=model,
                users_data=users_data,
                items_data_og=items_data_og,
                ratings_data_og=ratings_data_og,
                top_k=10
            )
            # Truyền dữ liệu vào template để hiển thị
            return render_template('recommend_result.html', tables=[recommended_df.to_html(classes='data')], user_id=user_id)
        except Exception as e:
            return f"Lỗi: {str(e)}"
    return render_template('recommend_form.html')

if __name__ == '__main__':
    app.run(debug=True)
