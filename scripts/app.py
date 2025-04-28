from flask import Flask, render_template, request
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
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

# Chia tập dữ liệu
X_users_train, X_users_test = train_test_split(users_data, test_size=0.2, random_state=42)
X_items_train, X_items_test = train_test_split(items_data, test_size=0.2, random_state=42)
y_ratings_train, y_ratings_test = train_test_split(ratings_data, test_size=0.2, random_state=42)

# Chuyển dữ liệu sang dạng numpy
X_users_train, X_users_test = X_users_train.values, X_users_test.values
X_items_train, X_items_test = X_items_train.values, X_items_test.values
y_ratings_train, y_ratings_test = y_ratings_train.values, y_ratings_test.values

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

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_type = request.form['user_type']
    
    if user_type == 'new':
        # Xử lý người dùng mới
        age = int(request.form['age'])
        occupation = request.form['occupation']
        gender = request.form['gender']
        genres = request.form.getlist('genres')
        
        new_user = {
            'id': 7000,  # ID người dùng mới (có thể thay đổi)
            'age': age,
            'occupation': occupation,
            'gender': gender,
            'genres': genres
        }
        
        user_id, user, weights, _ = Utils.preprocess_user(
            user=new_user,
            num_items=items_data_og.shape[0],
            users=users_data_og.drop_duplicates(inplace=False).values,
            weights=[model.user_embedding_mlp.weight.data.cpu().numpy(), model.user_embedding_mf.weight.data.cpu().numpy()]
        )
        
    else:
        # Xử lý người dùng cũ
        old_user_id = int(request.form['old_user_id'])
        old_user = {'id': old_user_id}
        
        user_id, user, weights, _ = Utils.preprocess_user(
            user=old_user,
            num_items=items_data_og.shape[0],
            users=users_data_og.drop_duplicates(inplace=False).values,
            topk=3
        )
        
    # Tiền xử lý item
    items = Utils.preprocess_items(items_data_og)
    
    user_id, user = user_id.to(model.device), user.to(model.device)
    
    # 1. Retrieval Stage
    movies = Utils.retrieve(
        movies=items,
        user=user.detach().cpu().numpy(),
        num_genres=3,
        k=300,
        random_state=0
    )
    
    # 2. Filtering Stage
    movie_ids, movies = Utils.filter(
        movies=movies,
        ratings=ratings_data_og,
        user_id=user_id.item()
    )
    
    movie_ids, movies = movie_ids.to(model.device), movies.to(model.device)
    
    # 3. Ranking Stage
    y_pred = model(
        user_id[:len(movies)],
        movie_ids,
        user[:len(movies)],
        movies,
        weights
    ).cpu().detach().numpy()
    
    # 4. Ordering Stage
    movies_retrieved = items_data_og[items_data_og['movie_id'].isin(movie_ids.cpu().numpy())].sort_values(
        by='movie_id', key=lambda x: pd.Categorical(x, categories=movie_ids.cpu().numpy(), ordered=True)
    )
    
    # Lấy top 10 phim
    top_movies = Utils.order(y_pred, movies_retrieved, 'explicit', top_k=10)
    
    return render_template('results.html', movies=top_movies)

if __name__ == '__main__':
    app.run(debug=True)
