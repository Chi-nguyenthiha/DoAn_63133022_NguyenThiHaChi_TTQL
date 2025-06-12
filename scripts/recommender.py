# file pythone xây dựng mô hình NCF (Neural Collaborative Filtering) với dữ liệu MovieLens 1M
import torch, pandas, numpy, sklearn, IPython
import sys
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from IPython.display import display
sys.path.append('/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/scripts/ncf-utils') #đổi đường dẫn này theo thư mục của bạn
from utils import Utils, EarlyStopping, cols_dict
from model import NCF

sys.path.append('/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/scripts/ncf-utils')
pd.set_option("display.max_columns", 6)
import os
os.makedirs("weightss", exist_ok=True)
print("🔹 Loading data...")
#đổi đường dẫn này theo thư mục của bạn
ratings_data = pd.read_csv('/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/ratings.dat', sep='::', names=cols_dict['ratings'], engine='python')
users_data = pd.read_csv('/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/users.dat', sep='::', names=cols_dict['users'], engine='python')
items_data = pd.read_csv('/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/movies.dat', sep='::', names=cols_dict['items'], encoding='latin-1', engine='python')

print("✅ Data loaded.")
# --- BACKUP ORIGINAL DATA ---
users_data_og = users_data.copy()
items_data_og = items_data.copy()
ratings_data_og = ratings_data.copy()

# One-hot encode gender and occupation for users
print("🔹 One-hot encoding users' gender and occupation...")
users_data = Utils.one_hot_encode(users_data, ['occupation', 'gender', 'age'])

# Multi-hot encode genres for items
print("🔹 Multi-hot encoding items' genres...")
items_data = Utils.multi_hot_encode(items_data, 'genre')

# Display a sample of transformed data
print("Users data after encoding:")
display(users_data.head(3))

print("Items data after encoding:")
display(items_data.head(3))

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

# --- MODEL SETUP ---
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

early_stopping = EarlyStopping(patience=3, delta=0.0002, path='weightss/explicit.pth')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer, mode='min', factor=0.1, patience=0)

history = model.fit(
    X=[y_ratings_train[:, 0], y_ratings_train[:, 1], X_users_train, X_items_train],
    y=y_ratings_train[:, 2],
    X_val=[y_ratings_val[:, 0], y_ratings_val[:, 1], X_users_val, X_items_val],
    y_val=y_ratings_val[:, 2],
    epochs=12,
    batch_size=2048,
    early_stopping=early_stopping,
    scheduler=scheduler
)

Utils.plot_metrics(history, 'Explicit Feedback')
# --- Step 6: Model Evaluation ---
print("🔹 Evaluating model on test data...")

model.eval() # Set model to evaluation mode (disables dropout layers)

with torch.no_grad():
    user_ids = torch.tensor(y_ratings_test[:, 0], dtype=torch.long).to(model.device)
    item_ids = torch.tensor(y_ratings_test[:, 1], dtype=torch.long).to(model.device)
    user_features = torch.tensor(X_users_test, dtype=torch.float32).to(model.device)
    item_features = torch.tensor(X_items_test, dtype=torch.float32).to(model.device)
    
    # predict
    y_pred = model(user_ids, item_ids, user_features, item_features)
    y_pred = y_pred.cpu().numpy()
# Đưa dữ liệu thực và dự đoán vào DataFrame
results_df = pd.DataFrame({
    'user_id': y_ratings_test[:, 0],
    'movie_id': y_ratings_test[:, 1],
    'true_rating': y_ratings_test[:, 2],    # Rating gốc
    'predicted_rating': y_pred.flatten()    # Rating model dự đoán
})

# Scale lại nếu cần (vì bạn normalize rating từ 0–1 lúc training)
results_df['true_rating'] = results_df['true_rating'] * 5
results_df['predicted_rating'] = results_df['predicted_rating'] * 5

# Preview
display(results_df.head(10))
results_df['abs_error'] = abs(results_df['true_rating'] - results_df['predicted_rating'])
mae = results_df['abs_error'].mean()
print(f"MAE: {mae:.4f}")
new_user = {
    'id': 7000, # new user id
    'age': 20,
    'occupation': 'engineer',
    'gender': 'F',
    'genres': ['Children', 'Comedy', 'Animation'],
}

user_id, user, weights, _ = Utils.preprocess_user(
                                user=new_user,
                                num_items=items_data_og.shape[0],
                                users=users_data.drop_duplicates(inplace=False).values,
                                weights=[model.user_embedding_mlp.weight.data.cpu().numpy(), model.user_embedding_mf.weight.data.cpu().numpy()]
                                )
items = Utils.preprocess_items(items_data_og)

user_id, user = user_id.to(model.device), user.to(model.device)
# 1- Retrieval Stage
movies = Utils.retrieve(
    movies=items,
    user=user.detach().cpu().numpy(),
    num_genres=3,
    k=300, # retrieve 300 relevant movies. Note: higher k leads to better recommendations but slower inference
    random_state=0
)

# 2- Filtering Stage
movie_ids, movies = Utils.filter( # Removes already-rated / duplicate movies
    movies=movies,
    ratings=ratings_data_og,
    user_id=new_user['id']
)
movie_ids, movies = movie_ids.to(model.device), movies.to(model.device)

# 3- Ranking Stage
y_pred = model(
    user_id[:len(movies)],
    movie_ids,
    user[:len(movies)],
    movies,
    weights
).cpu().detach().numpy()

# 4- Ordering Stage
movies_retrieved = items_data_og[items_data_og['movie_id'].isin(movie_ids.cpu().numpy())].sort_values(by='movie_id', key=lambda x: pd.Categorical(x, categories=movie_ids.cpu().numpy(), ordered=True))
Utils.order(y_pred, movies_retrieved, 'explicit', top_k=10)
old_user = {
    'id': 400
}

# preprocess the old user
user_id, user, _, _ = Utils.preprocess_user(
                                user=old_user,
                                num_items=items_data_og.shape[0],
                                users=users_data.drop_duplicates(inplace=False).values,
                                topk=3, # top 3 genres the user has interacted with. MAX: 18
                                verbose=True
                                )
items = Utils.preprocess_items(items_data_og)

user_id, user = user_id.to(model.device), user.to(model.device)

# 1- Retrieval Stage
movies = Utils.retrieve(
    movies=items,
    user=user.detach().cpu().numpy(),
    k=300, # retrieve 300 relevant movies. Note: higher k leads to better recommendations but slower inference
    random_state=0
)

# 2- Filtering Stage
movie_ids, movies = Utils.filter( # Removes already-rated / duplicate movies
    movies=movies,
    ratings=ratings_data_og,
    user_id=new_user['id']
)
movie_ids, movies = movie_ids.to(model.device), movies.to(model.device)

# 3- Ranking Stage
y_pred = model(
    user_id[:len(movies)],
    movie_ids,
    user[:len(movies)],
    movies
).cpu().detach().numpy()

# 4- Ordering Stage
movies_retrieved = items_data_og[items_data_og['movie_id'].isin(movie_ids.cpu().numpy())].sort_values(by='movie_id', key=lambda x: pd.Categorical(x, categories=movie_ids.cpu().numpy(), ordered=True))
Utils.order(y_pred, movies_retrieved, 'eplicit', top_k=10)

