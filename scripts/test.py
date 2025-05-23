# --- Import libraries ---
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# --- Load your data ---
ratings_data = pd.read_csv("/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/ratings.dat", sep='::', 
                            names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')

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
train_data, temp_data = train_test_split(ratings_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# --- Pivot full matrix from train only ---
ratings_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating')
rating_matrix_filled = ratings_matrix.fillna(0)

# --- Recompute similarities using train set ---
user_similarity = cosine_similarity(rating_matrix_filled)
item_similarity = cosine_similarity(rating_matrix_filled.T)

user_similarity_df = pd.DataFrame(user_similarity, index=rating_matrix_filled.index, columns=rating_matrix_filled.index)
item_similarity_df = pd.DataFrame(item_similarity, index=rating_matrix_filled.columns, columns=rating_matrix_filled.columns)

# --- Predict function ---
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

# --- Apply prediction on test set ---
test_data = test_data.copy()
test_data['pred_user_cf'] = test_data.apply(lambda row: predict_user_cf(row['user_id'], row['movie_id']), axis=1)
test_data['pred_item_cf'] = test_data.apply(lambda row: predict_item_cf(row['user_id'], row['movie_id']), axis=1)

# --- Drop NA predictions ---
test_data = test_data.dropna(subset=['pred_user_cf', 'pred_item_cf'])

# --- Evaluate ---
mae_user = mean_absolute_error(test_data['rating'], test_data['pred_user_cf'])
rmse_user = mean_squared_error(test_data['rating'], test_data['pred_user_cf'], squared=False)

mae_item = mean_absolute_error(test_data['rating'], test_data['pred_item_cf'])
rmse_item = mean_squared_error(test_data['rating'], test_data['pred_item_cf'], squared=False)

print(f"User-based CF -> MAE: {mae_user:.4f}, RMSE: {rmse_user:.4f}")
print(f"Item-based CF -> MAE: {mae_item:.4f}, RMSE: {rmse_item:.4f}")
