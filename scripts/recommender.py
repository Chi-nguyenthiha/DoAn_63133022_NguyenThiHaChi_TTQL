import os
import sys
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import (
    map, ndcg_at_k, precision_at_k, recall_at_k
)
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.utils.notebook_utils import store_metadata
## top k items to recommend
TOP_K = 10

# Model parameters
EPOCHS = 20
BATCH_SIZE = 256

SEED = DEFAULT_SEED # Set None for non-deterministic results
# Đọc dữ liệu từ file CSV
# Đọc movies.dat và đổi tên cột MovieID -> ItemID
movies_df = pd.read_csv("/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/movies.dat", sep="::", engine="python", 
                     names=["itemID", "title", "genres"], encoding="latin1")

# Đọc ratings.dat và đổi tên cột MovieID -> ItemID
df = pd.read_csv("/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/ratings.dat", sep="::", engine="python", 
                      names=["userID", "itemID", "rating", "timestamp"])

print(movies_df.head())
print(df.head())
train, test = python_chrono_split(df, 0.75)
test = test[test["userID"].isin(train["userID"].unique())]
test = test[test["itemID"].isin(train["itemID"].unique())]
leave_one_out_test = test.groupby("userID").last().reset_index()
train_file = "/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/test_train_1m/train_1m.csv"
test_file = "/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/test_train_1m/test_1m.csv"
leave_one_out_test_file = "/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/test_train_1m/leave_one_out_test_1m.csv"
#train.to_csv(train_file, index=False)
#test.to_csv(test_file, index=False)
#leave_one_out_test.to_csv(leave_one_out_test_file, index=False)
# Đọc file CSV huấn luyện
df_train = pd.read_csv(train_file)

# Kiểm tra dữ liệu
print(df_train.head())

# Tách dữ liệu thành đầu vào và nhãn
user_input = df_train["userID"].values
item_input = df_train["itemID"].values
labels = df_train["rating"].values  # Nếu cần nhị phân hóa: labels = (labels > 0).astype(int)

print(user_input.shape, item_input.shape, labels.shape)
data = NCFDataset(train_file=train_file, test_file=leave_one_out_test_file, seed=SEED, overwrite_test_file_full=True)

model = NCF(
    n_users=data.n_users, 
    n_items=data.n_items,
    model_type="GMF",
    n_factors=8,
    layer_sizes=[16,8,4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
    seed=SEED
)
with Timer() as train_time:
    model.fit(data)

print("Took {} seconds for training.".format(train_time.interval))

model.save(dir_name=".pretrain/GMF")
model = NCF(
    n_users=data.n_users, 
    n_items=data.n_items,
    model_type="MLP",
    n_factors=8,
    layer_sizes=[16,8,4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
    seed=SEED
)
with Timer() as train_time:
    model.fit(data)

print("Took {} seconds for training.".format(train_time.interval))

model.save(dir_name=".pretrain/MLP")
model = NCF(
    n_users=data.n_users, 
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=8,
    layer_sizes=[16,8,4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
    seed=SEED
)

model.load(gmf_dir=".pretrain/GMF", mlp_dir=".pretrain/MLP", alpha=0.5)
with Timer() as train_time:
    model.fit(data)

print("Took {} seconds for training.".format(train_time.interval))
k = TOP_K

ndcgs = []
hit_ratio = []

for b in data.test_loader():
    user_input, item_input, labels = b
    output = model.predict(user_input, item_input, is_list=True)

    output = np.squeeze(output)
    rank = sum(output >= output[0])
    if rank <= k:
        ndcgs.append(1 / np.log(rank + 1))
        hit_ratio.append(1)
    else:
        ndcgs.append(0)
        hit_ratio.append(0)

eval_ndcg = np.mean(ndcgs)
eval_hr = np.mean(hit_ratio)

print("HR:\t%f" % eval_hr)
print("NDCG:\t%f" % eval_ndcg)
predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)]
               for (_, row) in test.iterrows()]


predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
predictions.head()
with Timer() as test_time:

    users, items, preds = [], [], []
    item = list(train.itemID.unique())
    for user in train.userID.unique():
        user = [user] * len(item) 
        users.extend(user)
        items.extend(item)
        preds.extend(list(model.predict(user, item, is_list=True)))

    all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

    merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
    all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

print("Took {} seconds for prediction.".format(test_time.interval))
#nếu tôi lấy 50% ngẫu nhiên từ data ra, 50% còn lại để trai/test huấn luyện mô hình thì tôi dùng mô hình đó dự đoán cho 50% tôi đã lấy ra trước đó không