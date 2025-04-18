import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
import random

# 1. Đọc dữ liệu
df = pd.read_csv('/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/dataset/1m/ratings.dat', sep='::', engine='python',
                 names=['user_id', 'item_id', 'rating', 'timestamp'])

# 2. Chia user 80/20
unique_users = df['user_id'].unique()
train_users, holdout_users = train_test_split(unique_users, test_size=0.2, random_state=42)

df_train = df[df['user_id'].isin(train_users)].copy()
df_holdout = df[df['user_id'].isin(holdout_users)].copy()

# 3. Encode user/item ID từ tập train
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

df_train['user'] = user_encoder.fit_transform(df_train['user_id'])
df_train['item'] = item_encoder.fit_transform(df_train['item_id'])

# Encode holdout theo mapping của train
df_holdout = df_holdout[df_holdout['user_id'].isin(user_encoder.classes_) & df_holdout['item_id'].isin(item_encoder.classes_)]
df_holdout['user'] = user_encoder.transform(df_holdout['user_id'])
df_holdout['item'] = item_encoder.transform(df_holdout['item_id'])

# 4. Gán nhãn nhị phân
df_train['label'] = (df_train['rating'] >= 4).astype(int)
df_holdout['label'] = (df_holdout['rating'] >= 4).astype(int)

# 5. Tạo negative sampling từ train
num_users = df_train['user'].nunique()
num_items = df_train['item'].nunique()

user_item_set = set(zip(df_train['user'], df_train['item']))
positive_samples = df_train[df_train['label'] == 1][['user', 'item']].values.tolist()
negative_samples = []

np.random.seed(42)
for (u, _) in positive_samples:
    for _ in range(4):  # 4 negative mỗi positive
        j = np.random.randint(num_items)
        while (u, j) in user_item_set:
            j = np.random.randint(num_items)
        negative_samples.append([u, j])

# 6. Chuẩn bị dữ liệu train
train_data = [[u, i, 1] for (u, i) in positive_samples] + [[u, i, 0] for (u, i) in negative_samples]
random.shuffle(train_data)

users_train = np.array([d[0] for d in train_data])
items_train = np.array([d[1] for d in train_data])
labels_train = np.array([d[2] for d in train_data])

# 7. Xây mô hình NCF
embedding_dim = 32
user_input = Input(shape=(1,))
item_input = Input(shape=(1,))
user_embedding = Embedding(num_users, embedding_dim)(user_input)
item_embedding = Embedding(num_items, embedding_dim)(item_input)

user_vec = Flatten()(user_embedding)
item_vec = Flatten()(item_embedding)

concat = Concatenate()([user_vec, item_vec])
x = Dense(64, activation='relu')(concat)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model([user_input, item_input], output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 8. Train mô hình
model.fit([users_train, items_train], labels_train,
          batch_size=256, epochs=5, validation_split=0.1)

# 9. Dự đoán cho tập holdout
users_holdout = df_holdout['user'].values
items_holdout = df_holdout['item'].values
labels_holdout = df_holdout['label'].values

pred_probs = model.predict([users_holdout, items_holdout], batch_size=512)

# 10. Tính AUC & Accuracy
auc = roc_auc_score(labels_holdout, pred_probs)
acc = accuracy_score(labels_holdout, pred_probs >= 0.5)

print(f"\n🎯 Holdout Test Results:")
print(f"AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
