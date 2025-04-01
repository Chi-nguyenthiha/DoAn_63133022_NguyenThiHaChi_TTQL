import tensorflow as tf
from tensorflow.keras.models import load_model

# Load mô hình đã train

model_path = "/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/model/ncf_model"
# Load mô hình
model = load_model(model_path)


