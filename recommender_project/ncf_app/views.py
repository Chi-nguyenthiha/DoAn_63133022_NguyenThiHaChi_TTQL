from django.shortcuts import render

# Create your views here.
import pandas as pd
import numpy as np
from django.shortcuts import render
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.utils.constants import SEED as DEFAULT_SEED

## top k items to recommend
TOP_K = 10

# Model parameters
EPOCHS = 20
BATCH_SIZE = 256

SEED = DEFAULT_SEED # Set None for non-deterministic results
# Load mô hình đã huấn luyện
MODEL_PATH = "/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/notebooks/checkpoints/ncf_model" # <-- Đổi thành đúng đường dẫn
train_file = "/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/test_train_1m/train_1m.csv"
test_file = "/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/test_train_1m/test_1m.csv"
leave_one_out_test_file = "/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/test_train_1m/leave_one_out_test_1m.csv"
# Load dữ liệu train (dùng để lấy danh sách itemID)


# Load lại mô hình NCF
data = NCFDataset(train_file=train_file, test_file=leave_one_out_test_file, seed=SEED, overwrite_test_file_full=True)

model = NCF(
    n_users=data.n_users, 
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=8,
    layer_sizes=[16, 8, 4],
    n_epochs=20,
    batch_size=256,
    learning_rate=1e-3,
    seed=DEFAULT_SEED
)
model.load(gmf_dir="/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/notebooks/.pretrain/GMF",
            mlp_dir="/Users/chi.nguyenth/Documents/DoAn_63133022_NguyenThiHaChi/notebooks/.pretrain/MLP", alpha=0.5)
#model.fit(data)
# Hàm chính để render form và trả kết quả
def recommend_view(request):
    recommendations = []

    if request.method == "POST":
        user_id = int(request.POST.get("user_id"))

        # Lấy danh sách items
        all_items = list(range(data.n_items))  # Giả sử n_items có sẵn trong dataset

        # Nếu mô hình yêu cầu ánh xạ user ID, hãy kiểm tra xem bạn cần lấy ánh xạ nào
        user_list = [user_id] * len(all_items)  # Hoặc sử dụng ánh xạ nếu cần

        item_list = all_items

        predictions = model.predict(user_list, item_list, is_list=True)
        pred_df = pd.DataFrame({
            "itemID": item_list,
            "prediction": predictions
        })

        top_k = pred_df.sort_values("prediction", ascending=False).head(10)
        recommendations = top_k["itemID"].tolist()

    return render(request, "recommend.html", {"recommendations": recommendations})


