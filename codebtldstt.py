import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ==========================================
# 1. KHỞI TẠO THÔNG SỐ (Mục 1 & 4)
# ==========================================
n = 100            # Số lượng trang web (tối thiểu 100 theo đề bài)
alpha = 0.85       # Hệ số cản (damping factor)
epsilon = 1e-8     # Sai số cho phép để dừng vòng lặp

# Tạo ma trận kề A ngẫu nhiên (Mô hình hóa hệ thống web dưới dạng đồ thị có hướng)
# p=[0.97, 0.03] nghĩa là xác suất có link giữa 2 trang là 3% (để đồ thị không quá dày)
A = np.random.choice([0, 1], size=(n, n), p=[0.97, 0.03])
np.fill_diagonal(A, 0) # Loại bỏ việc trang web tự trỏ đến chính nó

# ==========================================
# 2. XÂY DỰNG MA TRẬN LIÊN KẾT P & MA TRẬN GOOGLE G (Mục 2)
# ==========================================
# Xây dựng ma trận xác suất chuyển trạng thái P
P = A.astype(float)
for i in range(n):
    row_sum = np.sum(P[i, :])
    if row_sum == 0:
        P[i, :] = 1.0 / n  # Xử lý nút cụt (dangling nodes)
    else:
        P[i, :] = P[i, :] / row_sum

# Chuyển vị để tính toán theo cột: pi = G * pi
P = P.T 

# Xây dựng ma trận Google G theo công thức trong đề
# G = alpha * P + (1 - alpha) * (1/n) * E
E = np.ones((n, n)) / n
G = alpha * P + (1 - alpha) * E

# ==========================================
# 3. TÍNH TOÁN VECTƠ XẾP HẠNG PI (Mục 3)
# ==========================================
# Sử dụng phương pháp lặp lũy thừa (Power Method)
pi = np.ones(n) / n  # Vector ban đầu
for i in range(1000):
    pi_new = np.dot(G, pi)
    if np.linalg.norm(pi_new - pi, 1) < epsilon:
        print(f"--- Thuật toán hội tụ sau {i} bước lặp ---")
        break
    pi = pi_new

# ==========================================
# 4. TRỰC QUAN HÓA MÔ HÌNH (Mục 1 & Kết quả)
# ==========================================
def visualize_pagerank(A, scores):
    G_plot = nx.from_numpy_array(A, create_using=nx.DiGraph())
    pos = nx.spring_layout(G_plot, k=0.5)
    
    plt.figure(figsize=(15, 10))
    plt.title(f"Mô hình đồ thị PageRank cho {n} trang web")
    
    # Vẽ các nút: kích thước tỷ lệ với điểm số PageRank
    nodes = nx.draw_networkx_nodes(G_plot, pos, 
                                   node_size=scores * 20000, 
                                   node_color=scores, 
                                   cmap=plt.cm.YlOrRd)
    
    # Vẽ các cạnh (link) mờ để dễ nhìn
    nx.draw_networkx_edges(G_plot, pos, alpha=0.1, arrows=True)
    
    plt.colorbar(nodes, label='Chỉ số quan trọng (PageRank Score)')
    plt.axis('off')
    plt.show()

# Hiển thị Top 10 kết quả
print("\nTop 10 trang web có thứ hạng cao nhất:")
top_indices = np.argsort(pi)[-10:][::-1]
for rank, idx in enumerate(top_indices):
    print(f"Hạng {rank+1}: Trang {idx} - Điểm: {pi[idx]:.6f}")

# Gọi hàm vẽ đồ thị
visualize_pagerank(A, pi)
