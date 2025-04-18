import numpy as np
import matplotlib.pyplot as plt
import random

# Đọc dữ liệu từ file gr10.tsp
def read_tsp_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
        # Tìm dòng bắt đầu ma trận khoảng cách
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == "EDGE_WEIGHT_SECTION":
                start_idx = i + 2  # +2 để bỏ qua dòng "EDGE_WEIGHT_SECTION" và dòng "0"
                break
                
        # Đọc ma trận khoảng cách
        distance_matrix = np.zeros((10, 10))
        row = 0
        
        while row < 10 and start_idx < len(lines):
            if lines[start_idx].strip() == "EOF":
                break
                
            values = lines[start_idx].strip().split()
            
            for i in range(row + 1):
                if i < len(values) - 1:  # -1 vì mỗi dòng kết thúc bằng 0
                    distance = int(values[i])
                    distance_matrix[row][i] = distance
                    distance_matrix[i][row] = distance  # Ma trận đối xứng
                    
            row += 1
            start_idx += 1
            
        return distance_matrix
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None

# Thuật toán Ant Colony Optimization cho TSP
class AntColony:
    def __init__(self, distances, n_ants=10, n_iterations=100, decay=0.5, alpha=1.0, beta=2.0):
        """
        Args:
            distances: Ma trận khoảng cách giữa các thành phố
            n_ants: Số lượng kiến
            n_iterations: Số lần lặp tối đa
            decay: Tốc độ bay hơi pheromone
            alpha: Trọng số cho pheromone
            beta: Trọng số cho khoảng cách
        """
        self.distances = distances
        self.pheromone = np.ones(distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
    def run(self):
        """Chạy thuật toán ACO"""
        shortest_path = None
        best_path = ("placeholder", np.inf)
        all_time_best_path = ("placeholder", np.inf)
        
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths, self.distances, self.decay)
            shortest_path = min(all_paths, key=lambda x: x[1])
            
            if shortest_path[1] < best_path[1]:
                best_path = shortest_path
                
            if best_path[1] < all_time_best_path[1]:
                all_time_best_path = best_path
                
            # In tiến trình
            if i % 10 == 0:
                print(f"Lặp {i}: Đường đi tốt nhất = {best_path[1]}")
                
        return all_time_best_path
    
    def spread_pheromone(self, all_paths, distances, decay):
        """Cập nhật mức pheromone trên các cạnh"""
        self.pheromone = self.pheromone * decay
        
        for path, dist in all_paths:
            for move in range(len(path) - 1):
                i, j = path[move], path[move + 1]
                self.pheromone[i, j] += 1.0 / distances[i, j]
                self.pheromone[j, i] += 1.0 / distances[j, i]  # Đồ thị vô hướng
    
    def gen_path_dist(self, path):
        """Tính tổng khoảng cách của đường đi"""
        total_dist = 0
        for i in range(len(path) - 1):
            total_dist += self.distances[path[i], path[i+1]]
        return total_dist
    
    def gen_all_paths(self):
        """Tạo đường đi cho tất cả kiến"""
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)  # Bắt đầu từ thành phố 0
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths
    
    def gen_path(self, start):
        """Tạo đường đi cho một con kiến từ điểm start"""
        path = [start]
        visited = {start}
        
        while len(visited) < len(self.distances):
            current = path[-1]
            unvisited = set(self.all_inds) - visited
            
            # Tính xác suất để chọn thành phố tiếp theo
            probs = np.zeros(len(self.distances))
            for j in unvisited:
                # Công thức: [pheromone]^alpha * [1/distance]^beta
                probs[j] = self.pheromone[current, j] ** self.alpha * \
                          (1.0 / self.distances[current, j]) ** self.beta
            
            # Chuẩn hóa xác suất
            probs = probs / probs.sum()
            
            # Chọn thành phố tiếp theo dựa trên xác suất
            next_city = np.random.choice(self.all_inds, p=probs)
            path.append(next_city)
            visited.add(next_city)
            
        # Thêm thành phố xuất phát vào cuối để tạo chu trình
        path.append(start)
        return path

# Hàm trực quan hóa đường đi tốt nhất
def visualize_path(path, distances):
    """Vẽ đồ thị đường đi tốt nhất"""
    plt.figure(figsize=(10, 6))
    
    # Tạo tọa độ ngẫu nhiên cho các thành phố để vẽ (do không có tọa độ thực trong file)
    coords = np.random.rand(len(distances), 2) * 10
    
    # Vẽ các thành phố
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=100)
    
    # Đánh số các thành phố
    for i in range(len(coords)):
        plt.annotate(f"{i}", (coords[i, 0] + 0.1, coords[i, 1] + 0.1), fontsize=12)
    
    # Vẽ đường đi
    for i in range(len(path) - 1):
        plt.plot([coords[path[i], 0], coords[path[i+1], 0]], 
                 [coords[path[i], 1], coords[path[i+1], 1]], 'b-')
    
    # Vẽ đường nối từ thành phố cuối về thành phố đầu
    plt.plot([coords[path[-1], 0], coords[path[0], 0]], 
             [coords[path[-1], 1], coords[path[0], 1]], 'b-')
    
    plt.title(f"Đường đi tối ưu: {path}\nTổng khoảng cách: {calculate_total_distance(path, distances)}")
    plt.grid(True)
    plt.savefig('tsp_best_path.png')
    plt.show()

def calculate_total_distance(path, distances):
    """Tính tổng khoảng cách của đường đi"""
    total = 0
    for i in range(len(path) - 1):
        total += distances[path[i], path[i+1]]
    return total

if __name__ == "__main__":
    # Đường dẫn đến file gr10.tsp
    file_path = "gr10.tsp"
    
    # Đọc ma trận khoảng cách từ file
    distances = read_tsp_file(file_path)
    
    if distances is None:
        # Nếu không đọc được file, tạo ma trận khoảng cách từ dữ liệu trong hình ảnh
        distances = np.zeros((10, 10))
        
        # Điền ma trận dựa trên dữ liệu trong ảnh
        distances[1, 0] = distances[0, 1] = 633
        distances[2, 0] = distances[0, 2] = 257
        distances[2, 1] = distances[1, 2] = 390
        distances[3, 0] = distances[0, 3] = 91
        distances[3, 1] = distances[1, 3] = 661
        distances[3, 2] = distances[2, 3] = 228
        distances[4, 0] = distances[0, 4] = 412
        distances[4, 1] = distances[1, 4] = 227
        distances[4, 2] = distances[2, 4] = 169
        distances[4, 3] = distances[3, 4] = 383
        distances[5, 0] = distances[0, 5] = 150
        distances[5, 1] = distances[1, 5] = 488
        distances[5, 2] = distances[2, 5] = 112
        distances[5, 3] = distances[3, 5] = 120
        distances[5, 4] = distances[4, 5] = 267
        distances[6, 0] = distances[0, 6] = 80
        distances[6, 1] = distances[1, 6] = 572
        distances[6, 2] = distances[2, 6] = 196
        distances[6, 3] = distances[3, 6] = 77
        distances[6, 4] = distances[4, 6] = 351
        distances[6, 5] = distances[5, 6] = 63
        distances[7, 0] = distances[0, 7] = 134
        distances[7, 1] = distances[1, 7] = 530
        distances[7, 2] = distances[2, 7] = 154
        distances[7, 3] = distances[3, 7] = 105
        distances[7, 4] = distances[4, 7] = 309
        distances[7, 5] = distances[5, 7] = 34
        distances[7, 6] = distances[6, 7] = 29
        distances[8, 0] = distances[0, 8] = 259
        distances[8, 1] = distances[1, 8] = 555
        distances[8, 2] = distances[2, 8] = 372
        distances[8, 3] = distances[3, 8] = 175
        distances[8, 4] = distances[4, 8] = 338
        distances[8, 5] = distances[5, 8] = 264
        distances[8, 6] = distances[6, 8] = 232
        distances[8, 7] = distances[7, 8] = 249
        distances[9, 0] = distances[0, 9] = 505
        distances[9, 1] = distances[1, 9] = 289
        distances[9, 2] = distances[2, 9] = 262
        distances[9, 3] = distances[3, 9] = 476
        distances[9, 4] = distances[4, 9] = 196
        distances[9, 5] = distances[5, 9] = 360
        distances[9, 6] = distances[6, 9] = 444
        distances[9, 7] = distances[7, 9] = 402
        distances[9, 8] = distances[8, 9] = 495
    
    print("Ma trận khoảng cách đã tải xong")
    
    # Khởi tạo và chạy thuật toán ACO
    aco = AntColony(distances, n_ants=20, n_iterations=100, decay=0.95, alpha=1, beta=2)
    best_path = aco.run()
    
    # In kết quả
    print("\n--- Kết quả tối ưu ---")
    print(f"Lộ trình: {best_path[0]}")
    print(f"Tổng khoảng cách: {best_path[1]}")
    
    # Vẽ đường đi tốt nhất
    visualize_path(best_path[0], distances)