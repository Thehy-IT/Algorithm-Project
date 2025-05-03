import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from typing import List, Tuple
import sys
import os

def read_tsp_file(file_path: str) -> np.ndarray:
    """
    Đọc file TSP dạng LOWER_DIAG_ROW và trả về ma trận khoảng cách
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Tìm số lượng thành phố (DIMENSION)
    dimension = None
    for line in lines:
        if line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
            break
    
    if dimension is None:
        raise ValueError("Không tìm thấy thông tin về DIMENSION trong file TSP")
    
    # Tìm phần dữ liệu của ma trận khoảng cách
    edge_weight_section_index = lines.index("EDGE_WEIGHT_SECTION\n") + 1
    data_lines = lines[edge_weight_section_index:edge_weight_section_index + dimension]
    
    # Tạo ma trận khoảng cách từ dữ liệu LOWER_DIAG_ROW
    distance_matrix = np.zeros((dimension, dimension))
    
    row_index = 0
    col_index = 0
    
    for line in data_lines:
        values = list(map(int, line.strip().split()))
        for value in values:
            distance_matrix[row_index][col_index] = value
            distance_matrix[col_index][row_index] = value  # Ma trận đối xứng
            col_index += 1
            if col_index > row_index:
                row_index += 1
                col_index = 0
    
    return distance_matrix

def measure_performance(func):
    """
    Decorator để đo thời gian thực thi và bộ nhớ sử dụng của một hàm
    """
    def wrapper(*args, **kwargs):
        # Đo thời gian
        start_time = time.time()
        
        # Đo bộ nhớ trước khi chạy
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Chạy hàm
        result = func(*args, **kwargs)
        
        # Đo thời gian sau khi chạy
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Đo bộ nhớ sau khi chạy
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        return result, execution_time, memory_used
    
    return wrapper

def calculate_path_distance(path: List[int], distance_matrix: np.ndarray) -> float:
    """
    Tính tổng khoảng cách cho một đường đi
    """
    total_distance = 0
    n = len(path)
    
    for i in range(n - 1):
        total_distance += distance_matrix[path[i]][path[i+1]]
    
    # Thêm khoảng cách từ thành phố cuối cùng về thành phố đầu tiên
    total_distance += distance_matrix[path[n-1]][path[0]]
    
    return total_distance

@measure_performance
def greedy_algorithm(distance_matrix: np.ndarray) -> List[int]:
    """
    Thuật toán tham lam (Nearest Neighbor)
    """
    n = distance_matrix.shape[0]
    if n > 1000:
        return [], float('inf')  # Bỏ qua nếu n quá lớn
    
    # Bắt đầu từ thành phố 0
    current_city = 0
    unvisited_cities = set(range(1, n))
    path = [current_city]
    
    # Lặp cho đến khi tất cả các thành phố đều được thăm
    while unvisited_cities:
        # Tìm thành phố kế tiếp gần nhất chưa được thăm
        next_city = min(unvisited_cities, key=lambda city: distance_matrix[current_city][city])
        path.append(next_city)
        unvisited_cities.remove(next_city)
        current_city = next_city
    
    total_distance = calculate_path_distance(path, distance_matrix)
    return path, total_distance

@measure_performance
def brute_force(distance_matrix: np.ndarray) -> List[int]:
    """
    Thuật toán vét cạn (Brute Force)
    """
    n = distance_matrix.shape[0]
    if n > 11:  # Giới hạn n <= 11 vì độ phức tạp thời gian là O(n!)
        return [], float('inf')
    
    # Tạo tất cả các hoán vị có thể
    all_cities = list(range(1, n))  # Không bao gồm thành phố đầu tiên (0)
    best_path = None
    best_distance = float('inf')
    
    # Kiểm tra tất cả các hoán vị
    for perm in itertools.permutations(all_cities):
        path = [0] + list(perm)  # Bắt đầu từ thành phố 0
        distance = calculate_path_distance(path, distance_matrix)
        
        if distance < best_distance:
            best_distance = distance
            best_path = path
    
    return best_path, best_distance

@measure_performance
def approximation_algorithm(distance_matrix: np.ndarray) -> List[int]:
    """
    Thuật toán xấp xỉ (2-opt)
    """
    n = distance_matrix.shape[0]
    if n > 500:
        return [], float('inf')  # Bỏ qua nếu n quá lớn
    
    # Bắt đầu với một đường đi ngẫu nhiên
    current_path = list(range(n))
    random.shuffle(current_path)
    best_distance = calculate_path_distance(current_path, distance_matrix)
    
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Đảo ngược đoạn từ i đến j
                new_path = current_path[:i] + current_path[i:j+1][::-1] + current_path[j+1:]
                new_distance = calculate_path_distance(new_path, distance_matrix)
                
                if new_distance < best_distance:
                    current_path = new_path
                    best_distance = new_distance
                    improved = True
                    break
            if improved:
                break
    
    return current_path, best_distance

@measure_performance
def ant_colony_optimization(distance_matrix: np.ndarray) -> List[int]:
    """
    Thuật toán bầy kiến (Ant Colony Optimization)
    """
    n = distance_matrix.shape[0]
    if n > 150:
        return [], float('inf')  # Bỏ qua nếu n quá lớn
    
    # Tham số thuật toán
    num_ants = min(n * 2, 50)  # Số lượng kiến
    num_iterations = 50  # Số lần lặp
    alpha = 1.0  # Trọng số pheromone
    beta = 2.0  # Trọng số khoảng cách
    evaporation_rate = 0.5  # Tỷ lệ bay hơi pheromone
    
    # Khởi tạo pheromone
    pheromone = np.ones((n, n))
    
    # Khởi tạo ma trận heuristic (1/distance)
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and distance_matrix[i][j] > 0:
                heuristic[i][j] = 1.0 / distance_matrix[i][j]
    
    best_path = None
    best_distance = float('inf')
    
    # Lặp qua các vòng
    for iteration in range(num_iterations):
        # Mỗi con kiến tạo một đường đi
        all_paths = []
        
        for ant in range(num_ants):
            # Chọn thành phố bắt đầu ngẫu nhiên
            current_city = random.randint(0, n - 1)
            path = [current_city]
            unvisited = set(range(n))
            unvisited.remove(current_city)
            
            # Xây dựng đường đi
            while unvisited:
                # Tính xác suất chọn thành phố tiếp theo
                probabilities = []
                denominator = 0.0
                
                for next_city in unvisited:
                    p = (pheromone[current_city][next_city] ** alpha) * (heuristic[current_city][next_city] ** beta)
                    probabilities.append((next_city, p))
                    denominator += p
                
                # Chuẩn hóa xác suất
                for i in range(len(probabilities)):
                    city, p = probabilities[i]
                    probabilities[i] = (city, p / denominator if denominator > 0 else 0)
                
                # Chọn thành phố tiếp theo theo xác suất
                r = random.random()
                cumulative_prob = 0.0
                next_city = None
                
                for city, p in probabilities:
                    cumulative_prob += p
                    if r <= cumulative_prob:
                        next_city = city
                        break
                
                if next_city is None and unvisited:  # Trường hợp lỗi xác suất
                    next_city = random.choice(list(unvisited))
                
                path.append(next_city)
                unvisited.remove(next_city)
                current_city = next_city
            
            # Tính khoảng cách cho đường đi
            distance = calculate_path_distance(path, distance_matrix)
            all_paths.append((path, distance))
            
            # Cập nhật đường đi ngắn nhất
            if distance < best_distance:
                best_distance = distance
                best_path = path
        
        # Cập nhật pheromone
        pheromone *= (1 - evaporation_rate)  # Bay hơi
        
        # Thêm pheromone từ các con kiến
        for path, dist in all_paths:
            # Đóng góp pheromone tỷ lệ nghịch với khoảng cách
            contribution = 1.0 / dist if dist > 0 else 0
            
            for i in range(len(path) - 1):
                pheromone[path[i]][path[i+1]] += contribution
                pheromone[path[i+1]][path[i]] += contribution  # Đồ thị vô hướng
            
            # Pheromone cho cạnh cuối cùng (từ thành phố cuối cùng về thành phố đầu tiên)
            pheromone[path[-1]][path[0]] += contribution
            pheromone[path[0]][path[-1]] += contribution
    
    return best_path, best_distance

@measure_performance
def genetic_algorithm(distance_matrix: np.ndarray) -> List[int]:
    """
    Thuật toán di truyền (Genetic Algorithm)
    """
    n = distance_matrix.shape[0]
    if n > 150:
        return [], float('inf')  # Bỏ qua nếu n quá lớn
    
    # Tham số thuật toán
    population_size = min(n * 10, 200)  # Kích thước quần thể
    num_generations = 100  # Số thế hệ
    mutation_rate = 0.1  # Tỷ lệ đột biến
    
    # Tạo quần thể ban đầu
    population = []
    for _ in range(population_size):
        path = list(range(n))
        random.shuffle(path)
        fitness = 1.0 / calculate_path_distance(path, distance_matrix)
        population.append((path, fitness))
    
    best_path = None
    best_distance = float('inf')
    
    # Lặp qua các thế hệ
    for generation in range(num_generations):
        # Sắp xếp quần thể theo độ thích nghi (fitness)
        population.sort(key=lambda x: x[1], reverse=True)
        
        # Cập nhật đường đi tốt nhất
        current_best_path = population[0][0]
        current_best_distance = calculate_path_distance(current_best_path, distance_matrix)
        
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_path = current_best_path
        
        # Chọn các cá thể để sinh sản (selection)
        parents = population[:population_size // 2]
        
        # Tạo thế hệ mới
        offspring = []
        
        while len(offspring) < population_size:
            # Chọn ngẫu nhiên 2 cha mẹ
            parent1_idx = random.randint(0, len(parents) - 1)
            parent2_idx = random.randint(0, len(parents) - 1)
            
            # Đảm bảo 2 cha mẹ khác nhau
            while parent2_idx == parent1_idx:
                parent2_idx = random.randint(0, len(parents) - 1)
            
            parent1 = parents[parent1_idx][0]
            parent2 = parents[parent2_idx][0]
            
            # Lai ghép (crossover) - Partially Mapped Crossover (PMX)
            crossover_point1 = random.randint(0, n - 2)
            crossover_point2 = random.randint(crossover_point1 + 1, n - 1)
            
            # Khởi tạo con với giá trị -1
            child = [-1] * n
            
            # Sao chép đoạn từ parent1
            for i in range(crossover_point1, crossover_point2 + 1):
                child[i] = parent1[i]
            
            # Ánh xạ các phần tử từ parent2
            for i in range(n):
                if i < crossover_point1 or i > crossover_point2:
                    # Kiểm tra xem phần tử từ parent2 đã có trong child chưa
                    element = parent2[i]
                    while element in child:
                        # Tìm vị trí của element trong parent1
                        idx = parent1.index(element)
                        # Lấy phần tử từ parent2 tại vị trí đó
                        element = parent2[idx]
                    
                    child[i] = element
            
            # Đột biến (mutation) - Swap Mutation
            if random.random() < mutation_rate:
                idx1 = random.randint(0, n - 1)
                idx2 = random.randint(0, n - 1)
                child[idx1], child[idx2] = child[idx2], child[idx1]
            
            # Tính fitness cho con
            child_fitness = 1.0 / calculate_path_distance(child, distance_matrix)
            offspring.append((child, child_fitness))
        
        # Thay thế quần thể cũ bằng quần thể mới
        population = offspring
    
    return best_path, best_distance

@measure_performance
def nearest_neighbor(distance_matrix: np.ndarray) -> List[int]:
    """
    Thuật toán láng giềng gần nhất (Nearest Neighbor)
    """
    n = distance_matrix.shape[0]
    if n > 10000:
        return [], float('inf')  # Bỏ qua nếu n quá lớn
    
    # Thử với mỗi thành phố là điểm xuất phát
    best_path = None
    best_distance = float('inf')
    
    for start_city in range(n):
        current_city = start_city
        unvisited_cities = set(range(n))
        unvisited_cities.remove(current_city)
        path = [current_city]
        
        # Lặp cho đến khi tất cả các thành phố đều được thăm
        while unvisited_cities:
            # Tìm thành phố kế tiếp gần nhất chưa được thăm
            next_city = min(unvisited_cities, key=lambda city: distance_matrix[current_city][city])
            path.append(next_city)
            unvisited_cities.remove(next_city)
            current_city = next_city
        
        # Tính khoảng cách cho đường đi
        distance = calculate_path_distance(path, distance_matrix)
        
        # Cập nhật đường đi ngắn nhất
        if distance < best_distance:
            best_distance = distance
            best_path = path
    
    return best_path, best_distance

def visualize_results(results: List[Tuple[str, float, float, float]]):
    """
    Trực quan hóa kết quả thực hiện của các thuật toán
    """
    algorithms = [result[0] for result in results]
    distances = [result[1] for result in results]
    times = [result[2] for result in results]
    memories = [result[3] for result in results]
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Đồ thị cột cho thời gian thực hiện
    x = np.arange(len(algorithms))
    width = 0.35
    bars = ax1.bar(x - width/2, times, width, label='Thời gian (giây)', color='skyblue')
    
    # Thêm nhãn cho các cột
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Trục y bên phải cho bộ nhớ
    ax2 = ax1.twinx()
    ax2.plot(x, memories, 'ro-', linewidth=2, markersize=8, label='Bộ nhớ (MB)')
    
    # Thêm nhãn cho đường
    for i, memory in enumerate(memories):
        ax2.annotate(f'{memory:.2f}',
                    xy=(i, memory),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Thêm các nhãn và tiêu đề
    ax1.set_xlabel('Thuật toán')
    ax1.set_ylabel('Thời gian thực hiện (giây)')
    ax2.set_ylabel('Bộ nhớ sử dụng (MB)')
    ax1.set_title('So sánh hiệu suất của các thuật toán TSP')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    
    # Thêm chú thích
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('tsp_algorithm_comparison.png')
    plt.close()

def run_tsp_solver(file_path: str):
    """
    Hàm chính để chạy toàn bộ quá trình giải quyết TSP
    """
    # Đọc dữ liệu từ file
    print(f"Đọc dữ liệu từ file {file_path}...")
    distance_matrix = read_tsp_file(file_path)
    n = distance_matrix.shape[0]
    print(f"Đã đọc xong ma trận khoảng cách cho {n} thành phố.")
    
    # Khởi tạo kết quả
    results = []
    
    # Chạy các thuật toán
    print("\nBắt đầu chạy các thuật toán:")
    print("=" * 50)
    
    # 1. Thuật toán tham lam
    print("1. Đang chạy thuật toán tham lam...")
    if n <= 1000:
        (greedy_path, greedy_distance), greedy_time, greedy_memory = greedy_algorithm(distance_matrix)
        print(f"   Lộ trình: {greedy_path}")
        print(f"   Tổng khoảng cách: {greedy_distance}")
        print(f"   Thời gian: {greedy_time:.6f} giây")
        print(f"   Bộ nhớ: {greedy_memory:.6f} MB")
        results.append(("Tham lam", greedy_distance, greedy_time, greedy_memory))
    else:
        print("   Bỏ qua vì số lượng thành phố quá lớn (> 1000)")
    
    # 2. Thuật toán brute force
    print("\n2. Đang chạy thuật toán brute force...")
    if n <= 11:
        (bf_path, bf_distance), bf_time, bf_memory = brute_force(distance_matrix)
        print(f"   Lộ trình: {bf_path}")
        print(f"   Tổng khoảng cách: {bf_distance}")
        print(f"   Thời gian: {bf_time:.6f} giây")
        print(f"   Bộ nhớ: {bf_memory:.6f} MB")
        results.append(("Brute Force", bf_distance, bf_time, bf_memory))
    else:
        print("   Bỏ qua vì số lượng thành phố quá lớn (> 11)")
    
    # 3. Thuật toán xấp xỉ (2-opt)
    print("\n3. Đang chạy thuật toán xấp xỉ (2-opt)...")
    if n <= 500:
        (approx_path, approx_distance), approx_time, approx_memory = approximation_algorithm(distance_matrix)
        print(f"   Lộ trình: {approx_path}")
        print(f"   Tổng khoảng cách: {approx_distance}")
        print(f"   Thời gian: {approx_time:.6f} giây")
        print(f"   Bộ nhớ: {approx_memory:.6f} MB")
        results.append(("Xấp xỉ (2-opt)", approx_distance, approx_time, approx_memory))
    else:
        print("   Bỏ qua vì số lượng thành phố quá lớn (> 500)")
    
    # 4. Thuật toán bầy kiến
    print("\n4. Đang chạy thuật toán bầy kiến...")
    if n <= 150:
        (ant_path, ant_distance), ant_time, ant_memory = ant_colony_optimization(distance_matrix)
        print(f"   Lộ trình: {ant_path}")
        print(f"   Tổng khoảng cách: {ant_distance}")
        print(f"   Thời gian: {ant_time:.6f} giây")
        print(f"   Bộ nhớ: {ant_memory:.6f} MB")
        results.append(("Bầy kiến", ant_distance, ant_time, ant_memory))
    else:
        print("   Bỏ qua vì số lượng thành phố quá lớn (> 150)")
    
    # 5. Thuật toán di truyền
    print("\n5. Đang chạy thuật toán di truyền...")
    if n <= 150:
        (ga_path, ga_distance), ga_time, ga_memory = genetic_algorithm(distance_matrix)
        print(f"   Lộ trình: {ga_path}")
        print(f"   Tổng khoảng cách: {ga_distance}")
        print(f"   Thời gian: {ga_time:.6f} giây")
        print(f"   Bộ nhớ: {ga_memory:.6f} MB")
        results.append(("Di truyền", ga_distance, ga_time, ga_memory))
    else:
        print("   Bỏ qua vì số lượng thành phố quá lớn (> 150)")
    
    # 6. Thuật toán láng giềng gần nhất
    print("\n6. Đang chạy thuật toán láng giềng gần nhất...")
    if n <= 10000:
        (nn_path, nn_distance), nn_time, nn_memory = nearest_neighbor(distance_matrix)
        print(f"   Lộ trình: {nn_path}")
        print(f"   Tổng khoảng cách: {nn_distance}")
        print(f"   Thời gian: {nn_time:.6f} giây")
        print(f"   Bộ nhớ: {nn_memory:.6f} MB")
        results.append(("Láng giềng gần nhất", nn_distance, nn_time, nn_memory))
    else:
        print("   Bỏ qua vì số lượng thành phố quá lớn (> 10000)")
    
    print("\n" + "=" * 50)
    
    # Trực quan hóa kết quả
    if results:
        print("\nTạo biểu đồ so sánh hiệu suất...")
        visualize_results(results)
        print("Đã lưu biểu đồ vào file 'tsp_algorithm_comparison.png'")
    
    print("\nHoàn thành!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "city.tsp"  # Mặc định
    
    run_tsp_solver(file_path)