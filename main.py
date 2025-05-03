import numpy as np
import time
import matplotlib.pyplot as plt
import psutil
import random
import itertools
import copy
from typing import List, Tuple


class TSPSolver:
    def __init__(self, distance_matrix, city_count):
        self.distance_matrix = distance_matrix
        self.city_count = city_count
        self.city_names = [str(i) for i in range(city_count)]
        
    def read_tsp_file(file_path):
        """Đọc file TSP và trả về ma trận khoảng cách"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Tìm số thành phố
        dimension_line = [line for line in lines if line.strip().startswith("DIMENSION")][0]
        city_count = int(dimension_line.split(":")[1].strip())
        
        # Tìm vị trí bắt đầu của ma trận khoảng cách
        start_idx = lines.index("EDGE_WEIGHT_SECTION\n") + 1
        
        # Đọc ma trận khoảng cách dạng LOWER_DIAG_ROW
        dist_matrix = np.zeros((city_count, city_count))
        row_idx = 0
        col_idx = 0
        
        for line in lines[start_idx:]:
            if line.strip() == "EOF":
                break
                
            values = [int(val) for val in line.strip().split()]
            for val in values:
                dist_matrix[row_idx][col_idx] = val
                dist_matrix[col_idx][row_idx] = val  # Đối xứng
                col_idx += 1
                if col_idx > row_idx:
                    row_idx += 1
                    col_idx = 0
        
        return dist_matrix, city_count
    
    def calculate_path_distance(self, path):
        """Tính tổng khoảng cách của một lộ trình"""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.distance_matrix[path[i]][path[i+1]]
        
        # Thêm khoảng cách từ thành phố cuối về thành phố đầu
        total_distance += self.distance_matrix[path[-1]][path[0]]
        return total_distance
        
    def greedy_algorithm(self):
        """Thuật toán tham lam"""
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Bắt đầu từ thành phố 0
        current_city = 0
        tour = [current_city]
        unvisited = set(range(1, self.city_count))
        
        while unvisited:
            # Tìm thành phố gần nhất chưa thăm
            next_city = min(unvisited, key=lambda city: self.distance_matrix[current_city][city])
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
            
        execution_time = time.time() - start_time
        memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - memory_before
        
        distance = self.calculate_path_distance(tour)
        return {
            "algorithm": "Greedy",
            "path": tour,
            "distance": distance,
            "execution_time": execution_time,
            "memory_used": memory_used
        }
        
    def nearest_neighbor(self):
        """Thuật toán láng giềng gần nhất"""
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        best_tour = None
        best_distance = float('inf')
        
        # Thử với mỗi thành phố làm điểm xuất phát
        for start_city in range(self.city_count):
            current_city = start_city
            tour = [current_city]
            unvisited = set(range(self.city_count))
            unvisited.remove(current_city)
            
            while unvisited:
                # Tìm thành phố gần nhất chưa thăm
                next_city = min(unvisited, key=lambda city: self.distance_matrix[current_city][city])
                tour.append(next_city)
                unvisited.remove(next_city)
                current_city = next_city
                
            distance = self.calculate_path_distance(tour)
            if distance < best_distance:
                best_distance = distance
                best_tour = tour
                
        execution_time = time.time() - start_time
        memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - memory_before
        
        return {
            "algorithm": "Nearest Neighbor",
            "path": best_tour,
            "distance": best_distance,
            "execution_time": execution_time,
            "memory_used": memory_used
        }
        
    def brute_force(self):
        """Thuật toán brute force (chỉ sử dụng cho n <= 11)"""
        if self.city_count > 11:
            return {
                "algorithm": "Brute Force",
                "path": None,
                "distance": None,
                "execution_time": None,
                "memory_used": None,
                "skipped": True
            }
            
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Tạo tất cả hoán vị có thể
        cities = list(range(1, self.city_count))
        best_tour = None
        best_distance = float('inf')
        
        for perm in itertools.permutations(cities):
            # Luôn bắt đầu từ thành phố 0
            tour = [0] + list(perm)
            distance = self.calculate_path_distance(tour)
            
            if distance < best_distance:
                best_distance = distance
                best_tour = tour
                
        execution_time = time.time() - start_time
        memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - memory_before
        
        return {
            "algorithm": "Brute Force",
            "path": best_tour,
            "distance": best_distance,
            "execution_time": execution_time,
            "memory_used": memory_used
        }
    
    def approximation_algorithm(self):
        """Thuật toán xấp xỉ 2-approximation sử dụng cây khung nhỏ nhất"""
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Khởi tạo graph và MST
        vertices = list(range(self.city_count))
        edges = []
        
        # Tạo danh sách các cạnh
        for i in range(self.city_count):
            for j in range(i+1, self.city_count):
                edges.append((i, j, self.distance_matrix[i][j]))
        
        # Sắp xếp các cạnh theo trọng số tăng dần
        edges.sort(key=lambda x: x[2])
        
        # Thuật toán Kruskal để tạo MST
        parent = list(range(self.city_count))
        rank = [0] * self.city_count
        
        def find(vertex):
            if parent[vertex] != vertex:
                parent[vertex] = find(parent[vertex])
            return parent[vertex]
        
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            
            if root_x == root_y:
                return
            
            if rank[root_x] < rank[root_y]:
                parent[root_x] = root_y
            else:
                parent[root_y] = root_x
                if rank[root_x] == rank[root_y]:
                    rank[root_x] += 1
        
        mst_edges = []
        for u, v, w in edges:
            if find(u) != find(v):
                union(u, v)
                mst_edges.append((u, v))
        
        # Tạo đồ thị từ MST
        graph = {i: [] for i in range(self.city_count)}
        for u, v in mst_edges:
            graph[u].append(v)
            graph[v].append(u)
        
        # DFS để tạo tour
        visited = [False] * self.city_count
        tour = []
        
        def dfs(vertex):
            visited[vertex] = True
            tour.append(vertex)
            
            for neighbor in graph[vertex]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        dfs(0)  # Bắt đầu từ thành phố 0
        
        execution_time = time.time() - start_time
        memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - memory_before
        
        distance = self.calculate_path_distance(tour)
        return {
            "algorithm": "Approximation",
            "path": tour,
            "distance": distance,
            "execution_time": execution_time,
            "memory_used": memory_used
        }
    
    def ant_colony(self, num_ants=10, num_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100):
        """Thuật toán bầy kiến - Ant Colony Optimization"""
        if self.city_count > 300:
            return {
                "algorithm": "Ant Colony",
                "path": None,
                "distance": None,
                "execution_time": None,
                "memory_used": None,
                "skipped": True
            }
            
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Khởi tạo ma trận pheromone
        pheromone = np.ones((self.city_count, self.city_count)) / self.city_count
        
        # Tính ma trận visibility (nghịch đảo của khoảng cách)
        visibility = np.zeros((self.city_count, self.city_count))
        for i in range(self.city_count):
            for j in range(self.city_count):
                if i != j and self.distance_matrix[i][j] > 0:
                    visibility[i][j] = 1.0 / self.distance_matrix[i][j]
        
        # Lưu tour tốt nhất
        best_tour = None
        best_distance = float('inf')
        
        for iteration in range(num_iterations):
            # Mỗi kiến xây dựng một tour
            all_tours = []
            all_distances = []
            
            for ant in range(num_ants):
                # Chọn thành phố bắt đầu ngẫu nhiên
                current_city = random.randint(0, self.city_count - 1)
                tour = [current_city]
                unvisited = set(range(self.city_count))
                unvisited.remove(current_city)
                
                # Xây dựng tour đầy đủ
                while unvisited:
                    # Tính xác suất chọn thành phố tiếp theo
                    probabilities = []
                    total = 0.0
                    
                    for city in unvisited:
                        prob = (pheromone[current_city][city] ** alpha) * (visibility[current_city][city] ** beta)
                        probabilities.append((city, prob))
                        total += prob
                    
                    # Chuẩn hóa xác suất
                    if total > 0:
                        probabilities = [(city, prob/total) for city, prob in probabilities]
                    else:
                        # Nếu tất cả đều không có pheromone, chọn ngẫu nhiên
                        next_city = random.choice(list(unvisited))
                        tour.append(next_city)
                        unvisited.remove(next_city)
                        current_city = next_city
                        continue
                        
                    # Chọn thành phố tiếp theo theo xác suất
                    r = random.random()
                    cum_prob = 0.0
                    for city, prob in probabilities:
                        cum_prob += prob
                        if r <= cum_prob:
                            next_city = city
                            break
                    
                    tour.append(next_city)
                    unvisited.remove(next_city)
                    current_city = next_city
                
                # Tính khoảng cách tour
                distance = self.calculate_path_distance(tour)
                all_tours.append(tour)
                all_distances.append(distance)
                
                # Cập nhật tour tốt nhất
                if distance < best_distance:
                    best_distance = distance
                    best_tour = tour
            
            # Cập nhật pheromone
            pheromone *= (1 - evaporation_rate)  # Bay hơi
            
            # Thêm pheromone dựa trên chất lượng các tour
            for tour, distance in zip(all_tours, all_distances):
                for i in range(len(tour) - 1):
                    pheromone[tour[i]][tour[i+1]] += Q / distance
                    pheromone[tour[i+1]][tour[i]] += Q / distance  # Đối xứng
                
                # Thêm pheromone cho cạnh cuối-đầu
                pheromone[tour[-1]][tour[0]] += Q / distance
                pheromone[tour[0]][tour[-1]] += Q / distance
        
        execution_time = time.time() - start_time
        memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - memory_before
        
        return {
            "algorithm": "Ant Colony",
            "path": best_tour,
            "distance": best_distance,
            "execution_time": execution_time,
            "memory_used": memory_used
        }
    
    def genetic_algorithm(self, population_size=50, num_generations=100, mutation_rate=0.01):
        """Thuật toán di truyền"""
        if self.city_count > 500:
            return {
                "algorithm": "Genetic",
                "path": None,
                "distance": None,
                "execution_time": None,
                "memory_used": None,
                "skipped": True
            }
            
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Tạo quần thể ban đầu
        population = []
        for _ in range(population_size):
            # Mỗi cá thể là một tour ngẫu nhiên bắt đầu từ thành phố 0
            individual = [0] + random.sample(range(1, self.city_count), self.city_count - 1)
            population.append(individual)
        
        def fitness(tour):
            return 1.0 / self.calculate_path_distance(tour)  # Fitness cao = khoảng cách ngắn
        
        def crossover(parent1, parent2):
            # Phương pháp Ordered Crossover (OX)
            size = len(parent1)
            child = [-1] * size
            child[0] = 0  # Luôn bắt đầu từ 0
            
            # Chọn đoạn từ parent1
            start, end = sorted(random.sample(range(1, size), 2))
            for i in range(start, end + 1):
                child[i] = parent1[i]
            
            # Điền các phần tử còn lại từ parent2
            j = 1
            for i in range(1, size):
                if child[i] == -1:
                    while parent2[j] in child:
                        j += 1
                        if j >= size:
                            j = 1
                    child[i] = parent2[j]
                    j += 1
                    if j >= size:
                        j = 1
            
            return child
        
        def mutate(tour):
            # Đảo ngẫu nhiên hai vị trí (không bao gồm thành phố đầu tiên)
            if random.random() < mutation_rate:
                i, j = random.sample(range(1, len(tour)), 2)
                tour[i], tour[j] = tour[j], tour[i]
            return tour
        
        best_tour = None
        best_distance = float('inf')
        
        for generation in range(num_generations):
            # Đánh giá quần thể
            fitness_scores = [fitness(ind) for ind in population]
            total_fitness = sum(fitness_scores)
            
            # Chọn cá thể để sinh sản (selection)
            selected = []
            for _ in range(population_size):
                r = random.uniform(0, total_fitness)
                cum_sum = 0
                for i, fit in enumerate(fitness_scores):
                    cum_sum += fit
                    if cum_sum >= r:
                        selected.append(copy.deepcopy(population[i]))
                        break
            
            # Tạo thế hệ mới thông qua lai ghép và đột biến
            new_population = []
            while len(new_population) < population_size:
                # Chọn cha mẹ
                parent1, parent2 = random.sample(selected, 2)
                
                # Lai ghép
                child = crossover(parent1, parent2)
                
                # Đột biến
                child = mutate(child)
                
                new_population.append(child)
            
            # Cập nhật quần thể
            population = new_population
            
            # Tìm tour tốt nhất trong thế hệ hiện tại
            for tour in population:
                distance = self.calculate_path_distance(tour)
                if distance < best_distance:
                    best_distance = distance
                    best_tour = tour
        
        execution_time = time.time() - start_time
        memory_used = psutil.Process().memory_info().rss / 1024 / 1024 - memory_before
        
        return {
            "algorithm": "Genetic",
            "path": best_tour,
            "distance": best_distance,
            "execution_time": execution_time,
            "memory_used": memory_used
        }

def run_experiments(file_path, n_values=None):
    """Chạy các thuật toán với các giá trị n khác nhau"""
    if n_values is None:
        n_values = [10, 20, 50, 150, 300, 1000, 10000]
    
    all_results = {}
    
    for n in n_values:
        print(f"\nRunning experiments for n = {n}")
        
        if n <= 10:
            # Sử dụng ma trận khoảng cách từ file
            distance_matrix, city_count = TSPSolver.read_tsp_file(file_path)
        else:
            # Tạo ma trận khoảng cách ngẫu nhiên cho n > 10
            distance_matrix = np.random.randint(1, 100, size=(n, n))
            np.fill_diagonal(distance_matrix, 0)  # Khoảng cách từ thành phố tới chính nó = 0
            distance_matrix = (distance_matrix + distance_matrix.T) // 2  # Đảm bảo tính đối xứng
            city_count = n
            
        solver = TSPSolver(distance_matrix, city_count)
        
        algorithms = [
            ("Greedy", solver.greedy_algorithm),
            ("Brute Force", solver.brute_force),
            ("Approximation", solver.approximation_algorithm),
            ("Ant Colony", solver.ant_colony),
            ("Genetic", solver.genetic_algorithm),
            ("Nearest Neighbor", solver.nearest_neighbor)
        ]
        
        results = []
        
        for name, algorithm in algorithms:
            print(f"Running {name} algorithm...")
            
            try:
                result = algorithm()
                results.append(result)
                
                if result.get("skipped"):
                    print(f"{name} algorithm was skipped (n = {n} too large)")
                else:
                    print(f"{name} algorithm completed in {result['execution_time']:.6f} seconds")
                    print(f"Path: {result['path']}")
                    print(f"Distance: {result['distance']}")
                    print(f"Memory used: {result['memory_used']:.6f} MB")
            except Exception as e:
                print(f"Error running {name} algorithm: {str(e)}")
                results.append({
                    "algorithm": name,
                    "path": None,
                    "distance": None,
                    "execution_time": None,
                    "memory_used": None,
                    "error": str(e)
                })
        
        all_results[n] = results
        
        # Visualize results for this n
        visualize_performance(results, n)
    
    return all_results

def visualize_performance(results, n):
    """Trực quan hóa hiệu suất của các thuật toán"""
    # Lọc các kết quả có dữ liệu đầy đủ
    valid_results = [r for r in results if r.get("execution_time") is not None and not r.get("skipped")]
    
    if not valid_results:
        print(f"No valid results to visualize for n = {n}")
        return
    
    algorithms = [r["algorithm"] for r in valid_results]
    execution_times = [r["execution_time"] for r in valid_results]
    memory_used = [r["memory_used"] for r in valid_results]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Đồ thị cột cho thời gian thực thi
    x = np.arange(len(algorithms))
    width = 0.35
    rects = ax1.bar(x, execution_times, width, label='Execution Time (s)', color='skyblue')
    ax1.set_ylabel('Execution Time (seconds)', color='blue')
    ax1.set_xlabel('Algorithm')
    ax1.set_title(f'Algorithm Performance Comparison (n = {n})')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Đồ thị đường cho bộ nhớ sử dụng
    ax2 = ax1.twinx()
    ax2.plot(x, memory_used, 'ro-', linewidth=2, markersize=8, label='Memory Used (MB)')
    ax2.set_ylabel('Memory Used (MB)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Thêm nhãn dữ liệu
    for i, v in enumerate(execution_times):
        ax1.text(i, v + 0.01, f'{v:.3f}s', ha='center', va='bottom', color='blue', fontweight='bold')
    
    for i, v in enumerate(memory_used):
        ax2.text(i, v + 0.05, f'{v:.2f}MB', ha='center', va='bottom', color='red', fontweight='bold')
    
    # Thêm legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'tsp_performance_n{n}.png')
    plt.close()
    
    # Trực quan hóa khoảng cách tìm được
    if all(r.get("distance") is not None for r in valid_results):
        distances = [r["distance"] for r in valid_results]
        
        plt.figure(figsize=(10, 6))
        plt.bar(algorithms, distances, color='green')
        plt.ylabel('Total Distance')
        plt.xlabel('Algorithm')
        plt.title(f'Total Distance Comparison (n = {n})')
        plt.xticks(rotation=45, ha='right')
        
        # Thêm nhãn dữ liệu
        for i, v in enumerate(distances):
            plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', color='black')
        
        plt.tight_layout()
        plt.savefig(f'tsp_distance_n{n}.png')
        plt.close()

def main():
    file_path = "city.tsp"
    
    # Chạy với n nhỏ để thử nghiệm
    n_values_to_test = [10, 20, 50, 150, 500, 1000, 10000]  # Có thể thêm các giá trị lớn hơn tùy theo khả năng máy tính
    
    # Chạy thử nghiệm
    results = run_experiments(file_path, n_values_to_test)
    
    print("\n===== Kết quả tóm tắt =====")
    for n, n_results in results.items():
        print(f"\nKết quả cho n = {n}:")
        for result in n_results:
            if result.get("skipped"):
                print(f"  {result['algorithm']}: Bỏ qua (n quá lớn)")
            elif result.get("error"):
                print(f"  {result['algorithm']}: Lỗi - {result['error']}")
            else:
                print(f"  {result['algorithm']}: Khoảng cách = {result['distance']}, Thời gian = {result['execution_time']:.6f}s, Bộ nhớ = {result['memory_used']:.2f}MB")

if __name__ == "__main__":
    main()