import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import heapq
import random
import sys
import psutil
import os
from itertools import permutations

class TSPSolver:
    def __init__(self, distance_matrix, cities=None):
        """
        Khởi tạo bài toán TSP
        
        Parameters:
        - distance_matrix: Ma trận khoảng cách giữa các thành phố
        - cities: Danh sách tên các thành phố (nếu có)
        """
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        
        if cities is None:
            self.cities = [str(i) for i in range(self.num_cities)]
        else:
            self.cities = cities
            
        self.results = {}
        
    def read_tsp_file(file_path):
        """
        Đọc file TSP từ đường dẫn
        
        Parameters:
        - file_path: Đường dẫn đến file TSP
        
        Returns:
        - Một đối tượng TSPSolver với ma trận khoảng cách đã đọc
        """
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                
            # Tìm các thông tin cần thiết từ file
            dimension = None
            edge_weight_section = False
            distance_matrix = None
            city_names = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                if line.startswith("DIMENSION"):
                    dimension = int(line.split(":")[1].strip())
                    
                if line.startswith("EDGE_WEIGHT_SECTION") or line == "EDGE_WEIGHT_SECTION":
                    edge_weight_section = True
                    distance_matrix = np.zeros((dimension, dimension))
                    start_line = i + 1
                    break
            
            # Đọc ma trận khoảng cách
            if edge_weight_section:
                row = 0
                col = 0
                
                for i in range(start_line, len(lines)):
                    if lines[i].strip() == "EOF":
                        break
                    
                    values = lines[i].strip().split()
                    for val in values:
                        if val.strip():  # Skip empty values
                            distance_matrix[row][col] = float(val)
                            col += 1
                            if col == dimension:
                                col = 0
                                row += 1
                                if row == dimension:
                                    break
            
            return TSPSolver(distance_matrix)
        except Exception as e:
            print(f"Lỗi khi đọc file TSP: {e}")
            return None
    
    def _calculate_memory_usage(self):
        """Tính toán dung lượng bộ nhớ sử dụng"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Trả về MB
    
    def greedy_algorithm(self, start_city=0):
        """
        Thuật toán tham lam (Greedy Algorithm)
        
        Parameters:
        - start_city: Thành phố bắt đầu
        
        Returns:
        - path: Lộ trình thành phố
        - total_distance: Tổng khoảng cách
        - execution_time: Thời gian thực thi
        - memory_usage: Bộ nhớ sử dụng
        """
        start_memory = self._calculate_memory_usage()
        start_time = time.time()
        
        unvisited = set(range(self.num_cities))
        path = [start_city]
        unvisited.remove(start_city)
        total_distance = 0
        
        current_city = start_city
        
        while unvisited:
            # Tìm thành phố gần nhất chưa thăm
            nearest_city = min(unvisited, key=lambda city: self.distance_matrix[current_city][city])
            total_distance += self.distance_matrix[current_city][nearest_city]
            current_city = nearest_city
            path.append(current_city)
            unvisited.remove(current_city)
        
        # Quay về thành phố ban đầu
        total_distance += self.distance_matrix[current_city][start_city]
        path.append(start_city)
        
        end_time = time.time()
        end_memory = self._calculate_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Chuyển đổi path thành tên thành phố
        city_path = [self.cities[i] for i in path]
        
        self.results["Greedy"] = {
            "path": city_path,
            "total_distance": total_distance,
            "execution_time": execution_time,
            "memory_usage": memory_usage
        }
        
        # Lưu kết quả
        with open("tsp_results.txt", "a") as f:
            f.write(f"Greedy Algorithm: {city_path}\n")
            f.write(f"Tổng khoảng cách: {total_distance}\n")
            f.write(f"Thời gian thực thi: {execution_time} giây\n")
            f.write(f"Bộ nhớ sử dụng: {memory_usage} MB\n\n")
        
        return city_path, total_distance, execution_time, memory_usage
    
    def brute_force(self, start_city=0, max_cities=10):
        """
        Thuật toán vét cạn (Brute Force)
        
        Parameters:
        - start_city: Thành phố bắt đầu
        - max_cities: Số thành phố tối đa để chạy thuật toán
        
        Returns:
        - path: Lộ trình thành phố
        - total_distance: Tổng khoảng cách
        - execution_time: Thời gian thực thi
        - memory_usage: Bộ nhớ sử dụng
        """
        if self.num_cities > max_cities:
            print(f"Bỏ qua thuật toán Brute Force vì số thành phố ({self.num_cities}) > {max_cities}")
            self.results["Brute Force"] = {
                "path": None,
                "total_distance": float('inf'),
                "execution_time": float('inf'),
                "memory_usage": float('inf')
            }
            return None, float('inf'), float('inf'), float('inf')
        
        start_memory = self._calculate_memory_usage()
        start_time = time.time()
        
        # Tạo tất cả các hoán vị của các thành phố (trừ thành phố xuất phát)
        cities = list(range(self.num_cities))
        cities.remove(start_city)
        
        best_path = None
        best_distance = float('inf')
        
        for perm in permutations(cities):
            # Thêm thành phố xuất phát vào đầu và cuối
            current_path = [start_city] + list(perm) + [start_city]
            
            # Tính tổng khoảng cách
            current_distance = 0
            for i in range(len(current_path) - 1):
                current_distance += self.distance_matrix[current_path[i]][current_path[i+1]]
            
            # Cập nhật nếu tìm thấy lộ trình tốt hơn
            if current_distance < best_distance:
                best_distance = current_distance
                best_path = current_path
        
        end_time = time.time()
        end_memory = self._calculate_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Chuyển đổi path thành tên thành phố
        city_path = [self.cities[i] for i in best_path]
        
        self.results["Brute Force"] = {
            "path": city_path,
            "total_distance": best_distance,
            "execution_time": execution_time,
            "memory_usage": memory_usage
        }
        
        return city_path, best_distance, execution_time, memory_usage
    
    def approximation_algorithm(self, start_city=0):
        """
        Thuật toán xấp xỉ (MST-based Approximation Algorithm)
        
        Parameters:
        - start_city: Thành phố bắt đầu
        
        Returns:
        - path: Lộ trình thành phố
        - total_distance: Tổng khoảng cách
        - execution_time: Thời gian thực thi
        - memory_usage: Bộ nhớ sử dụng
        """
        start_memory = self._calculate_memory_usage()
        start_time = time.time()
        
        # Xây dựng MST (Minimum Spanning Tree) bằng thuật toán Prim
        mst = []
        visited = [False] * self.num_cities
        visited[start_city] = True
        
        # Thực hiện n-1 lần để thêm n-1 cạnh vào MST
        for _ in range(self.num_cities - 1):
            min_edge = (float('inf'), -1, -1)  # (weight, u, v)
            
            for u in range(self.num_cities):
                if visited[u]:
                    for v in range(self.num_cities):
                        if not visited[v] and self.distance_matrix[u][v] < min_edge[0]:
                            min_edge = (self.distance_matrix[u][v], u, v)
            
            _, u, v = min_edge
            mst.append((u, v))
            visited[v] = True
        
        # Xây dựng đồ thị từ MST
        graph = [[] for _ in range(self.num_cities)]
        for u, v in mst:
            graph[u].append(v)
            graph[v].append(u)
        
        # DFS để tạo đường đi
        path = []
        visited = [False] * self.num_cities
        
        def dfs(node):
            visited[node] = True
            path.append(node)
            for neighbor in graph[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        dfs(start_city)
        
        # Thêm thành phố đầu tiên để tạo chu trình
        path.append(start_city)
        
        # Tính tổng khoảng cách
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.distance_matrix[path[i]][path[i+1]]
        
        end_time = time.time()
        end_memory = self._calculate_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Chuyển đổi path thành tên thành phố
        city_path = [self.cities[i] for i in path]
        
        self.results["Approximation"] = {
            "path": city_path,
            "total_distance": total_distance,
            "execution_time": execution_time,
            "memory_usage": memory_usage
        }
        
        return city_path, total_distance, execution_time, memory_usage
    
    def ant_colony_optimization(self, start_city=0, n_ants=10, n_iterations=100, 
                           alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100, max_cities=50):
        """
        Thuật toán bầy kiến (Ant Colony Optimization)
        
        Parameters:
        - start_city: Thành phố bắt đầu
        - n_ants: Số lượng kiến
        - n_iterations: Số lần lặp
        - alpha: Hệ số ảnh hưởng của pheromone
        - beta: Hệ số ảnh hưởng của khoảng cách
        - evaporation_rate: Tỷ lệ bay hơi pheromone
        - Q: Hằng số lượng pheromone
        - max_cities: Số thành phố tối đa để chạy thuật toán
        
        Returns:
        - path: Lộ trình thành phố
        - total_distance: Tổng khoảng cách
        - execution_time: Thời gian thực thi
        - memory_usage: Bộ nhớ sử dụng
        """
        if self.num_cities > max_cities:
            print(f"Bỏ qua thuật toán Ant Colony vì số thành phố ({self.num_cities}) > {max_cities}")
            self.results["Ant Colony"] = {
                "path": None,
                "total_distance": float('inf'),
                "execution_time": float('inf'),
                "memory_usage": float('inf')
            }
            return None, float('inf'), float('inf'), float('inf')
        
        start_memory = self._calculate_memory_usage()
        start_time = time.time()
        
        # Khởi tạo pheromone
        pheromone = np.ones((self.num_cities, self.num_cities))
        
        # Tính toán ma trận visibility (1/khoảng cách)
        epsilon = 1e-10  # Để tránh chia cho 0
        visibility = 1 / (self.distance_matrix + epsilon)
        np.fill_diagonal(visibility, 0)  # Đặt đường chéo bằng 0
        
        # Lưu trữ lộ trình tốt nhất
        best_path = None
        best_distance = float('inf')
        
        for iteration in range(n_iterations):
            # Các lộ trình của các kiến trong lần lặp này
            all_paths = []
            all_distances = []
            
            # Cho mỗi con kiến tìm một lộ trình
            for ant in range(n_ants):
                current_city = start_city
                path = [current_city]
                visited = [False] * self.num_cities
                visited[current_city] = True
                tour_length = 0
                
                # Xây dựng lộ trình cho con kiến
                for _ in range(self.num_cities - 1):
                    # Tính xác suất chọn thành phố tiếp theo
                    probabilities = np.zeros(self.num_cities)
                    
                    for next_city in range(self.num_cities):
                        if not visited[next_city]:
                            # P(i,j) = [τ(i,j)]^α * [η(i,j)]^β / Σ [τ(i,k)]^α * [η(i,k)]^β
                            probabilities[next_city] = (pheromone[current_city][next_city] ** alpha) * \
                                                      (visibility[current_city][next_city] ** beta)
                    
                    # Chuẩn hóa xác suất
                    sum_prob = np.sum(probabilities)
                    if sum_prob > 0:
                        probabilities = probabilities / sum_prob
                    
                    # Chọn thành phố tiếp theo dựa trên xác suất
                    next_city = np.random.choice(range(self.num_cities), p=probabilities)
                    
                    # Cập nhật lộ trình
                    path.append(next_city)
                    visited[next_city] = True
                    tour_length += self.distance_matrix[current_city][next_city]
                    current_city = next_city
                
                # Hoàn thành chu trình bằng cách quay về thành phố ban đầu
                path.append(start_city)
                tour_length += self.distance_matrix[current_city][start_city]
                
                all_paths.append(path)
                all_distances.append(tour_length)
                
                # Cập nhật lộ trình tốt nhất
                if tour_length < best_distance:
                    best_distance = tour_length
                    best_path = path.copy()
            
            # Cập nhật pheromone
            # Bay hơi pheromone
            pheromone = (1 - evaporation_rate) * pheromone
            
            # Thêm pheromone mới
            for ant in range(n_ants):
                path = all_paths[ant]
                tour_length = all_distances[ant]
                
                for i in range(len(path) - 1):
                    pheromone[path[i]][path[i+1]] += Q / tour_length
                    pheromone[path[i+1]][path[i]] += Q / tour_length  # Đồ thị vô hướng
        
        end_time = time.time()
        end_memory = self._calculate_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Chuyển đổi path thành tên thành phố
        city_path = [self.cities[i] for i in best_path]
        
        self.results["Ant Colony"] = {
            "path": city_path,
            "total_distance": best_distance,
            "execution_time": execution_time,
            "memory_usage": memory_usage
        }
        
        return city_path, best_distance, execution_time, memory_usage
    
    def genetic_algorithm(self, start_city=0, population_size=50, n_generations=100, 
                      mutation_rate=0.01, max_cities=30):
        """
        Thuật toán di truyền (Genetic Algorithm)
        
        Parameters:
        - start_city: Thành phố bắt đầu
        - population_size: Kích thước quần thể
        - n_generations: Số thế hệ
        - mutation_rate: Tỷ lệ đột biến
        - max_cities: Số thành phố tối đa để chạy thuật toán
        
        Returns:
        - path: Lộ trình thành phố
        - total_distance: Tổng khoảng cách
        - execution_time: Thời gian thực thi
        - memory_usage: Bộ nhớ sử dụng
        """
        if self.num_cities > max_cities:
            print(f"Bỏ qua thuật toán Genetic vì số thành phố ({self.num_cities}) > {max_cities}")
            self.results["Genetic"] = {
                "path": None,
                "total_distance": float('inf'),
                "execution_time": float('inf'),
                "memory_usage": float('inf')
            }
            return None, float('inf'), float('inf'), float('inf')
        
        start_memory = self._calculate_memory_usage()
        start_time = time.time()
        
        # Tạo quần thể ban đầu
        cities = list(range(self.num_cities))
        cities.remove(start_city)  # Loại bỏ thành phố xuất phát
        population = []
        
        for _ in range(population_size):
            # Tạo một lộ trình ngẫu nhiên
            route = cities.copy()
            random.shuffle(route)
            # Thêm thành phố xuất phát vào đầu và cuối
            route = [start_city] + route + [start_city]
            population.append(route)
        
        # Hàm tính độ thích nghi (fitness) - nghịch đảo của tổng khoảng cách
        def calculate_fitness(route):
            total_distance = 0
            for i in range(len(route) - 1):
                total_distance += self.distance_matrix[route[i]][route[i+1]]
            return 1 / total_distance
        
        # Hàm lai ghép
        def crossover(parent1, parent2):
            # Loại bỏ thành phố đầu và cuối (start_city)
            p1 = parent1[1:-1]
            p2 = parent2[1:-1]
            
            # Chọn điểm cắt
            start = random.randint(0, len(p1) - 1)
            end = random.randint(start, len(p1) - 1)
            
            # Lấy đoạn từ parent1
            child = [-1] * len(p1)
            for i in range(start, end + 1):
                child[i] = p1[i]
            
            # Điền các thành phố còn lại từ parent2
            pointer = 0
            for city in p2:
                if city not in child:
                    while pointer < len(child) and child[pointer] != -1:
                        pointer += 1
                    if pointer < len(child):
                        child[pointer] = city
            
            # Thêm thành phố xuất phát vào đầu và cuối
            return [start_city] + child + [start_city]
        
        # Hàm đột biến
        def mutate(route, mutation_rate):
            # Loại bỏ thành phố đầu và cuối (start_city)
            route = route[1:-1]
            
            for i in range(len(route)):
                if random.random() < mutation_rate:
                    j = random.randint(0, len(route) - 1)
                    route[i], route[j] = route[j], route[i]
            
            # Thêm thành phố xuất phát vào đầu và cuối
            return [start_city] + route + [start_city]
        
        # Thuật toán di truyền chính
        best_route = None
        best_fitness = 0
        
        for generation in range(n_generations):
            # Tính độ thích nghi cho mỗi cá thể
            fitness_scores = [calculate_fitness(route) for route in population]
            total_fitness = sum(fitness_scores)
            
            # Tìm cá thể tốt nhất trong thế hệ hiện tại
            best_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_route = population[best_idx]
            
            # Chọn các cá thể để lai ghép
            selected_indices = random.choices(
                range(population_size), 
                weights=fitness_scores, 
                k=population_size
            )
            selected = [population[i] for i in selected_indices]
            
            # Tạo thế hệ mới
            new_population = []
            
            # Ưu tú hóa: giữ lại cá thể tốt nhất
            new_population.append(population[best_idx])
            
            # Lai ghép và đột biến
            for i in range(1, population_size):
                # Chọn cha mẹ
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                # Lai ghép
                child = crossover(parent1, parent2)
                
                # Đột biến
                child = mutate(child, mutation_rate)
                
                new_population.append(child)
            
            # Cập nhật quần thể
            population = new_population
        
        # Tính tổng khoảng cách của lộ trình tốt nhất
        best_distance = 0
        for i in range(len(best_route) - 1):
            best_distance += self.distance_matrix[best_route[i]][best_route[i+1]]
        
        end_time = time.time()
        end_memory = self._calculate_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Chuyển đổi path thành tên thành phố
        city_path = [self.cities[i] for i in best_route]
        
        self.results["Genetic"] = {
            "path": city_path,
            "total_distance": best_distance,
            "execution_time": execution_time,
            "memory_usage": memory_usage
        }
        
        return city_path, best_distance, execution_time, memory_usage
    
    def nearest_neighbor(self, start_city=0):
        """
        Thuật toán láng giềng gần nhất (Nearest Neighbor)
        
        Parameters:
        - start_city: Thành phố bắt đầu
        
        Returns:
        - path: Lộ trình thành phố
        - total_distance: Tổng khoảng cách
        - execution_time: Thời gian thực thi
        - memory_usage: Bộ nhớ sử dụng
        """
        start_memory = self._calculate_memory_usage()
        start_time = time.time()
        
        path = [start_city]
        unvisited = set(range(self.num_cities))
        unvisited.remove(start_city)
        current_city = start_city
        total_distance = 0
        
        while unvisited:
            # Tìm thành phố gần nhất chưa thăm
            nearest_city = min(unvisited, key=lambda city: self.distance_matrix[current_city][city])
            path.append(nearest_city)
            total_distance += self.distance_matrix[current_city][nearest_city]
            unvisited.remove(nearest_city)
            current_city = nearest_city
        
        # Quay về thành phố ban đầu
        path.append(start_city)
        total_distance += self.distance_matrix[current_city][start_city]
        
        end_time = time.time()
        end_memory = self._calculate_memory_usage()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Chuyển đổi path thành tên thành phố
        city_path = [self.cities[i] for i in path]
        
        self.results["Nearest Neighbor"] = {
            "path": city_path,
            "total_distance": total_distance,
            "execution_time": execution_time,
            "memory_usage": memory_usage
        }
        
        return city_path, total_distance, execution_time, memory_usage
    
    def visualize_results(self):
        """Trực quan hóa kết quả của các thuật toán"""
        if not self.results:
            print("Không có kết quả để trực quan hóa.")
            return
        
        # Chuẩn bị dữ liệu
        algorithms = []
        execution_times = []
        memory_usages = []
        distances = []
        
        for algo, result in self.results.items():
            if result["total_distance"] != float('inf'):
                algorithms.append(algo)
                execution_times.append(result["execution_time"])
                memory_usages.append(result["memory_usage"])
                distances.append(result["total_distance"])
        
        # Tạo bảng kết quả
        results_df = pd.DataFrame({
            'Thuật toán': algorithms,
            'Thời gian thực thi (s)': execution_times,
            'Bộ nhớ sử dụng (MB)': memory_usages,
            'Tổng khoảng cách': distances
        })
        
        print("Bảng kết quả:")
        print(results_df)
        
        # Vẽ biểu đồ thời gian thực thi
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 3, 1)
        plt.bar(algorithms, execution_times, color='blue')
        plt.title('Thời gian thực thi')
        plt.ylabel('Thời gian (giây)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Vẽ biểu đồ bộ nhớ sử dụng
        plt.subplot(1, 3, 2)
        plt.bar(algorithms, memory_usages, color='green')
        plt.title('Bộ nhớ sử dụng')
        plt.ylabel('Bộ nhớ (MB)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Vẽ biểu đồ tổng khoảng cách
        plt.subplot(1, 3, 3)
        plt.bar(algorithms, distances, color='red')
        plt.title('Tổng khoảng cách')
        plt.ylabel('Khoảng cách')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig('tsp_results.png')
        print("Đã lưu biểu đồ kết quả vào file 'tsp_results.png'")
        plt.show()

def main(file_path="city.tsp"):
    # Đọc file TSP
    tsp_solver = TSPSolver.read_tsp_file(file_path)
    
    if tsp_solver is None:
        print("Không thể đọc file TSP.")
        return
    
    print(f"Đã đọc file TSP với {tsp_solver.num_cities} thành phố.")
    
    # Thực hiện các thuật toán
    print("\n1. Thuật toán tham lam (Greedy):")
    path, total_distance, execution_time, memory_usage = tsp_solver.greedy_algorithm()
    print(f"Lộ trình: {path}")
    print(f"Tổng khoảng cách: {total_distance}")
    print(f"Thời gian thực thi: {execution_time} giây")
    print(f"Bộ nhớ sử dụng: {memory_usage} MB")
    
    print("\n2. Thuật toán vét cạn (Brute Force):")
    path, total_distance, execution_time, memory_usage = tsp_solver.brute_force(max_cities=10)
    if path:
        print(f"Lộ trình: {path}")
        print(f"Tổng khoảng cách: {total_distance}")
        print(f"Thời gian thực thi: {execution_time} giây")
        print(f"Bộ nhớ sử dụng: {memory_usage} MB")
    
    print("\n3. Thuật toán xấp xỉ (Approximation):")
    path, total_distance, execution_time, memory_usage = tsp_solver.approximation_algorithm()
    print(f"Lộ trình: {path}")
    print(f"Tổng khoảng cách: {total_distance}")
    print(f"Thời gian thực thi: {execution_time} giây")
    print(f"Bộ nhớ sử dụng: {memory_usage} MB")
    
    print("\n4. Thuật toán bầy kiến (Ant Colony):")
    path, total_distance, execution_time, memory_usage = tsp_solver.ant_colony_optimization(max_cities=30)
    if path:
        print(f"Lộ trình: {path}")
        print(f"Tổng khoảng cách: {total_distance}")
        print(f"Thời gian thực thi: {execution_time} giây")
        print(f"Bộ nhớ sử dụng: {memory_usage} MB")
    
    print("\n5. Thuật toán di truyền (Genetic):")
    path, total_distance, execution_time, memory_usage = tsp_solver.genetic_algorithm(max_cities=30)
    if path:
        print(f"Lộ trình: {path}")
        print(f"Tổng khoảng cách: {total_distance}")
        print(f"Thời gian thực thi: {execution_time} giây")
        print(f"Bộ nhớ sử dụng: {memory_usage} MB")
    
    print("\n6. Thuật toán láng giềng gần nhất (Nearest Neighbor):")
    path, total_distance, execution_time, memory_usage = tsp_solver.nearest_neighbor()
    print(f"Lộ trình: {path}")
    print(f"Tổng khoảng cách: {total_distance}")
    print(f"Thời gian thực thi: {execution_time} giây")
    print(f"Bộ nhớ sử dụng: {memory_usage} MB")
    
    # Trực quan hóa kết quả
    print("\nTrực quan hóa kết quả:")
    tsp_solver.visualize_results()

if __name__ == "__main__":
    main("city.tsp")