import numpy as np
import time
import matplotlib.pyplot as plt
import random
import sys
import psutil
import itertools
from memory_profiler import memory_usage
import warnings
warnings.filterwarnings('ignore')

# Đọc file TSP
def read_tsp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Tìm số thành phố
    for line in lines:
        if "DIMENSION" in line:
            num_cities = int(line.split(":")[1].strip())
            break
    
    # Tìm phần dữ liệu khoảng cách
    start_index = lines.index("EDGE_WEIGHT_SECTION\n") + 1
    distance_data = []
    for i in range(start_index, start_index + num_cities * (num_cities + 1) // 2):
        if "EOF" in lines[i]:
            break
        distance_data.extend([int(val) for val in lines[i].strip().split()])
    
    # Chuyển đổi dữ liệu thành ma trận khoảng cách
    dist_matrix = np.zeros((num_cities, num_cities))
    index = 0
    for i in range(num_cities):
        for j in range(i + 1):
            dist_matrix[i][j] = distance_data[index]
            dist_matrix[j][i] = distance_data[index]  # Ma trận đối xứng
            index += 1
    
    return dist_matrix

# 1. Thuật toán tham lam (Greedy Algorithm)
def greedy_algorithm(dist_matrix):
    n = len(dist_matrix)
    start_time = time.time()
    
    # Bắt đầu từ thành phố 0
    current_city = 0
    tour = [current_city]
    unvisited = set(range(1, n))
    
    # Chọn thành phố gần nhất ở mỗi bước
    while unvisited:
        next_city = min(unvisited, key=lambda city: dist_matrix[current_city][city])
        tour.append(next_city)
        unvisited.remove(next_city)
        current_city = next_city
    
    # Quay về thành phố đầu tiên
    tour.append(0)
    
    # Tính tổng khoảng cách
    total_distance = sum(dist_matrix[tour[i]][tour[i+1]] for i in range(n))
    
    execution_time = time.time() - start_time
    
    return tour, total_distance, execution_time

# 2. Thuật toán brute force (kiểm tra mọi hoán vị)
def brute_force(dist_matrix):
    n = len(dist_matrix)
    
    # Kiểm tra kích thước, nếu quá lớn thì không thực hiện
    if n > 11:
        return None, None, None
    
    start_time = time.time()
    
    # Sinh tất cả các hoán vị có thể
    cities = list(range(1, n))  # Không bao gồm thành phố 0 (bắt đầu và kết thúc)
    best_tour = None
    min_distance = float('inf')
    
    for perm in itertools.permutations(cities):
        # Thêm thành phố bắt đầu và kết thúc
        tour = [0] + list(perm) + [0]
        
        # Tính tổng khoảng cách
        distance = sum(dist_matrix[tour[i]][tour[i+1]] for i in range(n))
        
        if distance < min_distance:
            min_distance = distance
            best_tour = tour
    
    execution_time = time.time() - start_time
    
    return best_tour, min_distance, execution_time

# 3. Thuật toán xấp xỉ (Approximation Algorithm - 2-opt)
def two_opt(route, dist_matrix):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                if j - i == 1:
                    continue
                new_route = route[:]
                new_route[i:j+1] = route[j:i-1:-1]
                
                # Tính toán sự thay đổi khoảng cách
                old_distance = dist_matrix[route[i-1]][route[i]] + dist_matrix[route[j]][route[j+1]]
                new_distance = dist_matrix[route[i-1]][route[j]] + dist_matrix[route[i]][route[j+1]]
                
                if new_distance < old_distance:
                    best = new_route
                    improved = True
                    route = best
                    break
            if improved:
                break
    return best

def approximation_algorithm(dist_matrix):
    n = len(dist_matrix)
    start_time = time.time()
    
    # Bắt đầu với tour từ thuật toán tham lam
    tour, _, _ = greedy_algorithm(dist_matrix)
    
    # Cải thiện tour bằng 2-opt
    tour = two_opt(tour, dist_matrix)
    
    # Tính tổng khoảng cách
    total_distance = sum(dist_matrix[tour[i]][tour[i+1]] for i in range(n))
    
    execution_time = time.time() - start_time
    
    return tour, total_distance, execution_time

# 4. Thuật toán bầy kiến (Ant Colony Optimization)
def ant_colony_optimization(dist_matrix, n_ants=10, n_iterations=50, decay=0.5, alpha=1, beta=2):
    n = len(dist_matrix)
    
    # Kiểm tra kích thước, nếu quá lớn thì giảm số lượng kiến và số lần lặp
    if n > 20:
        n_ants = min(n_ants, 5)
        n_iterations = min(n_iterations, 20)
    
    start_time = time.time()
    
    # Khởi tạo ma trận pheromone
    pheromone = np.ones((n, n))
    best_tour = None
    best_distance = float('inf')
    
    for iteration in range(n_iterations):
        all_tours = []
        all_distances = []
        
        # Mỗi kiến tạo một tour
        for ant in range(n_ants):
            # Bắt đầu từ thành phố ngẫu nhiên
            current_city = random.randint(0, n-1)
            tour = [current_city]
            unvisited = set(range(n))
            unvisited.remove(current_city)
            
            # Xây dựng tour
            while unvisited:
                # Tính xác suất chọn mỗi thành phố
                probabilities = []
                for city in unvisited:
                    # Công thức ACO: tau^alpha * eta^beta
                    # tau: pheromone, eta: 1/khoảng cách (độ hấp dẫn)
                    tau = pheromone[current_city][city]
                    eta = 1.0 / (dist_matrix[current_city][city] + 1)  # Tránh chia cho 0
                    probability = (tau ** alpha) * (eta ** beta)
                    probabilities.append((city, probability))
                
                # Chọn thành phố tiếp theo dựa trên xác suất
                total = sum(prob for _, prob in probabilities)
                r = random.random() * total
                
                running_total = 0
                next_city = None
                for city, prob in probabilities:
                    running_total += prob
                    if running_total >= r:
                        next_city = city
                        break
                
                if next_city is None:  # Đề phòng lỗi số học
                    next_city = list(unvisited)[0]
                
                tour.append(next_city)
                unvisited.remove(next_city)
                current_city = next_city
            
            # Quay về thành phố đầu tiên
            tour.append(tour[0])
            
            # Tính tổng khoảng cách
            distance = sum(dist_matrix[tour[i]][tour[i+1]] for i in range(n))
            
            all_tours.append(tour)
            all_distances.append(distance)
            
            # Cập nhật tour tốt nhất
            if distance < best_distance:
                best_distance = distance
                best_tour = tour
        
        # Cập nhật pheromone
        pheromone *= decay  # Bay hơi
        
        # Thêm pheromone dựa trên chất lượng của tour
        for tour, distance in zip(all_tours, all_distances):
            amount = 1.0 / distance
            for i in range(n):
                pheromone[tour[i]][tour[i+1]] += amount
    
    execution_time = time.time() - start_time
    
    return best_tour, best_distance, execution_time

# 5. Thuật toán di truyền (Genetic Algorithm)
def genetic_algorithm(dist_matrix, pop_size=50, n_generations=100, mutation_rate=0.2):
    n = len(dist_matrix)
    
    # Kiểm tra kích thước, nếu quá lớn thì giảm kích thước quần thể và số thế hệ
    if n > 20:
        pop_size = min(pop_size, 20)
        n_generations = min(n_generations, 40)
    
    start_time = time.time()
    
    # Hàm tính toán độ thích nghi (fitness)
    def calculate_fitness(route):
        return 1.0 / (sum(dist_matrix[route[i]][route[i+1]] for i in range(n-1)) + dist_matrix[route[-1]][route[0]])
    
    # Khởi tạo quần thể
    population = []
    for _ in range(pop_size):
        route = list(range(n))
        random.shuffle(route)
        population.append(route)
    
    best_route = None
    best_fitness = -1
    
    for _ in range(n_generations):
        # Tính toán độ thích nghi
        fitness_scores = [calculate_fitness(route) for route in population]
        
        # Lưu cá thể tốt nhất
        max_fitness_idx = fitness_scores.index(max(fitness_scores))
        if fitness_scores[max_fitness_idx] > best_fitness:
            best_fitness = fitness_scores[max_fitness_idx]
            best_route = population[max_fitness_idx][:]
        
        # Chọn lọc (selection) - chọn các cá thể theo xác suất tỷ lệ với độ thích nghi
        total_fitness = sum(fitness_scores)
        selection_probs = [score/total_fitness for score in fitness_scores]
        
        # Tạo thế hệ mới
        new_population = []
        
        # Giữ lại một số cá thể tốt nhất (elitism)
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        elite_size = max(1, int(pop_size * 0.1))
        for i in range(elite_size):
            new_population.append(population[sorted_indices[i]])
        
        # Lai tạo (crossover) và đột biến (mutation)
        while len(new_population) < pop_size:
            # Chọn cha mẹ
            parent1 = random.choices(population, weights=selection_probs)[0]
            parent2 = random.choices(population, weights=selection_probs)[0]
            
            # Lai tạo (Ordered Crossover - OX)
            start, end = sorted(random.sample(range(n), 2))
            child = [-1] * n
            
            # Sao chép một đoạn từ parent1
            for i in range(start, end + 1):
                child[i] = parent1[i]
            
            # Điền các giá trị còn lại từ parent2 theo thứ tự
            j = 0
            for i in range(n):
                if parent2[i] not in child:
                    while j < n and child[j] != -1:
                        j += 1
                    if j < n:
                        child[j] = parent2[i]
            
            # Đột biến (swap mutation)
            if random.random() < mutation_rate:
                idx1, idx2 = random.sample(range(n), 2)
                child[idx1], child[idx2] = child[idx2], child[idx1]
            
            new_population.append(child)
        
        population = new_population
    
    # Chuyển đổi best_route thành tour khép kín
    best_tour = best_route + [best_route[0]]
    best_distance = 1.0 / best_fitness
    
    execution_time = time.time() - start_time
    
    return best_tour, best_distance, execution_time

# 6. Thuật toán láng giềng gần nhất (Nearest Neighbor)
def nearest_neighbor(dist_matrix):
    n = len(dist_matrix)
    start_time = time.time()
    
    # Thử bắt đầu từ mỗi thành phố và chọn tour tốt nhất
    best_tour = None
    best_distance = float('inf')
    
    for start_city in range(n):
        current_city = start_city
        tour = [current_city]
        unvisited = set(range(n))
        unvisited.remove(current_city)
        
        # Chọn thành phố gần nhất tiếp theo
        while unvisited:
            next_city = min(unvisited, key=lambda city: dist_matrix[current_city][city])
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        # Quay về thành phố đầu tiên
        tour.append(start_city)
        
        # Tính tổng khoảng cách
        distance = sum(dist_matrix[tour[i]][tour[i+1]] for i in range(n))
        
        if distance < best_distance:
            best_distance = distance
            best_tour = tour
    
    execution_time = time.time() - start_time
    
    return best_tour, best_distance, execution_time

# Đo lượng bộ nhớ sử dụng
def measure_memory_usage(algorithm, *args, **kwargs):
    mem_usage, result = memory_usage((algorithm, args, kwargs), retval=True, timeout=100, interval=0.1)
    return result, max(mem_usage) - min(mem_usage)

# Thực thi các thuật toán và so sánh kết quả
def solve_tsp(dist_matrix):
    results = {}
    memory_usages = {}
    
    print("Kích thước ma trận khoảng cách:", len(dist_matrix))
    
    # 1. Thuật toán tham lam
    print("\n1. Đang chạy thuật toán tham lam...")
    (tour, distance, time_taken), memory = measure_memory_usage(greedy_algorithm, dist_matrix)
    results["Tham lam"] = {"tour": tour, "distance": distance, "time": time_taken}
    memory_usages["Tham lam"] = memory
    print(f"Tour: {tour}")
    print(f"Tổng khoảng cách: {distance}")
    print(f"Thời gian thực thi: {time_taken:.6f} giây")
    print(f"Bộ nhớ sử dụng: {memory:.2f} MB")
    
    # 2. Thuật toán brute force
    n = len(dist_matrix)
    if n <= 10:
        print("\n2. Đang chạy thuật toán brute force...")
        (tour, distance, time_taken), memory = measure_memory_usage(brute_force, dist_matrix)
        results["Brute Force"] = {"tour": tour, "distance": distance, "time": time_taken}
        memory_usages["Brute Force"] = memory
        print(f"Tour: {tour}")
        print(f"Tổng khoảng cách: {distance}")
        print(f"Thời gian thực thi: {time_taken:.6f} giây")
        print(f"Bộ nhớ sử dụng: {memory:.2f} MB")
    else:
        print("\n2. Bỏ qua thuật toán brute force (n > 10)")
        results["Brute Force"] = {"tour": None, "distance": None, "time": None}
        memory_usages["Brute Force"] = None
    
    # 3. Thuật toán xấp xỉ (2-opt)
    print("\n3. Đang chạy thuật toán xấp xỉ (2-opt)...")
    (tour, distance, time_taken), memory = measure_memory_usage(approximation_algorithm, dist_matrix)
    results["Xấp xỉ (2-opt)"] = {"tour": tour, "distance": distance, "time": time_taken}
    memory_usages["Xấp xỉ (2-opt)"] = memory
    print(f"Tour: {tour}")
    print(f"Tổng khoảng cách: {distance}")
    print(f"Thời gian thực thi: {time_taken:.6f} giây")
    print(f"Bộ nhớ sử dụng: {memory:.2f} MB")
    
    # 4. Thuật toán bầy kiến
    if n <= 30:
        print("\n4. Đang chạy thuật toán bầy kiến...")
        (tour, distance, time_taken), memory = measure_memory_usage(ant_colony_optimization, dist_matrix)
        results["Bầy kiến"] = {"tour": tour, "distance": distance, "time": time_taken}
        memory_usages["Bầy kiến"] = memory
        print(f"Tour: {tour}")
        print(f"Tổng khoảng cách: {distance}")
        print(f"Thời gian thực thi: {time_taken:.6f} giây")
        print(f"Bộ nhớ sử dụng: {memory:.2f} MB")
    else:
        print("\n4. Bỏ qua thuật toán bầy kiến (n > 30)")
        results["Bầy kiến"] = {"tour": None, "distance": None, "time": None}
        memory_usages["Bầy kiến"] = None
    
    # 5. Thuật toán di truyền
    if n <= 100:
        print("\n5. Đang chạy thuật toán di truyền...")
        (tour, distance, time_taken), memory = measure_memory_usage(genetic_algorithm, dist_matrix)
        results["Di truyền"] = {"tour": tour, "distance": distance, "time": time_taken}
        memory_usages["Di truyền"] = memory
        print(f"Tour: {tour}")
        print(f"Tổng khoảng cách: {distance}")
        print(f"Thời gian thực thi: {time_taken:.6f} giây")
        print(f"Bộ nhớ sử dụng: {memory:.2f} MB")
    else:
        print("\n5. Bỏ qua thuật toán di truyền (n > 100)")
        results["Di truyền"] = {"tour": None, "distance": None, "time": None}
        memory_usages["Di truyền"] = None
    
    # 6. Thuật toán láng giềng gần nhất
    print("\n6. Đang chạy thuật toán láng giềng gần nhất...")
    (tour, distance, time_taken), memory = measure_memory_usage(nearest_neighbor, dist_matrix)
    results["Láng giềng gần nhất"] = {"tour": tour, "distance": distance, "time": time_taken}
    memory_usages["Láng giềng gần nhất"] = memory
    print(f"Tour: {tour}")
    print(f"Tổng khoảng cách: {distance}")
    print(f"Thời gian thực thi: {time_taken:.6f} giây")
    print(f"Bộ nhớ sử dụng: {memory:.2f} MB")
    
    return results, memory_usages

# Vẽ biểu đồ so sánh thời gian và bộ nhớ
def plot_comparison(results, memory_usages):
    algorithms = []
    times = []
    memories = []
    distances = []
    
    for algorithm, data in results.items():
        if data["time"] is not None:  # Chỉ vẽ các thuật toán đã thực hiện
            algorithms.append(algorithm)
            times.append(data["time"])
            memories.append(memory_usages[algorithm])
            distances.append(data["distance"])
    
    # Vẽ biểu đồ thời gian thực thi
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(algorithms, times, color='skyblue')
    plt.title('Thời gian thực thi các thuật toán')
    plt.xlabel('Thuật toán')
    plt.ylabel('Thời gian (giây)')
    plt.xticks(rotation=45, ha='right')
    
    # Thêm nhãn giá trị lên từng cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.6f}',
                 ha='center', va='bottom', rotation=45)
    
    # Vẽ biểu đồ bộ nhớ sử dụng
    plt.subplot(1, 2, 2)
    bars = plt.bar(algorithms, memories, color='lightgreen')
    plt.title('Bộ nhớ sử dụng của các thuật toán')
    plt.xlabel('Thuật toán')
    plt.ylabel('Bộ nhớ (MB)')
    plt.xticks(rotation=45, ha='right')
    
    # Thêm nhãn giá trị lên từng cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.2f}',
                 ha='center', va='bottom', rotation=45)
    
    plt.tight_layout()
    plt.savefig('tsp_comparison.png')
    plt.close()
    
    # Vẽ biểu đồ so sánh tổng khoảng cách
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, distances, color='salmon')
    plt.title('Tổng khoảng cách của các thuật toán')
    plt.xlabel('Thuật toán')
    plt.ylabel('Tổng khoảng cách')
    plt.xticks(rotation=45, ha='right')
    
    # Thêm nhãn giá trị lên từng cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('tsp_distances.png')
    plt.close()
    
    print("\nĐã lưu biểu đồ so sánh vào tsp_comparison.png và tsp_distances.png")

# Chương trình chính
def main():
    # Đọc dữ liệu từ file TSP
    file_path = 'city.tsp'
    dist_matrix = read_tsp_file(file_path)
    
    print("Ma trận khoảng cách:")
    print(dist_matrix)
    
    # Giải quyết bài toán TSP
    results, memory_usages = solve_tsp(dist_matrix)
    
    # Vẽ biểu đồ so sánh
    plot_comparison(results, memory_usages)
    
    # Tóm tắt kết quả
    print("\n--- TÓM TẮT KẾT QUẢ ---")
    headers = ["Thuật toán", "Khoảng cách", "Thời gian (s)", "Bộ nhớ (MB)"]
    print(f"{headers[0]:<20} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15}")
    print("-" * 70)
    
    for algorithm, data in results.items():
        distance = data["distance"] if data["distance"] is not None else "N/A"
        time_taken = f"{data['time']:.6f}" if data["time"] is not None else "N/A"
        memory = f"{memory_usages[algorithm]:.2f}" if memory_usages[algorithm] is not None else "N/A"
        
        print(f"{algorithm:<20} {distance:<15} {time_taken:<15} {memory:<15}")

if __name__ == "__main__":
    main()