import numpy as np
import random
import matplotlib.pyplot as plt

class BaiToanTSP_BandoKien:
    def __init__(self, ma_tran_khoang_cach, so_kien=10, so_lan_lap=100, alpha=1.0, beta=2.0,
                 toc_do_bay_hoi=0.5, Q=100):
        """
        Khởi tạo lớp giải bài toán TSP bằng thuật toán Bầy Kiến.

        Tham số:
        -----------
        ma_tran_khoang_cach : numpy.ndarray
            Ma trận khoảng cách giữa các thành phố.
        so_kien : int
            Số lượng kiến trong đàn.
        so_lan_lap : int
            Số lần lặp của thuật toán.
        alpha : float
            Hệ số ảnh hưởng của vết pheromone.
        beta : float
            Hệ số ảnh hưởng của khoảng cách (tính nghịch đảo).
        toc_do_bay_hoi : float
            Tốc độ bay hơi pheromone.
        Q : float
            Hằng số dùng trong cập nhật pheromone.
        """
        self.ma_tran_khoang_cach = ma_tran_khoang_cach
        self.so_thanh_pho = ma_tran_khoang_cach.shape[0]
        self.so_kien = so_kien
        self.so_lan_lap = so_lan_lap
        self.alpha = alpha
        self.beta = beta
        self.toc_do_bay_hoi = toc_do_bay_hoi
        self.Q = Q

        # Khởi tạo ma trận pheromone với giá trị nhỏ
        self.pheromone = np.ones((self.so_thanh_pho, self.so_thanh_pho)) * 0.1
        np.fill_diagonal(self.pheromone, 0)  # Đường đi đến chính nó không có pheromone

        # Lưu trữ đường đi và khoảng cách tốt nhất
        self.duong_di_tot_nhat = None
        self.khoang_cach_tot_nhat = float('inf')

        # Lịch sử khoảng cách tốt nhất qua các lần lặp (để vẽ đồ thị)
        self.lich_su_lap = []

    def _chon_thanh_pho_tiep_theo(self, duong_di_kien, thanh_pho_hien_tai):
        """Chọn thành phố tiếp theo mà kiến sẽ đi."""
        chua_tham = [tp for tp in range(self.so_thanh_pho) if tp not in duong_di_kien]

        if not chua_tham:
            return None

        # Tính xác suất đi đến các thành phố chưa thăm
        xac_suat = []
        tong_xac_suat = 0

        for tp in chua_tham:
            tau = self.pheromone[thanh_pho_hien_tai, tp]
            eta = 1.0 / self.ma_tran_khoang_cach[thanh_pho_hien_tai, tp] if self.ma_tran_khoang_cach[thanh_pho_hien_tai, tp] > 0 else 1.0
            prob = (tau ** self.alpha) * (eta ** self.beta)
            xac_suat.append(prob)
            tong_xac_suat += prob

        # Chuẩn hóa xác suất
        if tong_xac_suat > 0:
            xac_suat = [p / tong_xac_suat for p in xac_suat]
        else:
            xac_suat = [1.0 / len(chua_tham) for _ in chua_tham]

        # Chọn thành phố tiếp theo dựa trên xác suất
        idx_chon = np.random.choice(len(chua_tham), p=xac_suat)
        return chua_tham[idx_chon]

    def _tinh_khoang_cach_duong_di(self, duong_di, hien_thi_chi_tiet=False):
        """Tính tổng khoảng cách của một đường đi và hiển thị chi tiết nếu cần."""
        tong_kc = 0
        if hien_thi_chi_tiet:
            print("Lộ trình chi tiết:")
        for i in range(len(duong_di) - 1):
            kc_segment = self.ma_tran_khoang_cach[duong_di[i], duong_di[i+1]]
            tong_kc += kc_segment
            if hien_thi_chi_tiet:
                print(f"  {duong_di[i]} -> {duong_di[i+1]}: {kc_segment:.2f}")
        kc_ve_dau = self.ma_tran_khoang_cach[duong_di[-1], duong_di[0]]
        tong_kc += kc_ve_dau
        if hien_thi_chi_tiet:
            print(f"  {duong_di[-1]} -> {duong_di[0]}: {kc_ve_dau:.2f}")
            print(f"Tổng khoảng cách: {tong_kc:.2f}")
        return tong_kc

    def _cap_nhat_pheromone(self, cac_duong_di_kien, cac_khoang_cach_kien):
        """Cập nhật lượng pheromone trên các cạnh."""
        self.pheromone *= (1 - self.toc_do_bay_hoi)  # Pheromone bay hơi

        for duong_di, khoang_cach in zip(cac_duong_di_kien, cac_khoang_cach_kien):
            if khoang_cach > 0:
                luong_pheromone_moi = self.Q / khoang_cach
                for i in range(len(duong_di) - 1):
                    u = duong_di[i]
                    v = duong_di[i+1]
                    self.pheromone[u, v] += luong_pheromone_moi
                    self.pheromone[v, u] += luong_pheromone_moi  # TSP đối xứng
                # Cập nhật cho cạnh quay về điểm đầu
                self.pheromone[duong_di[-1], duong_di[0]] += luong_pheromone_moi
                self.pheromone[duong_di[0], duong_di[-1]] += luong_pheromone_moi

    def chay(self, hien_thi_chi_tiet_lo_trinh=False):
        """Chạy thuật toán Bầy Kiến."""
        print(f"Bắt đầu thuật toán Bầy Kiến cho TSP với {self.so_thanh_pho} thành phố")

        for lan_lap in range(self.so_lan_lap):
            cac_duong_di_kien = []
            cac_khoang_cach_kien = []

            for kien in range(self.so_kien):
                # Kiến bắt đầu ở một thành phố ngẫu nhiên
                bat_dau = random.randint(0, self.so_thanh_pho - 1)
                thanh_pho_hien_tai = bat_dau
                duong_di = [thanh_pho_hien_tai]

                # Xây dựng đường đi cho kiến
                while len(duong_di) < self.so_thanh_pho:
                    thanh_pho_tiep_theo = self._chon_thanh_pho_tiep_theo(duong_di, thanh_pho_hien_tai)
                    if thanh_pho_tiep_theo is not None:
                        duong_di.append(thanh_pho_tiep_theo)
                        thanh_pho_hien_tai = thanh_pho_tiep_theo
                    else:
                        break  # Nếu không còn thành phố để đi

                # Nếu kiến đi hết các thành phố, tính khoảng cách
                if len(duong_di) == self.so_thanh_pho:
                    khoang_cach = self._tinh_khoang_cach_duong_di(duong_di)
                    cac_duong_di_kien.append(duong_di)
                    cac_khoang_cach_kien.append(khoang_cach)

                    # Cập nhật đường đi tốt nhất
                    if khoang_cach < self.khoang_cach_tot_nhat:
                        self.khoang_cach_tot_nhat = khoang_cach
                        self.duong_di_tot_nhat = duong_di.copy()

            # Cập nhật pheromone sau khi tất cả kiến hoàn thành đường đi (hoặc bị kẹt)
            if cac_duong_di_kien:
                self._cap_nhat_pheromone(cac_duong_di_kien, cac_khoang_cach_kien)

            # Lưu lại khoảng cách tốt nhất của lần lặp này
            if cac_khoang_cach_kien:
                self.lich_su_lap.append(min(cac_khoang_cach_kien))
            elif self.lich_su_lap:
                self.lich_su_lap.append(self.lich_su_lap[-1]) # Giữ nguyên nếu không có đường đi hoàn chỉnh

            if (lan_lap + 1) % 10 == 0 or lan_lap == 0:
                print(f"Lần lặp {lan_lap + 1}/{self.so_lan_lap}, Khoảng cách tốt nhất hiện tại: {self.khoang_cach_tot_nhat:.2f}")

        # Thêm điểm bắt đầu vào cuối đường đi để hiển thị hành trình đầy đủ
        duong_di_hien_thi = self.duong_di_tot_nhat + [self.duong_di_tot_nhat[0]] if self.duong_di_tot_nhat else None

        return self.duong_di_tot_nhat, self.khoang_cach_tot_nhat, duong_di_hien_thi

    def ve_do_thi_hoi_tu(self):
        """Vẽ đồ thị hội tụ của thuật toán."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.lich_su_lap) + 1), self.lich_su_lap)
        plt.title('Đồ thị hội tụ của thuật toán Bầy Kiến')
        plt.xlabel('Lần lặp')
        plt.ylabel('Khoảng cách tốt nhất')
        plt.grid(True)
        plt.show()


def doc_file_tsp_lower_diag_row(duong_dan_file):
    """Đọc file TSP định dạng LOWER_DIAG_ROW."""
    with open(duong_dan_file, 'r') as f:
        lines = f.readlines()

    kich_thuoc = None
    phan_du_lieu = False
    cac_dong_du_lieu = []

    for line in lines:
        line = line.strip()
        if "DIMENSION" in line:
            kich_thuoc = int(line.split(":")[-1].strip())
        elif "EDGE_WEIGHT_SECTION" in line:
            phan_du_lieu = True
            continue

        if phan_du_lieu and line and not line.startswith('EOF'):
            cac_dong_du_lieu.append(line)

    if kich_thuoc is None:
        raise ValueError("Không tìm thấy kích thước trong file TSP")

    ma_tran_khoang_cach = np.zeros((kich_thuoc, kich_thuoc))
    dong = 0
    cot = 0

    for line in cac_dong_du_lieu:
        cac_gia_tri = line.split()
        for gia_tri in cac_gia_tri:
            ma_tran_khoang_cach[dong, cot] = float(gia_tri)
            if dong != cot:
                ma_tran_khoang_cach[cot, dong] = float(gia_tri)
            cot += 1
            if cot > dong:
                dong += 1
                cot = 0

    return ma_tran_khoang_cach


def main():
    """Hàm chính để chạy thuật toán."""
    duong_dan_file = "citytest.tsp"
    ma_tran_khoang_cach = doc_file_tsp_lower_diag_row(duong_dan_file)

    print(f"Đã tải ma trận khoảng cách kích thước: {ma_tran_khoang_cach.shape}")

    # Khởi tạo và chạy thuật toán ACO
    aco = BaiToanTSP_BandoKien(
        ma_tran_khoang_cach=ma_tran_khoang_cach,
        so_kien=20,
        so_lan_lap=100,
        alpha=1.0,
        beta=2.5,
        toc_do_bay_hoi=0.5,
        Q=100
    )

    duong_di_tot_nhat, khoang_cach_tot_nhat, duong_di_hien_thi = aco.chay(hien_thi_chi_tiet_lo_trinh=True)

    # Hiển thị kết quả cuối cùng
    print("\nKết quả cuối cùng:")
    print(f"Khoảng cách tốt nhất: {khoang_cach_tot_nhat:.2f}")
    print(f"Đường đi tốt nhất (bắt đầu từ 0): {duong_di_tot_nhat}")
    print(f"Hành trình đầy đủ (quay về điểm đầu): {duong_di_hien_thi}")

    # Chuyển sang chỉ số 1 cho người dùng
    duong_di_1indexed = [tp + 1 for tp in duong_di_hien_thi] if duong_di_hien_thi else None
    print(f"Hành trình tốt nhất (chỉ số 1): {duong_di_1indexed}")

    # Vẽ đồ thị hội tụ
    aco.ve_do_thi_hoi_tu()


if __name__ == "__main__":
    main()