import numpy as np
import random
import matplotlib.pyplot as plt

class BaiToanTSPGiaiThuatDiTruyen:
    def __init__(self, ma_tran_khoang_cach, kich_thuoc_quan_the=50, so_the_he=100, ty_le_lai_ghep=0.8,
                 ty_le_dot_bien=0.05, ty_le_chon_loc=0.5):
        """
        Khởi tạo lớp giải bài toán TSP bằng thuật toán Di truyền.

        Tham số:
        -----------
        ma_tran_khoang_cach : numpy.ndarray
            Ma trận khoảng cách giữa các thành phố.
        kich_thuoc_quan_the : int
            Số lượng cá thể trong quần thể.
        so_the_he : int
            Số lượng thế hệ tiến hóa.
        ty_le_lai_ghep : float
            Xác suất lai ghép giữa hai cá thể.
        ty_le_dot_bien : float
            Xác suất đột biến của một cá thể.
        ty_le_chon_loc : float
            Tỷ lệ cá thể tốt nhất được giữ lại cho thế hệ sau.
        """
        self.ma_tran_khoang_cach = ma_tran_khoang_cach
        self.so_thanh_pho = ma_tran_khoang_cach.shape[0]
        self.kich_thuoc_quan_the = kich_thuoc_quan_the
        self.so_the_he = so_the_he
        self.ty_le_lai_ghep = ty_le_lai_ghep
        self.ty_le_dot_bien = ty_le_dot_bien
        self.ty_le_chon_loc = ty_le_chon_loc

        # Khởi tạo quần thể ban đầu với các đường đi ngẫu nhiên
        self.quan_the = self._khoi_tao_quan_the()
        self.khoang_cach_tot_nhat = float('inf')
        self.duong_di_tot_nhat = None
        self.lich_su_tien_hoa = []

    def _khoi_tao_quan_the(self):
        """Khởi tạo quần thể ban đầu với các đường đi ngẫu nhiên."""
        quan_the = []
        for _ in range(self.kich_thuoc_quan_the):
            duong_di = list(range(self.so_thanh_pho))
            random.shuffle(duong_di)
            quan_the.append(duong_di)
        return quan_the

    def _tinh_do_thich_nghi(self, ca_the):
        """Tính độ thích nghi (nghịch đảo của tổng khoảng cách) của một cá thể."""
        tong_khoang_cach = 0
        for i in range(self.so_thanh_pho):
            tong_khoang_cach += self.ma_tran_khoang_cach[ca_the[i], ca_the[(i + 1) % self.so_thanh_pho]]
        return 1 / tong_khoang_cach if tong_khoang_cach > 0 else 0

    def _chon_loc(self, quan_the, do_thich_nghi):
        """Chọn lọc các cá thể tốt nhất để tạo thế hệ mới."""
        so_luong_chon_loc = int(self.kich_thuoc_quan_the * self.ty_le_chon_loc)
        cac_ca_the_duoc_chon = [quan_the[i] for i in np.argsort(do_thich_nghi)[::-1][:so_luong_chon_loc]]
        return cac_ca_the_duoc_chon

    def _lai_ghep(self, cha, me):
        """Thực hiện lai ghép OX1 (Order Crossover 1) giữa hai cá thể."""
        diem_bat_dau = random.randint(0, self.so_thanh_pho - 1)
        diem_ket_thuc = random.randint(diem_bat_dau + 1, self.so_thanh_pho)

        con = [-1] * self.so_thanh_pho
        phan_cua_cha = cha[diem_bat_dau:diem_ket_thuc]
        con[diem_bat_dau:diem_ket_thuc] = phan_cua_cha

        vi_tri_me = 0
        for i in range(self.so_thanh_pho):
            if con[i] == -1:
                while me[vi_tri_me] in phan_cua_cha:
                    vi_tri_me += 1
                con[i] = me[vi_tri_me]
                vi_tri_me += 1
        return con

    def _dot_bien(self, ca_the):
        """Thực hiện đột biến bằng cách đổi chỗ hai thành phố ngẫu nhiên."""
        if random.random() < self.ty_le_dot_bien:
            idx1, idx2 = random.sample(range(self.so_thanh_pho), 2)
            ca_the[idx1], ca_the[idx2] = ca_the[idx2], ca_the[idx1]
        return ca_the

    def chay(self):
        """Chạy thuật toán Di truyền."""
        print(f"Bắt đầu thuật toán Di truyền cho TSP với {self.so_thanh_pho} thành phố")

        for the_he in range(self.so_the_he):
            do_thich_nghi = np.array([self._tinh_do_thich_nghi(ca_the) for ca_the in self.quan_the])
            ca_the_tot_nhat_idx = np.argmax(do_thich_nghi)
            ca_the_tot_nhat_hien_tai = self.quan_the[ca_the_tot_nhat_idx]
            khoang_cach_hien_tai = 1 / do_thich_nghi[ca_the_tot_nhat_idx] if do_thich_nghi[ca_the_tot_nhat_idx] > 0 else float('inf')

            if khoang_cach_hien_tai < self.khoang_cach_tot_nhat:
                self.khoang_cach_tot_nhat = khoang_cach_hien_tai
                self.duong_di_tot_nhat = ca_the_tot_nhat_hien_tai

            self.lich_su_tien_hoa.append(self.khoang_cach_tot_nhat)
            print(f"Thế hệ {the_he + 1}/{self.so_the_he}, Khoảng cách tốt nhất: {self.khoang_cach_tot_nhat:.2f}")

            # Chọn lọc
            cac_ca_the_duoc_chon = self._chon_loc(self.quan_the, do_thich_nghi)
            quan_the_moi = cac_ca_the_duoc_chon.copy()

            # Lai ghép và đột biến để tạo thế hệ mới
            while len(quan_the_moi) < self.kich_thuoc_quan_the:
                cha = random.choice(cac_ca_the_duoc_chon)
                me = random.choice(cac_ca_the_duoc_chon)
                if random.random() < self.ty_le_lai_ghep:
                    con = self._lai_ghep(cha, me)
                    con = self._dot_bien(con)
                    quan_the_moi.append(con)
                else:
                    quan_the_moi.append(random.choice(cac_ca_the_duoc_chon).copy()) # Giữ lại cá thể cũ nếu không lai ghép

            self.quan_the = quan_the_moi

        duong_di_hien_thi = self.duong_di_tot_nhat + [self.duong_di_tot_nhat[0]] if self.duong_di_tot_nhat else None
        return self.duong_di_tot_nhat, self.khoang_cach_tot_nhat, duong_di_hien_thi

    def ve_do_thi_tien_hoa(self):
        """Vẽ đồ thị tiến hóa của thuật toán."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.lich_su_tien_hoa) + 1), self.lich_su_tien_hoa)
        plt.title('Đồ thị tiến hóa của thuật toán Di truyền')
        plt.xlabel('Thế hệ')
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
    """Hàm chính để chạy thuật toán Di truyền."""
    duong_dan_file = "citytest.tsp"
    ma_tran_khoang_cach = doc_file_tsp_lower_diag_row(duong_dan_file)

    print(f"Đã tải ma trận khoảng cách kích thước: {ma_tran_khoang_cach.shape}")

    # Khởi tạo và chạy thuật toán Di truyền
    ga = BaiToanTSPGiaiThuatDiTruyen(
        ma_tran_khoang_cach=ma_tran_khoang_cach,
        kich_thuoc_quan_the=50,
        so_the_he=200,
        ty_le_lai_ghep=0.8,
        ty_le_dot_bien=0.05,
        ty_le_chon_loc=0.4
    )

    duong_di_tot_nhat, khoang_cach_tot_nhat, duong_di_hien_thi = ga.chay()

    # Hiển thị kết quả
    print("\nKết quả:")
    print(f"Khoảng cách tốt nhất: {khoang_cach_tot_nhat:.2f}")
    print(f"Đường đi tốt nhất (bắt đầu từ 0): {duong_di_tot_nhat}")
    print(f"Hành trình đầy đủ (quay về điểm đầu): {duong_di_hien_thi}")

    # Chuyển sang chỉ số 1 cho người dùng
    duong_di_1indexed = [tp + 1 for tp in duong_di_hien_thi] if duong_di_hien_thi else None
    print(f"Hành trình tốt nhất (chỉ số 1): {duong_di_1indexed}")

    # Vẽ đồ thị tiến hóa
    ga.ve_do_thi_tien_hoa()


if __name__ == "__main__":
    main()