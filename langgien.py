import numpy as np
import random
import matplotlib.pyplot as plt

class BaiToanTSPLangGieng:
    def __init__(self, ma_tran_khoang_cach):
        """
        Khởi tạo lớp giải bài toán TSP bằng thuật toán Láng giềng (Local Search).

        Tham số:
        -----------
        ma_tran_khoang_cach : numpy.ndarray
            Ma trận khoảng cách giữa các thành phố.
        """
        self.ma_tran_khoang_cach = ma_tran_khoang_cach
        self.so_thanh_pho = ma_tran_khoang_cach.shape[0]

    def _tinh_khoang_cach_duong_di(self, duong_di):
        """Tính tổng khoảng cách của một đường đi."""
        tong_kc = 0
        for i in range(self.so_thanh_pho):
            tong_kc += self.ma_tran_khoang_cach[duong_di[i], duong_di[(i + 1) % self.so_thanh_pho]]
        return tong_kc

    def _tao_lang_gieng(self, duong_di):
        """Tạo một đường đi láng giềng bằng cách đổi chỗ hai thành phố ngẫu nhiên."""
        lang_gieng = duong_di[:]
        idx1, idx2 = random.sample(range(self.so_thanh_pho), 2)
        lang_gieng[idx1], lang_gieng[idx2] = lang_gieng[idx2], lang_gieng[idx1]
        return lang_gieng

    def tim_duong_di(self, bat_dau_duong_di=None, so_lan_lap=1000):
        """
        Tìm đường đi tốt nhất bằng thuật toán Láng giềng.

        Tham số:
        -----------
        bat_dau_duong_di : list, optional
            Đường đi ban đầu. Nếu None, một đường đi ngẫu nhiên sẽ được tạo.
        so_lan_lap : int, optional
            Số lần lặp của thuật toán.

        Trả về:
        -------
        duong_di_tot_nhat : list
            Đường đi tốt nhất tìm được.
        khoang_cach_tot_nhat : float
            Tổng khoảng cách của đường đi tốt nhất.
        lich_su_tien_hoa : list
            Lịch sử các khoảng cách tốt nhất tìm được qua các lần lặp.
        """
        if bat_dau_duong_di is None:
            duong_di_hien_tai = list(range(self.so_thanh_pho))
            random.shuffle(duong_di_hien_tai)
        else:
            duong_di_hien_tai = bat_dau_duong_di[:]

        khoang_cach_hien_tai = self._tinh_khoang_cach_duong_di(duong_di_hien_tai)
        duong_di_tot_nhat = duong_di_hien_tai[:]
        khoang_cach_tot_nhat = khoang_cach_hien_tai
        lich_su_tien_hoa = [khoang_cach_tot_nhat]

        print("Bắt đầu thuật toán Láng giềng cho TSP...")

        for i in range(so_lan_lap):
            lang_gieng = self._tao_lang_gieng(duong_di_hien_tai)
            khoang_cach_lang_gieng = self._tinh_khoang_cach_duong_di(lang_gieng)

            if khoang_cach_lang_gieng < khoang_cach_hien_tai:
                duong_di_hien_tai = lang_gieng
                khoang_cach_hien_tai = khoang_cach_lang_gieng

                if khoang_cach_hien_tai < khoang_cach_tot_nhat:
                    khoang_cach_tot_nhat = khoang_cach_hien_tai
                    duong_di_tot_nhat = duong_di_hien_tai[:]

            lich_su_tien_hoa.append(khoang_cach_tot_nhat)
            if (i + 1) % (so_lan_lap // 10) == 0 or i == 0:
                print(f"Lần lặp {i + 1}/{so_lan_lap}, Khoảng cách tốt nhất hiện tại: {khoang_cach_tot_nhat:.2f}")

        duong_di_hien_thi = duong_di_tot_nhat + [duong_di_tot_nhat[0]]
        return duong_di_tot_nhat, khoang_cach_tot_nhat, lich_su_tien_hoa

    def ve_do_thi_tien_hoa(self, lich_su_tien_hoa):
        """Vẽ đồ thị tiến hóa của thuật toán."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(lich_su_tien_hoa)), lich_su_tien_hoa)
        plt.title('Đồ thị tiến hóa của thuật toán Láng giềng')
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
    """Hàm chính để chạy thuật toán Láng giềng."""
    duong_dan_file = "citytest.tsp"
    ma_tran_khoang_cach = doc_file_tsp_lower_diag_row(duong_dan_file)

    print(f"Đã tải ma trận khoảng cách kích thước: {ma_tran_khoang_cach.shape}")

    # Khởi tạo và chạy thuật toán Láng giềng
    lang_gieng = BaiToanTSPLangGieng(ma_tran_khoang_cach=ma_tran_khoang_cach)

    # Tạo đường đi ban đầu (có thể ngẫu nhiên hoặc một heuristic khác)
    bat_dau_duong_di = list(range(lang_gieng.so_thanh_pho))
    random.shuffle(bat_dau_duong_di)

    duong_di_tot_nhat, khoang_cach_tot_nhat, lich_su_tien_hoa = lang_gieng.tim_duong_di(
        bat_dau_duong_di=bat_dau_duong_di,
        so_lan_lap=5000  # Tăng số lần lặp để tìm kiếm kỹ hơn
    )

    # Hiển thị kết quả
    print("\nKết quả (Thuật toán Láng giềng):")
    print(f"Đường đi tốt nhất (bắt đầu từ 0): {duong_di_tot_nhat}")
    print(f"Hành trình đầy đủ (quay về điểm đầu): {duong_di_tot_nhat + [duong_di_tot_nhat[0]]}")
    print(f"Tổng khoảng cách: {khoang_cach_tot_nhat:.2f}")

    # Chuyển sang chỉ số 1 cho người dùng
    duong_di_1indexed = [tp + 1 for tp in duong_di_tot_nhat + [duong_di_tot_nhat[0]]]
    print(f"Hành trình tốt nhất (chỉ số 1): {duong_di_1indexed}")

    # Vẽ đồ thị tiến hóa
    lang_gieng.ve_do_thi_tien_hoa(lich_su_tien_hoa)


if __name__ == "__main__":
    main()