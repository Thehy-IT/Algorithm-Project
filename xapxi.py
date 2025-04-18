import numpy as np
import random
import matplotlib.pyplot as plt

class BaiToanTSPXapXi:
    def __init__(self, ma_tran_khoang_cach):
        """
        Khởi tạo lớp giải bài toán TSP bằng thuật toán xấp xỉ (Nearest Neighbor).

        Tham số:
        -----------
        ma_tran_khoang_cach : numpy.ndarray
            Ma trận khoảng cách giữa các thành phố.
        """
        self.ma_tran_khoang_cach = ma_tran_khoang_cach
        self.so_thanh_pho = ma_tran_khoang_cach.shape[0]

    def tim_duong_di(self, bat_dau_thanh_pho=0):
        """
        Tìm đường đi theo thuật toán xấp xỉ (Nearest Neighbor).

        Tham số:
        -----------
        bat_dau_thanh_pho : int
            Chỉ số của thành phố bắt đầu (mặc định là 0).

        Trả về:
        -------
        duong_di : list
            Danh sách các thành phố theo thứ tự đường đi.
        tong_khoang_cach : float
            Tổng khoảng cách của đường đi tìm được.
        """
        da_tham = [False] * self.so_thanh_pho
        duong_di = [bat_dau_thanh_pho]
        da_tham[bat_dau_thanh_pho] = True
        tong_khoang_cach = 0

        thanh_pho_hien_tai = bat_dau_thanh_pho
        for _ in range(self.so_thanh_pho - 1):
            thanh_pho_gan_nhat = -1
            khoang_cach_nho_nhat = float('inf')

            for thanh_pho_ke_tiep in range(self.so_thanh_pho):
                if not da_tham[thanh_pho_ke_tiep]:
                    khoang_cach = self.ma_tran_khoang_cach[thanh_pho_hien_tai, thanh_pho_ke_tiep]
                    if khoang_cach < khoang_cach_nho_nhat:
                        khoang_cach_nho_nhat = khoang_cach
                        thanh_pho_gan_nhat = thanh_pho_ke_tiep

            if thanh_pho_gan_nhat != -1:
                duong_di.append(thanh_pho_gan_nhat)
                da_tham[thanh_pho_gan_nhat] = True
                tong_khoang_cach += khoang_cach_nho_nhat
                thanh_pho_hien_tai = thanh_pho_gan_nhat
            else:
                # Trường hợp không tìm thấy thành phố tiếp theo (có thể do đồ thị không liên thông)
                break

        # Quay về thành phố bắt đầu
        tong_khoang_cach += self.ma_tran_khoang_cach[duong_di[-1], duong_di[0]]
        duong_di.append(duong_di[0])

        return duong_di, tong_khoang_cach


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
    """Hàm chính để chạy thuật toán xấp xỉ (Nearest Neighbor)."""
    duong_dan_file = "citytest.tsp"
    ma_tran_khoang_cach = doc_file_tsp_lower_diag_row(duong_dan_file)

    print(f"Đã tải ma trận khoảng cách kích thước: {ma_tran_khoang_cach.shape}")

    # Khởi tạo và chạy thuật toán xấp xỉ
    xap_xi = BaiToanTSPXapXi(ma_tran_khoang_cach=ma_tran_khoang_cach)

    bat_dau_thanh_pho = 0  # Chọn thành phố bắt đầu
    duong_di_tot_nhat, khoang_cach_tot_nhat = xap_xi.tim_duong_di(bat_dau_thanh_pho)

    # Hiển thị kết quả
    print("\nKết quả (Thuật toán xấp xỉ - Nearest Neighbor):")
    print(f"Thành phố bắt đầu: {bat_dau_thanh_pho}")
    print(f"Đường đi tìm được (bắt đầu từ 0): {duong_di_tot_nhat[:-1]}")
    print(f"Hành trình đầy đủ (quay về điểm đầu): {duong_di_tot_nhat}")
    print(f"Tổng khoảng cách: {khoang_cach_tot_nhat:.2f}")

    # Chuyển sang chỉ số 1 cho người dùng
    duong_di_1indexed = [tp + 1 for tp in duong_di_tot_nhat]
    print(f"Hành trình tốt nhất (chỉ số 1): {duong_di_1indexed}")


if __name__ == "__main__":
    main()