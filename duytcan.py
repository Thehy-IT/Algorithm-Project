import numpy as np
import itertools

class BaiToanTSPDuyetCan:
    def __init__(self, ma_tran_khoang_cach):
        """
        Khởi tạo lớp giải bài toán TSP bằng thuật toán Duyệt cạn.

        Tham số:
        -----------
        ma_tran_khoang_cach : numpy.ndarray
            Ma trận khoảng cách giữa các thành phố.
        """
        self.ma_tran_khoang_cach = ma_tran_khoang_cach
        self.so_thanh_pho = ma_tran_khoang_cach.shape[0]
        self.khoang_cach_tot_nhat = float('inf')
        self.duong_di_tot_nhat = None

    def _tinh_khoang_cach_duong_di(self, duong_di):
        """Tính tổng khoảng cách của một đường đi."""
        tong_kc = 0
        for i in range(self.so_thanh_pho):
            tong_kc += self.ma_tran_khoang_cach[duong_di[i], duong_di[(i + 1) % self.so_thanh_pho]]
        return tong_kc

    def chay(self):
        """Chạy thuật toán Duyệt cạn."""
        print(f"Bắt đầu thuật toán Duyệt cạn cho TSP với {self.so_thanh_pho} thành phố")

        # Tạo tất cả các hoán vị có thể của các thành phố (trừ thành phố bắt đầu cố định là 0)
        cac_thanh_pho = list(range(1, self.so_thanh_pho))
        tat_ca_cac_duong_di = itertools.permutations(cac_thanh_pho)

        for duong_di_hoan_vi in tat_ca_cac_duong_di:
            # Thêm thành phố bắt đầu (0) vào đầu và cuối đường đi để tạo thành chu trình
            duong_di = [0] + list(duong_di_hoan_vi)
            khoang_cach = self._tinh_khoang_cach_duong_di(duong_di)

            if khoang_cach < self.khoang_cach_tot_nhat:
                self.khoang_cach_tot_nhat = khoang_cach
                self.duong_di_tot_nhat = duong_di

        duong_di_hien_thi = self.duong_di_tot_nhat + [self.duong_di_tot_nhat[0]] if self.duong_di_tot_nhat else None
        return self.duong_di_tot_nhat, self.khoang_cach_tot_nhat, duong_di_hien_thi


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
    """Hàm chính để chạy thuật toán Duyệt cạn."""
    duong_dan_file = "citytest.tsp"
    ma_tran_khoang_cach = doc_file_tsp_lower_diag_row(duong_dan_file)
    so_thanh_pho = ma_tran_khoang_cach.shape[0]

    print(f"Đã tải ma trận khoảng cách kích thước: {so_thanh_pho}x{so_thanh_pho}")
    if so_thanh_pho > 10:
        print("Cảnh báo: Thuật toán Duyệt cạn có độ phức tạp giai thừa. Có thể mất rất nhiều thời gian với số lượng thành phố lớn hơn 10.")

    # Khởi tạo và chạy thuật toán Duyệt cạn
    duyet_can = BaiToanTSPDuyetCan(ma_tran_khoang_cach=ma_tran_khoang_cach)
    duong_di_tot_nhat, khoang_cach_tot_nhat, duong_di_hien_thi = duyet_can.chay()

    # Hiển thị kết quả
    print("\nKết quả (Thuật toán Duyệt cạn):")
    print(f"Khoảng cách tốt nhất: {khoang_cach_tot_nhat:.2f}")
    print(f"Đường đi tốt nhất (bắt đầu từ 0): {duong_di_tot_nhat}")
    print(f"Hành trình đầy đủ (quay về điểm đầu): {duong_di_hien_thi}")

    # Chuyển sang chỉ số 1 cho người dùng
    duong_di_1indexed = [tp + 1 for tp in duong_di_hien_thi] if duong_di_hien_thi else None
    print(f"Hành trình tốt nhất (chỉ số 1): {duong_di_1indexed}")


if __name__ == "__main__":
    main()