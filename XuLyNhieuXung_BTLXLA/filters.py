import numpy as np
import random


# =============================
# Thêm nhiễu Salt & Pepper
# =============================
def add_salt_pepper_noise(image, prob=0.05):

    noisy = image.copy()

    if len(image.shape) == 2:

        h, w = image.shape

        for i in range(h):
            for j in range(w):

                r = random.random()

                if r < prob / 2:
                    noisy[i, j] = 0

                elif r < prob:
                    noisy[i, j] = 255

    else:

        h, w, c = image.shape

        for i in range(h):
            for j in range(w):

                r = random.random()

                if r < prob / 2:
                    noisy[i, j] = [0, 0, 0]

                elif r < prob:
                    noisy[i, j] = [255, 255, 255]

    return noisy


# =============================
# Insertion Sort 
# =============================
def insertion_sort(arr):

    a = arr.copy()

    for i in range(1, len(a)):

        key = a[i]
        j = i - 1

        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1

        a[j + 1] = key

    return a


# =============================
# Lấy cửa sổ W×W tại (row, col)
# =============================
def get_window(padded, row, col, half):

    values = []

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            values.append(float(padded[row + half + dy, col + half + dx]))

    return values


# =============================
# Median Filter
# =============================
def median_filter(image, window_size=3):
    
    half = window_size // 2
    padded = np.pad(image, half, mode='edge')

    h, w = image.shape
    result = np.zeros_like(image)

    for i in range(h):
        for j in range(w):

            window = get_window(padded, i, j, half)
            sorted_w = insertion_sort(window)
            mid = len(sorted_w) // 2

            result[i, j] = sorted_w[mid]

    return result.astype(np.uint8)


# =============================
# Pseudo-Median Filter
# =============================
def pseudo_median_filter(image, window_size=3):
   
    half = window_size // 2
    padded = np.pad(image, half, mode='edge')

    h, w = image.shape
    result = np.zeros_like(image, dtype=np.float64)

    for i in range(h):
        for j in range(w):

            row_mins = []
            row_maxs = []

            # Duyệt từng hàng trong cửa sổ
            for dy in range(-half, half + 1):

                row_vals = []

                for dx in range(-half, half + 1):
                    row_vals.append(float(padded[i + half + dy, j + half + dx]))

                row_mins.append(min(row_vals))
                row_maxs.append(max(row_vals))

            maximin = max(row_mins)   # max của các min hàng
            minimax = min(row_maxs)   # min của các max hàng

            result[i, j] = (maximin + minimax) / 2.0

    return np.clip(result, 0, 255).astype(np.uint8)


# =============================
# Adaptive Median Filter (AMF)
# =============================
def adaptive_median_filter(image, max_window=7):
    """
    Lọc trung vị thích nghi:
      - Giai đoạn A: kiểm tra median của cửa sổ có phải nhiễu không
        A1 = Zmed - Zmin > 0
        A2 = Zmed - Zmax < 0
        Nếu đúng: sang B. Sai: mở rộng cửa sổ.

      - Giai đoạn B: kiểm tra pixel gốc có phải nhiễu không
        B1 = Zxy  - Zmin > 0
        B2 = Zxy  - Zmax < 0
        Đúng: giữ pixel gốc. Sai: thay bằng Zmed.
    """

    h, w = image.shape
    result = np.zeros_like(image, dtype=np.float64)
    max_pad = max_window // 2
    padded = np.pad(image, max_pad, mode='edge')

    for i in range(h):
        for j in range(w):

            zxy = float(image[i, j])
            w_size = 3

            while w_size <= max_window:

                half = w_size // 2
                window = []

                for dy in range(-half, half + 1):
                    for dx in range(-half, half + 1):
                        window.append(float(padded[i + max_pad + dy, j + max_pad + dx]))

                sorted_w = insertion_sort(window)
                zmin = sorted_w[0]
                zmax = sorted_w[-1]
                zmed = sorted_w[len(sorted_w) // 2]

                # Giai đoạn A
                if zmed - zmin > 0 and zmed - zmax < 0:

                    # Giai đoạn B
                    if zxy - zmin > 0 and zxy - zmax < 0:
                        result[i, j] = zxy   # giữ nguyên
                    else:
                        result[i, j] = zmed  # là nhiễu → dùng median

                    break

                else:
                    w_size += 2  # mở rộng cửa sổ

            else:
                # Đạt max_window: dùng median của cửa sổ lớn nhất
                half = max_window // 2
                window = []
                for dy in range(-half, half + 1):
                    for dx in range(-half, half + 1):
                        window.append(float(padded[i + max_pad + dy, j + max_pad + dx]))
                sorted_w = insertion_sort(window)
                result[i, j] = sorted_w[len(sorted_w) // 2]

    return np.clip(result, 0, 255).astype(np.uint8)


# =============================
# Combined Filter (Kết hợp)
# =============================
def combined_filter(image):

    # Bước 1: Adaptive Median (loại nhiễu mạnh, giữ biên)
    step1 = adaptive_median_filter(image, max_window=7)

    # Bước 2: Median 3x3 (làm mịn tàn dư)
    step2 = median_filter(step1, window_size=3)

    return step2


# =============================
# Xử lý ảnh màu
# =============================
def process_color(image, func):

    b = func(image[:, :, 0])
    g = func(image[:, :, 1])
    r = func(image[:, :, 2])

    result = image.copy()
    result[:, :, 0] = b
    result[:, :, 1] = g
    result[:, :, 2] = r

    return result


# =============================
# MSE
# =============================
def mse(img1, img2):

    err = np.mean((img1.astype("float") - img2.astype("float")) ** 2)

    return round(err, 3)


# =============================
# PSNR
# =============================
def psnr(img1, img2):

    m = mse(img1, img2)

    if m == 0:
        return 100.0

    return round(10 * np.log10((255.0 ** 2) / m), 3)