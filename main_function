import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Heiti TC'

# 二進位編碼與解碼
def message_to_bits(message):
    return ''.join(f'{ord(c):08b}' for c in message)


def find_zero(hist, peak):
    zeros_right = [i for i in range(peak+1, 256) if hist[i] == 0]
    if zeros_right:
        return zeros_right[0]
    zeros_left = [i for i in range(peak-1, -1, -1) if hist[i] == 0]
    if zeros_left:
        return zeros_left[0]
    # 找頻率最小的右側
    candidates = [(i, hist[i]) for i in range(peak+1, 256)]
    if candidates:
        return min(candidates, key=lambda x: x[1])[0]
    # 找頻率最小的左側
    candidates = [(i, hist[i]) for i in range(peak-1, -1, -1)]
    if candidates:
        return min(candidates, key=lambda x: x[1])[0]
    raise ValueError("無法找到 suitable zero 點")




def bits_to_message(bits):
    chars = [chr(int(bits[i:i+8], 2)) for i in range(0, len(bits), 8)]
    return ''.join(chars)

# 嵌入資料：直方圖平移法
def embed_histogram_shift(image_path, message, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("無法讀取圖片，請確認路徑正確")

    flat = img.flatten()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    peak = np.argmax(hist)

    zero = find_zero(hist, peak)

    bits = message_to_bits(message)
    if hist[peak] < len(bits):
        raise ValueError("圖片空間不足，無法嵌入訊息")

    # 位移區間 [peak+1, zero)
    for i in range(flat.size):
        if peak < flat[i] < zero:
            flat[i] += 1

    # 嵌入訊息
    embedded = 0
    for i in range(flat.size):
        if flat[i] == peak and embedded < len(bits):
            flat[i] += int(bits[embedded])  # 若是 1 則 peak+1，若是 0 保持 peak
            embedded += 1

    new_img = flat.reshape(img.shape)
    cv2.imwrite(output_path, new_img)
    print(f"[完成] 訊息已嵌入並儲存至 {output_path}")
    return peak, zero, len(bits)


# 解碼訊息
def extract_histogram_shift(image_path, peak, length):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    flat = img.flatten()

    bits = ''
    extracted = 0
    for i in range(flat.size):
        if flat[i] == peak:
            bits += '0'
            extracted += 1
        elif flat[i] == peak + 1:
            bits += '1'
            extracted += 1
        if extracted == length:
            break

    message = bits_to_message(bits)
    print(f"[提取結果] 隱藏訊息為：{message}")
    return message

# 還原原圖
def recover_image(image_path, peak, zero, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    flat = img.flatten()

    for i in range(flat.size):
        if peak < flat[i] <= zero:
            flat[i] -= 1
        elif flat[i] == peak + 1:
            flat[i] = peak

    recovered = flat.reshape(img.shape)
    cv2.imwrite(output_path, recovered)
    print(f"[還原完成] 圖片已還原儲存為 {output_path}")
    return recovered

# 差異比較
def compare_images(original_path, embedded_path):
    orig = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    emb = cv2.imread(embedded_path, cv2.IMREAD_GRAYSCALE)
    diff = cv2.absdiff(orig, emb)

    mse = np.mean((orig - emb) ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse != 0 else float('inf')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("原始圖片")
    plt.imshow(orig, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("嵌入後圖片")
    plt.imshow(emb, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("差異圖 (abs)")
    plt.imshow(diff, cmap='hot')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"MSE: {mse:.4f}, PSNR: {psnr:.2f} dB")

# 主執行流程
if __name__ == "__main__":
    original = "original.png"
    embedded = "embedded.png"
    recovered = "recovered.png"
    secret = "This is secret"

    peak, zero, msg_len = embed_histogram_shift(original, secret, embedded)
    _ = extract_histogram_shift(embedded, peak, msg_len)
    _ = recover_image(embedded, peak, zero, recovered)
    compare_images(original, embedded)
