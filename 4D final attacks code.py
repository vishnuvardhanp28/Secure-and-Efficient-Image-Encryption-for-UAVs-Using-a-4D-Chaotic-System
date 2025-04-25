import cv2
import numpy as np
import os
import hashlib
import time
from skimage.metrics import structural_similarity as ssim

# Set folder for output images (adjust this path as needed)
output_folder = r"C:\Users\vishn\Desktop\vishnu_final\output"
os.makedirs(output_folder, exist_ok=True)

####################################
# Utility Functions: Loading Images
####################################
def load_image(path, color=False):
    """
    Load an image and resize to 256x256.
    If color=True, the image is loaded in color; otherwise, in grayscale.
    """
    if color:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Error: Could not load color image from " + path)
        if img.shape[:2] != (256, 256):
            img = cv2.resize(img, (256, 256))
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Error: Could not load grayscale image from " + path)
        if img.shape != (256, 256):
            img = cv2.resize(img, (256, 256))
    return img

####################################
# Key Generation and Chaotic Sequences
####################################
def generate_key_params(img):
    """
    Generate key parameters using SHA-256 and image statistics.
    """
    sha_digest = hashlib.sha256(img.tobytes()).digest()
    K = list(sha_digest)
    N = img.shape[1]
    q = np.mean(img, axis=0)
    frac = lambda val: val - np.floor(val)
    N_quarter = N // 4
    Q1 = frac((4.0 / N) * np.sum(q[0:N_quarter]))
    Q2 = frac((4.0 / N) * np.sum(q[N_quarter:2*N_quarter]))
    Q3 = frac((4.0 / N) * np.sum(q[2*N_quarter:3*N_quarter]))
    Q4 = frac((4.0 / N) * np.sum(q[3*N_quarter:]))
    xor4 = lambda a, b, c, d: a ^ b ^ c ^ d
    alpha = (xor4(K[0], K[1], K[2], K[3]) + sum(K[4:8])) / (256.0 * 5)
    beta  = (xor4(K[8], K[9], K[10], K[11]) + sum(K[12:16])) / (256.0 * 5)
    gamma = (xor4(K[16], K[17], K[18], K[19]) + sum(K[20:24])) / (256.0 * 5)
    delta = (xor4(K[24], K[25], K[26], K[27]) + sum(K[28:32])) / (256.0 * 5)
    x0 = (Q1 + Q2) / 2.0
    y0 = (Q2 + Q3) / 2.0
    z0 = (Q3 + Q4) / 2.0
    w0 = (Q4 + Q1) / 2.0
    return x0, y0, z0, w0, alpha, beta, gamma, delta

def generate_chaotic_sequences(img, x0, y0, z0, w0, alpha=10, beta=8/3, gamma=28, delta=0.1):
    """
    Generate chaotic sequences using a 4D system.
    """
    M, N = img.shape[:2]
    num_iter = M * N
    transient = 1000
    total_steps = num_iter + transient
    X = np.zeros(total_steps)
    Y = np.zeros(total_steps)
    Z = np.zeros(total_steps)
    W = np.zeros(total_steps)
    X[0], Y[0], Z[0], W[0] = x0, y0, z0, w0
    dt = 0.01
    for i in range(total_steps - 1):
        dx = alpha * (Y[i] - X[i]) + W[i]
        dy = gamma * X[i] - Y[i] - X[i] * Z[i]
        dz = X[i] * Y[i] - beta * Z[i]
        dw = -delta * X[i] * Z[i]
        X[i+1] = X[i] + dt * dx
        Y[i+1] = Y[i] + dt * dy
        Z[i+1] = Z[i] + dt * dz
        W[i+1] = W[i] + dt * dw
        # Clip to avoid numerical issues
        X[i+1] = np.clip(X[i+1], -1e10, 1e10)
        Y[i+1] = np.clip(Y[i+1], -1e10, 1e10)
        Z[i+1] = np.clip(Z[i+1], -1e10, 1e10)
        W[i+1] = np.clip(W[i+1], -1e10, 1e10)
    # Discard transient and scale to [0,255]
    X, Y, Z, W = X[transient:], Y[transient:], Z[transient:], W[transient:]
    X = np.mod(np.abs(X) * 1e6, 256).astype(np.uint8)
    Y = np.mod(np.abs(Y) * 1e6, 256).astype(np.uint8)
    Z = np.mod(np.abs(Z) * 1e6, 256).astype(np.uint8)
    W = np.mod(np.abs(W) * 1e6, 256).astype(np.uint8)
    return X, Y, Z, W

####################################
# Encryption / Decryption Functions (Grayscale)
####################################
def encrypt_image_channel(img, X_seq, Y_seq, Z_seq, W_seq):
    """
    Encrypt a single-channel (grayscale) image.
    """
    M, N = img.shape
    block_size = 8
    num_blocks_per_row = N // block_size
    permuted = np.zeros((M, N), dtype=np.uint8)
    # Bitwise Permutation using X_seq
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            block = img[i:i+block_size, j:j+block_size]
            start_idx = (i // block_size * num_blocks_per_row + j // block_size) * block_size
            perm_order = np.argsort(X_seq[start_idx:start_idx+block_size])
            bits = np.unpackbits(block[..., None], axis=-1)
            permuted_bits = bits[..., perm_order]
            permuted[i:i+block_size, j:j+block_size] = np.packbits(permuted_bits, axis=-1).reshape(block.shape)
    # XOR Diffusion using Y_seq
    Y_mod = np.resize(Y_seq, img.shape).astype(np.uint8)
    diffused = (permuted ^ Y_mod).astype(np.uint8)
    # S-Box Substitution using Z_seq
    sbox = np.argsort(Z_seq[:256])
    substituted = sbox[diffused].astype(np.uint8)
    # Chaotic XOR Diffusion using W_seq
    W_mod = np.resize(W_seq, img.shape).astype(np.uint8)
    ciphertext = (substituted ^ W_mod).astype(np.uint8)
    return ciphertext

def decrypt_image_channel(cipher, X_seq, Y_seq, Z_seq, W_seq):
    """
    Decrypt a single-channel (grayscale) image.
    """
    M, N = cipher.shape
    block_size = 8
    num_blocks_per_row = N // block_size
    # Reverse Chaotic XOR Diffusion using W_seq
    W_mod = np.resize(W_seq, (M, N)).astype(np.uint8)
    recovered1 = (cipher ^ W_mod).astype(np.uint8)
    # Reverse S-Box Substitution using Z_seq
    sbox = np.argsort(Z_seq[:256])
    inv_sbox = np.argsort(sbox)
    recovered2 = inv_sbox[recovered1].astype(np.uint8)
    # Reverse XOR Diffusion using Y_seq
    Y_mod = np.resize(Y_seq, (M, N)).astype(np.uint8)
    recovered3 = (recovered2 ^ Y_mod).astype(np.uint8)
    # Reverse Bitwise Permutation using X_seq
    recovered = np.zeros((M, N), dtype=np.uint8)
    for i in range(0, M, block_size):
        for j in range(0, N, block_size):
            block = recovered3[i:i+block_size, j:j+block_size]
            start_idx = (i // block_size * num_blocks_per_row + j // block_size) * block_size
            perm_order = np.argsort(X_seq[start_idx:start_idx+block_size])
            inv_perm = np.argsort(perm_order)
            bits = np.unpackbits(block[..., None], axis=-1)
            original_bits = bits[..., inv_perm]
            recovered[i:i+block_size, j:j+block_size] = np.packbits(original_bits, axis=-1).reshape(block.shape)

    return recovered

####################################
# Encryption / Decryption Functions (Color)
####################################
def encrypt_color_image(img, X_seq, Y_seq, Z_seq, W_seq):
    """
    Encrypt a color image by processing each channel separately.
    """
    channels = cv2.split(img)
    encrypted_channels = [encrypt_image_channel(ch, X_seq, Y_seq, Z_seq, W_seq) for ch in channels]
    return cv2.merge(encrypted_channels)

def decrypt_color_image(img, X_seq, Y_seq, Z_seq, W_seq):
    """
    Decrypt a color image by processing each channel separately.
    """
    channels = cv2.split(img)
    decrypted_channels = [decrypt_image_channel(ch, X_seq, Y_seq, Z_seq, W_seq) for ch in channels]
    return cv2.merge(decrypted_channels)

####################################
# Evaluation Metrics Functions
####################################
def compute_histogram(img):
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
    return hist

def shannon_entropy(img):
    hist = compute_histogram(img)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def shannon_entropy_color(img):
    """
    Compute the average Shannon entropy over the color channels.
    """
    channels = cv2.split(img)
    entropies = [shannon_entropy(ch) for ch in channels]
    return np.mean(entropies)

def correlation_coefficients(img):
    horiz = np.corrcoef(img[:, :-1].flatten(), img[:, 1:].flatten())[0, 1]
    vert  = np.corrcoef(img[:-1, :].flatten(), img[1:, :].flatten())[0, 1]
    diag  = np.corrcoef(img[:-1, :-1].flatten(), img[1:, 1:].flatten())[0, 1]
    return horiz, vert, diag

def npcr_uaci_gray(orig, proc):
    total = orig.size
    diff = (orig != proc).astype(int)
    npcr = np.sum(diff) / total * 100
    diff_intensity = np.abs(orig.astype(int) - proc.astype(int))
    uaci = np.sum(diff_intensity) / (255 * total) * 100
    return npcr, uaci

def npcr_uaci_color(orig, proc):
    channels_orig = cv2.split(orig)
    channels_proc = cv2.split(proc)
    npcr_vals, uaci_vals = [], []
    for o, p in zip(channels_orig, channels_proc):
        n, u = npcr_uaci_gray(o, p)
        npcr_vals.append(n)
        uaci_vals.append(u)
    return np.mean(npcr_vals), np.mean(uaci_vals)

def mse(imgA, imgB):
    return np.mean((imgA.astype("float") - imgB.astype("float")) ** 2)

def evaluate_attack(orig, proc, color=False):
    psnr_val = cv2.PSNR(orig, proc)
    if color:
        ssim_val = ssim(orig, proc, channel_axis=-1, win_size=7)
    else:
        ssim_val = ssim(orig, proc, win_size=7)
    return psnr_val, ssim_val

####################################
# Attack Simulation Functions
####################################
# Grayscale Attacks
def add_salt_pepper_noise(img, noise_level=0.05):
    noisy = img.copy()
    total = img.size
    num_noise = int(noise_level * total)
    coords = [np.random.randint(0, i, num_noise) for i in img.shape]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i, num_noise) for i in img.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

def add_gaussian_noise(img, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy = img.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def cropping_attack(img, crop_fraction=0.2):
    attacked = img.copy()
    h, w = img.shape[:2]
    ch = int(h * crop_fraction)
    cw = int(w * crop_fraction)
    sh = (h - ch) // 2
    sw = (w - cw) // 2
    if len(img.shape) == 2:
        attacked[sh:sh+ch, sw:sw+cw] = 0
    else:
        attacked[sh:sh+ch, sw:sw+cw, :] = 0
    return attacked

def add_bit_plane_noise_gray(img, noise_level=0.05, bit_plane=0):
    noisy = img.copy()
    total = img.size
    num_noise = int(noise_level * total)
    indices = np.unravel_index(np.random.choice(total, num_noise, replace=False), img.shape)
    noisy[indices] = noisy[indices] ^ (1 << bit_plane)
    return noisy

def add_median_filter_attack_gray(img, ksize=3):
    return cv2.medianBlur(img, ksize)

# Color Attacks
def add_salt_pepper_noise_color(img, noise_level=0.05):
    noisy = img.copy()
    h, w, c = img.shape
    total = h * w
    num_noise = int(noise_level * total)
    channels = cv2.split(noisy)
    noisy_channels = []
    for ch in channels:
        ch_noisy = ch.copy()
        coords = [np.random.randint(0, h, num_noise), np.random.randint(0, w, num_noise)]
        ch_noisy[coords[0], coords[1]] = 255
        coords = [np.random.randint(0, h, num_noise), np.random.randint(0, w, num_noise)]
        ch_noisy[coords[0], coords[1]] = 0
        noisy_channels.append(ch_noisy)
    return cv2.merge(noisy_channels)

def add_gaussian_noise_color(img, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy = img.astype(np.float32) + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_bit_plane_noise_color(img, noise_level=0.05, bit_plane=0):
    channels = cv2.split(img)
    noisy_channels = [add_bit_plane_noise_gray(ch, noise_level, bit_plane) for ch in channels]
    return cv2.merge(noisy_channels)

def add_median_filter_attack_color(img, ksize=3):
    channels = cv2.split(img)
    filtered = [cv2.medianBlur(ch, ksize) for ch in channels]
    return cv2.merge(filtered)

def cropping_attack_color(img, crop_fraction=0.2):
    attacked = img.copy()
    h, w, _ = img.shape
    ch = int(h * crop_fraction)
    cw = int(w * crop_fraction)
    sh = (h - ch) // 2
    sw = (w - cw) // 2
    attacked[sh:sh+ch, sw:sw+cw, :] = 0
    return attacked

####################################
# Main Function
####################################
def main():
    # Updated file paths:
    grayscale_path = r"C:\Users\vishn\Desktop\vishnu_final\images\30.png"
    color_path = r"C:\Users\vishn\Desktop\vishnu_final\images\012.jpg"
    
    # ===============================
    # Grayscale Image Processing
    # ===============================
    print("=== Grayscale Image Processing ===")
    plaintext_gray = load_image(grayscale_path, color=False)
    
    # Generate key parameters and chaotic sequences from grayscale image
    x0, y0, z0, w0, alpha, beta, gamma, delta = generate_key_params(plaintext_gray)
    print(f"Key Params: x0={x0:.4f}, y0={y0:.4f}, z0={z0:.4f}, w0={w0:.4f}")
    X_seq, Y_seq, Z_seq, W_seq = generate_chaotic_sequences(plaintext_gray, x0, y0, z0, w0, alpha, beta, gamma, delta)
    
    # Encrypt and Decrypt the grayscale image
    start_enc = time.time()
    ciphertext_gray = encrypt_image_channel(plaintext_gray, X_seq, Y_seq, Z_seq, W_seq)
    enc_time = time.time() - start_enc
    cv2.imwrite(os.path.join(output_folder, "encrypted_image.png"), ciphertext_gray)
    print(f"Grayscale Encryption Time: {enc_time:.4f} seconds")
    
    start_dec = time.time()
    decrypted_gray = decrypt_image_channel(ciphertext_gray, X_seq, Y_seq, Z_seq, W_seq)
    dec_time = time.time() - start_dec
    cv2.imwrite(os.path.join(output_folder, "decrypted_image.png"), decrypted_gray)
    print(f"Grayscale Decryption Time: {dec_time:.4f} seconds")
    
    # Evaluation Metrics for Grayscale
    entropy_plain = shannon_entropy(plaintext_gray)
    entropy_cipher = shannon_entropy(ciphertext_gray)
    horiz_corr, vert_corr, diag_corr = correlation_coefficients(ciphertext_gray)
    npcr_val, uaci_val = npcr_uaci_gray(plaintext_gray, ciphertext_gray)
    mse_val = mse(plaintext_gray, ciphertext_gray)
    psnr_val = cv2.PSNR(plaintext_gray, ciphertext_gray)
    ssim_val = ssim(plaintext_gray, ciphertext_gray, win_size=7)
    psnr_decrypted = cv2.PSNR(plaintext_gray, decrypted_gray)
    ssim_decrypted = ssim(plaintext_gray, decrypted_gray, win_size=7)
    
    print(f"Plaintext Entropy: {entropy_plain:.4f}")
    print(f"Ciphertext Entropy: {entropy_cipher:.4f}")
    print(f"Correlation Coefficients (H,V,D): {horiz_corr:.4f}, {vert_corr:.4f}, {diag_corr:.4f}")
    print(f"NPCR: {npcr_val:.2f}%, UACI: {uaci_val:.2f}%")
    print(f"MSE: {mse_val:.4f}")
    print(f"PSNR (Plain vs Cipher): {psnr_val:.4f} dB, SSIM: {ssim_val:.4f}")
    print(f"PSNR (Plain vs Decrypted): {psnr_decrypted:.4f} dB, SSIM: {ssim_decrypted:.4f}")
    
    if np.array_equal(plaintext_gray, decrypted_gray):
        print("Success: Decrypted grayscale image matches the original!")
    else:
        print("Error: Decrypted grayscale image does not match the original.")
    
    # Grayscale Attack Simulations
    # 1. Salt & Pepper Noise Attack
    noisy_sp_gray = add_salt_pepper_noise(ciphertext_gray, noise_level=0.05)
    cv2.imwrite(os.path.join(output_folder, "noisy_sp_encrypted.png"), noisy_sp_gray)
    decrypted_sp_gray = decrypt_image_channel(noisy_sp_gray, X_seq, Y_seq, Z_seq, W_seq)
    cv2.imwrite(os.path.join(output_folder, "decrypted_noisy_sp.png"), decrypted_sp_gray)
    psnr_sp, ssim_sp = evaluate_attack(plaintext_gray, decrypted_sp_gray, color=False)
    npcr_sp, uaci_sp = npcr_uaci_gray(plaintext_gray, decrypted_sp_gray)
    mse_sp = mse(plaintext_gray, decrypted_sp_gray)
    print("\nGrayscale Salt & Pepper Noise Attack:")
    print(f"  PSNR: {psnr_sp:.4f} dB, SSIM: {ssim_sp:.4f}, NPCR: {npcr_sp:.2f}%, UACI: {uaci_sp:.2f}%, MSE: {mse_sp:.4f}")
    
    # 2. Gaussian Noise Attack
    noisy_gauss_gray = add_gaussian_noise(ciphertext_gray, mean=0, sigma=25)
    cv2.imwrite(os.path.join(output_folder, "noisy_gauss_encrypted.png"), noisy_gauss_gray)
    decrypted_gauss_gray = decrypt_image_channel(noisy_gauss_gray, X_seq, Y_seq, Z_seq, W_seq)
    cv2.imwrite(os.path.join(output_folder, "decrypted_noisy_gauss.png"), decrypted_gauss_gray)
    psnr_gauss, ssim_gauss = evaluate_attack(plaintext_gray, decrypted_gauss_gray, color=False)
    npcr_gauss, uaci_gauss = npcr_uaci_gray(plaintext_gray, decrypted_gauss_gray)
    mse_gauss = mse(plaintext_gray, decrypted_gauss_gray)  # Added MSE calculation
    print("\nGrayscale Gaussian Noise Attack:")
    print(f"  PSNR: {psnr_gauss:.4f} dB, SSIM: {ssim_gauss:.4f}, NPCR: {npcr_gauss:.2f}%, UACI: {uaci_gauss:.2f}%, MSE: {mse_gauss:.4f}")

    # 3. Cropping Attack
    cropped_gray = cropping_attack(ciphertext_gray, crop_fraction=0.2)
    cv2.imwrite(os.path.join(output_folder, "cropped_encrypted.png"), cropped_gray)
    decrypted_cropped_gray = decrypt_image_channel(cropped_gray, X_seq, Y_seq, Z_seq, W_seq)
    cv2.imwrite(os.path.join(output_folder, "decrypted_cropped.png"), decrypted_cropped_gray)
    psnr_crop, ssim_crop = evaluate_attack(plaintext_gray, decrypted_cropped_gray, color=False)
    npcr_crop, uaci_crop = npcr_uaci_gray(plaintext_gray, decrypted_cropped_gray)
    mse_crop = mse(plaintext_gray, decrypted_cropped_gray)  # Added MSE calculation
    print("\nGrayscale Cropping Attack:")
    print(f"  PSNR: {psnr_crop:.4f} dB, SSIM: {ssim_crop:.4f}, NPCR: {npcr_crop:.2f}%, UACI: {uaci_crop:.2f}%, MSE: {mse_crop:.4f}")

    
    # Additional Grayscale Attacks
    # Bit Plane Noise Attack
    noisy_bp_gray = add_bit_plane_noise_gray(ciphertext_gray, noise_level=0.05, bit_plane=0)
    psnr_bp, ssim_bp = evaluate_attack(plaintext_gray, noisy_bp_gray, color=False)
    npcr_bp, uaci_bp = npcr_uaci_gray(plaintext_gray, noisy_bp_gray)
    mse_bp = mse(plaintext_gray, noisy_bp_gray)
    print("\nGrayscale Bit Plane Noise Attack:")
    print(f"  PSNR: {psnr_bp:.4f} dB, SSIM: {ssim_bp:.4f}, NPCR: {npcr_bp:.2f}%, UACI: {uaci_bp:.2f}%, MSE: {mse_bp:.4f}")
    
    # Median Filtering Attack
    noisy_med_gray = add_median_filter_attack_gray(ciphertext_gray, ksize=3)
    psnr_med, ssim_med = evaluate_attack(plaintext_gray, noisy_med_gray, color=False)
    npcr_med, uaci_med = npcr_uaci_gray(plaintext_gray, noisy_med_gray)
    mse_med = mse(plaintext_gray, noisy_med_gray)
    print("\nGrayscale Median Filtering Attack:")
    print(f"  PSNR: {psnr_med:.4f} dB, SSIM: {ssim_med:.4f}, NPCR: {npcr_med:.2f}%, UACI: {uaci_med:.2f}%, MSE: {mse_med:.4f}")
    
    # ===============================
    # Color Image Processing
    # ===============================
    print("\n=== Color Image Processing ===")
    color_img = load_image(color_path, color=True)
    
    # Use grayscale conversion for key generation
    gray_for_key = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    x0, y0, z0, w0, alpha, beta, gamma, delta = generate_key_params(gray_for_key)
    print(f"Key Params (Color): x0={x0:.4f}, y0={y0:.4f}, z0={z0:.4f}, w0={w0:.4f}")
    X_seq, Y_seq, Z_seq, W_seq = generate_chaotic_sequences(gray_for_key, x0, y0, z0, w0, alpha, beta, gamma, delta)
    
    # Encrypt and Decrypt the color image
    start_enc_color = time.time()
    encrypted_color = encrypt_color_image(color_img, X_seq, Y_seq, Z_seq, W_seq)
    enc_time_color = time.time() - start_enc_color
    cv2.imwrite(os.path.join(output_folder, "encrypted_color.png"), encrypted_color)
    print(f"Color Encryption Time: {enc_time_color:.4f} seconds")
    
    start_dec_color = time.time()
    decrypted_color = decrypt_color_image(encrypted_color, X_seq, Y_seq, Z_seq, W_seq)
    dec_time_color = time.time() - start_dec_color
    cv2.imwrite(os.path.join(output_folder, "decrypted_color.png"), decrypted_color)
    print(f"Color Decryption Time: {dec_time_color:.4f} seconds")
    
    # Color Image Entropy
    color_entropy_orig = shannon_entropy_color(color_img)
    color_entropy_enc = shannon_entropy_color(encrypted_color)
    print(f"Original Color Image Entropy: {color_entropy_orig:.4f}")
    print(f"Encrypted Color Image Entropy: {color_entropy_enc:.4f}")
    
    if np.array_equal(color_img, decrypted_color):
        print("Success: Decrypted color image matches the original!")
    else:
        print("Error: Decrypted color image does not match the original.")
    
    # Evaluation Metrics for Color Encryption
    psnr_color = cv2.PSNR(color_img, encrypted_color)
    ssim_color = ssim(color_img, encrypted_color, channel_axis=-1, win_size=7)
    npcr_color, uaci_color = npcr_uaci_color(color_img, encrypted_color)
    mse_color = mse(color_img, encrypted_color)
    print("\nColor Encryption Evaluation:")
    print(f"  PSNR: {psnr_color:.4f} dB, SSIM: {ssim_color:.4f}")
    print(f"  NPCR: {npcr_color:.2f}%, UACI: {uaci_color:.2f}%, MSE: {mse_color:.4f}")
    print(f"  Color Image Entropy: {color_entropy_orig:.4f}")
    
    # Color Attack Simulations
    # 1. Salt & Pepper Noise Attack (Color)
    noisy_sp_color = add_salt_pepper_noise_color(encrypted_color, noise_level=0.05)
    cv2.imwrite(os.path.join(output_folder, "noisy_sp_color_encrypted.png"), noisy_sp_color)
    decrypted_sp_color = decrypt_color_image(noisy_sp_color, X_seq, Y_seq, Z_seq, W_seq)
    cv2.imwrite(os.path.join(output_folder, "decrypted_noisy_sp_color.png"), decrypted_sp_color)
    psnr_sp_color, ssim_sp_color = evaluate_attack(color_img, decrypted_sp_color, color=True)
    npcr_sp_color, uaci_sp_color = npcr_uaci_color(color_img, decrypted_sp_color)
    mse_sp_color = mse(color_img, decrypted_sp_color)
    print("\nColor Salt & Pepper Noise Attack:")
    print(f"  PSNR: {psnr_sp_color:.4f} dB, SSIM: {ssim_sp_color:.4f}")
    print(f"  NPCR: {npcr_sp_color:.2f}%, UACI: {uaci_sp_color:.2f}%, MSE: {mse_sp_color:.4f}")
    
    # 2. Gaussian Noise Attack (Color)
    noisy_gauss_color = add_gaussian_noise_color(encrypted_color, mean=0, sigma=25)
    cv2.imwrite(os.path.join(output_folder, "noisy_gauss_color_encrypted.png"), noisy_gauss_color)
    decrypted_gauss_color = decrypt_color_image(noisy_gauss_color, X_seq, Y_seq, Z_seq, W_seq)
    cv2.imwrite(os.path.join(output_folder, "decrypted_gauss_color.png"), decrypted_gauss_color)
    psnr_gauss_color, ssim_gauss_color = evaluate_attack(color_img, decrypted_gauss_color, color=True)
    npcr_gauss_color, uaci_gauss_color = npcr_uaci_color(color_img, decrypted_gauss_color)
    mse_gauss_color = mse(color_img, decrypted_gauss_color)
    print("\nColor Gaussian Noise Attack:")
    print(f"  PSNR: {psnr_gauss_color:.4f} dB, SSIM: {ssim_gauss_color:.4f}")
    print(f"  NPCR: {npcr_gauss_color:.2f}%, UACI: {uaci_gauss_color:.2f}%, MSE: {mse_gauss_color:.4f}")
    
    # 3. Cropping Attack (Color)
    cropped_color = cropping_attack_color(encrypted_color, crop_fraction=0.2)
    cv2.imwrite(os.path.join(output_folder, "cropped_color_encrypted.png"), cropped_color)
    decrypted_cropped_color = decrypt_color_image(cropped_color, X_seq, Y_seq, Z_seq, W_seq)
    cv2.imwrite(os.path.join(output_folder, "decrypted_cropped_color.png"), decrypted_cropped_color)
    psnr_crop_color, ssim_crop_color = evaluate_attack(color_img, decrypted_cropped_color, color=True)
    npcr_crop_color, uaci_crop_color = npcr_uaci_color(color_img, decrypted_cropped_color)
    mse_crop_color = mse(color_img, decrypted_cropped_color)
    print("\nColor Cropping Attack:")
    print(f"  PSNR: {psnr_crop_color:.4f} dB, SSIM: {ssim_crop_color:.4f}")
    print(f"  NPCR: {npcr_crop_color:.2f}%, UACI: {uaci_crop_color:.2f}%, MSE: {mse_crop_color:.4f}")
    
    # Additional Color Attacks
    # Bit Plane Noise Attack (Color)
    noisy_bp_color = add_bit_plane_noise_color(encrypted_color, noise_level=0.05, bit_plane=0)
    psnr_bp_color, ssim_bp_color = evaluate_attack(color_img, noisy_bp_color, color=True)
    npcr_bp_color, uaci_bp_color = npcr_uaci_color(color_img, noisy_bp_color)
    mse_bp_color = mse(color_img, noisy_bp_color)
    print("\nColor Bit Plane Noise Attack:")
    print(f"  PSNR: {psnr_bp_color:.4f} dB, SSIM: {ssim_bp_color:.4f}")
    print(f"  NPCR: {npcr_bp_color:.2f}%, UACI: {uaci_bp_color:.2f}%, MSE: {mse_bp_color:.4f}")
    
    # Median Filtering Attack (Color)
    noisy_med_color = add_median_filter_attack_color(encrypted_color, ksize=3)
    psnr_med_color, ssim_med_color = evaluate_attack(color_img, noisy_med_color, color=True)
    npcr_med_color, uaci_med_color = npcr_uaci_color(color_img, noisy_med_color)
    mse_med_color = mse(color_img, noisy_med_color)
    print("\nColor Median Filtering Attack:")
    print(f"  PSNR: {psnr_med_color:.4f} dB, SSIM: {ssim_med_color:.4f}")
    print(f"  NPCR: {npcr_med_color:.2f}%, UACI: {uaci_med_color:.2f}%, MSE: {mse_med_color:.4f}")

if __name__ == "__main__":
    main()
