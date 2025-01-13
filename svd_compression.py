import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
import streamlit as st
from io import BytesIO

def load_image(path, gray=False):
    img = imread(path)
    if gray:
        img = rgb2gray(img)
    return img / 255.0  # Normalize to [0, 1]

def svd_compression(image, num_singular_values):
    U, S, V = np.linalg.svd(image, full_matrices=False)
    U_reduced = U[:, :num_singular_values]
    S_reduced = np.diag(S[:num_singular_values])
    V_reduced = V[:num_singular_values, :]
    compressed_image = np.dot(U_reduced, np.dot(S_reduced, V_reduced))
    return compressed_image, U, S, V

def svd_compression_rgb(image, num_singular_values):
    channels = []
    for i in range(image.shape[2]):  # R, G, B channels
        compressed_channel, _, _, _ = svd_compression(image[:, :, i], num_singular_values)
        channels.append(compressed_channel)
    return np.clip(np.stack(channels, axis=2), 0, 1)

def compute_metrics(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    ssim_value = sum(ssim(original[:, :, i], compressed[:, :, i], data_range=compressed[:, :, i].max() - compressed[:, :, i].min()) for i in range(3)) / 3
    return mse, ssim_value

def visualize_results(original, compressed, num_singular_values, mse, ssim_value):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    ax[1].imshow(compressed)
    ax[1].set_title(f"Compressed (k={num_singular_values})\nMSE: {mse:.2f}, SSIM: {ssim_value:.2f}")
    ax[1].axis("off")
    
    plt.tight_layout()
    plt.show()

# Main script
# image = load_image('cow.jpg', gray=False) 
# num_singular_values = 50

# compressed_image = svd_compression_rgb(image, num_singular_values)

# mse, ssim_value = compute_metrics(image, compressed_image)
# print(f"MSE: {mse:.2f}, SSIM: {ssim_value:.2f}")

# visualize_results(image, compressed_image, num_singular_values, mse, ssim_value)



def save_image_as_bytes(image):
    buf = BytesIO()
    plt.imsave(buf, image, format="jpg", cmap='gray')
    buf.seek(0)
    return buf

# Streamlit app code
st.title("Image Compression using SVD")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = load_image(uploaded_file, gray=False)
    
    # Show original image
    st.image(image, caption="Original Image", use_container_width=True)

    # User input for the number of singular values
    num_singular_values = st.slider("Select number of singular values (k)", min_value=1, max_value=100, value=50)

    # Perform SVD compression
    compressed_image = svd_compression_rgb(image, num_singular_values)

    # Compute metrics
    mse, ssim_value = compute_metrics(image, compressed_image)
    
    # Show compressed image
    st.image(compressed_image, caption=f"Compressed Image (k={num_singular_values})", use_container_width=True)
    
    # Display metrics
    st.write(f"MSE: {mse:.2f}")
    st.write(f"SSIM: {ssim_value:.2f}")

    # Save compressed image as bytes for download
    compressed_image_bytes = save_image_as_bytes(compressed_image)

    # Download button for the compressed image
    st.download_button(
        label="Download Compressed Image",
        data=compressed_image_bytes,
        file_name="compressed_image.png",
        mime="image/png"
    )