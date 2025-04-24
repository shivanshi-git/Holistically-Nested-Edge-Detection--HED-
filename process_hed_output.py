import cv2
import matplotlib.pyplot as plt

# ======= Load Images =======
original_path = r"image\00003_Mask.jpg"
edge_map_path = r"result/mask/output_mask.png"

# Read original image
original = cv2.imread(original_path)
edge_map = cv2.imread(edge_map_path, cv2.IMREAD_GRAYSCALE)

# Resize edge map to match original if needed
edge_map = cv2.resize(edge_map, (original.shape[1], original.shape[0]))

# ======= Overlay Edge Map on Original =======
edge_map_color = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
blended = cv2.addWeighted(original, 0.8, edge_map_color, 0.5, 0)

# Save overlay
cv2.imwrite("overlay_unmask.png", blended)

# Convert BGR to RGB for Matplotlib
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
edge_rgb = cv2.cvtColor(edge_map_color, cv2.COLOR_BGR2RGB)

# ======= Plot Images Side-by-Side =======
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(edge_map, cmap='gray')
plt.title("Edge Map")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(blended_rgb)
plt.title("Overlay")
plt.axis('off')

plt.tight_layout()
plt.show()
