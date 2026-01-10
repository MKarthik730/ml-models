import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img=Image.open(r"butterfly.jpg").convert("L")
original=np.array(img,dtype=float)
k_values=[10,50,100,200]


U,s,Vt=np.linalg.svd(original,full_matrices=False)
def compress_k(U,s,Vt,k):
    return U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]
plt.figure(figsize=(15, 6))
plt.subplot(1, len(k_values) + 1, 1)
plt.imshow(original,cmap="gray")
plt.title("original")

for i, k in enumerate(k_values, start=2):
    compressed=compress_k(U,s,Vt,k)
    plt.subplot(1, len(k_values) + 1, i)
    plt.imshow(compressed, cmap="gray")
    plt.title(f"k = {k}")
    plt.axis("off")
plt.show()
print("done")