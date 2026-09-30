import os
import json
import time
import shutil
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny

# ===========================
# Device
# ===========================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===========================
# Image preprocessing
# ===========================
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# ===========================
# Path configuration
# ===========================
index_file_absolute_path = r"..."
weight_file_absolute_path = r"..."

actual_images_folder_absolute_path = r"..."
output_directory = r"..."

os.makedirs(output_directory, exist_ok=True)

# ===========================
# Attack parameters
# ===========================
eps2 = 8
iteration_step_size = 0.017
max_num_iterative = 1000

# ===========================
# Load class indices
# ===========================
with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)

# ===========================
# Model (load only once)
# ===========================
model = convnext_tiny(num_classes=1000).to(device)
model.load_state_dict(torch.load(weight_file_absolute_path, map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

# ===========================
# Utility functions
# ===========================

@torch.no_grad()
def predict_tensor(img_tensor):
    output = model(img_tensor)
    prob = torch.softmax(output, dim=1)
    return torch.argmax(prob, dim=1).item(), output

def calculate_gradient_tensor(img_tensor, label):
    img_tensor.requires_grad_(True)
    output = model(img_tensor)
    loss = criterion(output, torch.tensor([label], device=device))
    grad = torch.autograd.grad(loss, img_tensor)[0]
    return grad.squeeze(0).cpu().numpy()

def divide_matrix(input_matrix):
    threshold = np.mean(input_matrix)
    return (input_matrix > threshold).astype(np.uint8)

# def apply_pixel_change(img_np, grad_np, mask_np):
#     """
#     img_np: (H,W,3)
#     grad_np: (3,H,W)
#     mask_np: (3,H,W)
#     """
#     delta = np.sign(grad_np) * mask_np
#     img_np = img_np + delta.transpose(1, 2, 0)
#     return np.clip(img_np, 0, 255)


def apply_pixel_change(img_np, grad_np, mask_np, original_img_np):
    # Add perturbations only where mask=1
    delta = np.sign(grad_np) * mask_np * iteration_step_size
    img_np_new = img_np + delta.transpose(1, 2, 0)
    img_np_new = np.clip(img_np_new, 0, 255)

    # Restore original-image values where mask=0
    # Convert the mask from (3, H, W) to (H, W, 3) to match the image
    mask_hwc = mask_np.transpose(1, 2, 0)  # (H, W, 3)

    # Use original-image values where mask=0
    # mask_hwc == 0 means the pixel must remain unchanged
    img_np_new[mask_hwc == 0] = original_img_np[mask_hwc == 0]

    return img_np_new
# ===========================
# Main workflow
# ===========================
image_files = (
    [f for f in os.listdir(actual_images_folder_absolute_path) if f.endswith('.png')]
)

image_num = 0
success_num = 0
start_time = time.time()

for image_file in image_files:

    print(f"\nProcessing image: {image_file}")
    image_num += 1

    img_path = os.path.join(actual_images_folder_absolute_path, image_file)

    # Original image
    img_pil = Image.open(img_path).convert("RGB").resize((224, 224))
    original_img_np = np.array(img_pil).astype(np.float64)
    img_np = original_img_np.copy()

    orig_norm_tensor = data_transform(img_pil).unsqueeze(0).to(device)
    img_tensor = data_transform(img_pil).unsqueeze(0).to(device)

    prev_img_np = original_img_np.copy()

    # Original class
    actual_label, _ = predict_tensor(img_tensor)

    flag = 0
    num_iter = 0

    while flag == 0 and num_iter <= max_num_iterative:

        num_iter += 1
        print(f"  Iteration: {num_iter}")

        # Gradient
        grad = calculate_gradient_tensor(img_tensor, actual_label)

        # Saliency mask
        mask = divide_matrix(np.abs(grad))

        # Pixel update
        img_np = apply_pixel_change(img_np, grad, mask, original_img_np)

        # ========== Calculate the current perturbation L2 norm in 0-255 pixel space ==========
        adv_pil = Image.fromarray(img_np.astype(np.uint8))
        adv_norm_tensor = data_transform(adv_pil).unsqueeze(0).to(device)
        delta_norm = adv_norm_tensor - orig_norm_tensor
        l2_norm = torch.norm(delta_norm, p=2).item()

        print(f"Normalized-space L2 norm: {l2_norm:.4f}, eps2={eps2}")

        print(f"Current L2 norm: {l2_norm:.4f}, eps2={eps2}")
        if l2_norm > eps2:
            print(f"    !!! L2 exceeds the threshold; stop and use the previous result")
            img_np = prev_img_np.copy()
            break

        # Within the threshold; save this image as the previous result and continue
        prev_img_np = img_np.copy()

        # Update the tensor
        img_tensor = data_transform(
            Image.fromarray(img_np.astype(np.uint8))
        ).unsqueeze(0).to(device)

        # Prediction
        pred_label, _ = predict_tensor(img_tensor)

        if pred_label != actual_label:
            print("Untargeted attack succeeded!")
            success_num += 1
            flag = 1

        # Save the current iteration result
        save_path = os.path.join(output_directory, f"{image_file}")
        Image.fromarray(img_np.astype(np.uint8)).save(save_path)

end_time = time.time()
print("\n====================================")
print(f"Successful attacks: {success_num}/{image_num}")
print(f"Average time: {(end_time - start_time) / image_num:.4f} seconds/image")
