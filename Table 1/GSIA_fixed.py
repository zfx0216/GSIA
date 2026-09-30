import os
import json
import time
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
# Transform
# ===========================
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


# ===========================
# Path
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
# Load class index
# ===========================
with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)


# ===========================
# Load model
# ===========================
model = convnext_tiny(num_classes=1000).to(device)

model.load_state_dict(
    torch.load(weight_file_absolute_path, map_location=device)
)

model.eval()

criterion = nn.CrossEntropyLoss()


# ===========================
# Functions
# ===========================
@torch.no_grad()
def predict_tensor(img_tensor):
    output = model(img_tensor)
    pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
    return pred.item(), output


def generate_fixed_mask(grad):
    """
    Generate fixed gradient mask
    grad: [C,H,W]
    """
    threshold = np.mean(np.abs(grad))
    mask = (np.abs(grad) > threshold).astype(np.float32)
    return mask


def l2_project(delta, eps):
    """
    L2 norm projection
    """
    norm = torch.norm(delta, p=2, dim=(1, 2, 3), keepdim=True)
    factor = torch.min(torch.ones_like(norm),
                       eps / (norm + 1e-8))
    return delta * factor



# ===========================
# Main
# ===========================
image_files = [
    f for f in os.listdir(actual_images_folder_absolute_path)
    if f.endswith(".png")
]

image_num = 0
success_num = 0

start_time = time.time()


for image_file in image_files:

    print(f"\nProcessing: {image_file}")

    image_num += 1

    img_path = os.path.join(
        actual_images_folder_absolute_path,
        image_file
    )

    img = Image.open(img_path).convert("RGB").resize((224, 224))

    orig_tensor = (
        data_transform(img)
        .unsqueeze(0)
        .to(device)
    )

    true_label, _ = predict_tensor(orig_tensor)

    delta = torch.zeros_like(orig_tensor).to(device)

    fixed_mask = None

    success = False
    iteration = 0


    while not success and iteration <= max_num_iterative:

        iteration += 1

        adv_tensor = (
            orig_tensor +
            delta.detach()
        )

        adv_tensor.requires_grad = True


        output = model(adv_tensor)

        loss = criterion(
            output,
            torch.tensor([true_label], device=device)
        )


        model.zero_grad()
        loss.backward()


        grad = adv_tensor.grad.detach()


        # ===========================
        # Generate fixed mask once
        # ===========================
        if fixed_mask is None:

            grad_np = (
                grad.cpu()
                .numpy()
                .squeeze(0)
            )

            mask_np = generate_fixed_mask(grad_np)

            fixed_mask = (
                torch.from_numpy(mask_np)
                .unsqueeze(0)
                .to(device)
            )

            ratio = (
                fixed_mask.sum()
                /
                fixed_mask.numel()
            )

            print(
                f"Fixed mask generated, selected ratio: {ratio.item()*100:.2f}%"
            )


        # ===========================
        # Update perturbation
        # ===========================
        masked_grad = grad * fixed_mask

        delta = (
            delta +
            iteration_step_size *
            torch.sign(masked_grad)
        )

        delta = l2_project(delta, eps2)


        adv_tensor = orig_tensor + delta

        pred_label, _ = predict_tensor(adv_tensor)

        l2 = torch.norm(delta, p=2).item()

        print(
            f"Iter {iteration}, L2={l2:.4f}"
        )


        if pred_label != true_label:

            print("Attack success!")

            success_num += 1
            success = True



    # ===========================
    # Save adversarial image
    # ===========================
    adv_img = inv_normalize(
        adv_tensor.squeeze(0)
    ).detach().cpu()

    adv_img = torch.clamp(
        adv_img,
        0,
        1
    )

    adv_img = transforms.ToPILImage()(adv_img)

    adv_img.save(
        os.path.join(output_directory, image_file)
    )


end_time = time.time()


print("\n==============================")
print(f"Success: {success_num}/{image_num}")
print(f"ASR: {success_num/image_num*100:.2f}%")
print(
    f"Average time: {(end_time-start_time)/image_num:.4f}s/image"
)