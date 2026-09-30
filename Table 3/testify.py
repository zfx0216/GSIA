import torch
from torchvision.models import resnet18
from torchvision import transforms
import os
import json
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    original = []
    modification = []
    # Store names of successful and failed images
    success_names = []
    fail_names = []

    # Image folder paths
    img_folder_path = r"..."
    assert os.path.exists(img_folder_path), "Folder '{}' does not exist".format(img_folder_path)
    img_modify_folder_path = r"..."
    assert os.path.exists(img_modify_folder_path), "Folder '{}' does not exist".format(img_modify_folder_path)

    # Data preprocessing
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Read the class-index file
    json_path = r"..."
    assert os.path.exists(json_path), "File '{}' does not exist".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create and load the model
    model = resnet18(num_classes=1000).to(device)
    weights_path = r"..."
    assert os.path.exists(weights_path), "File '{}' does not exist".format(weights_path)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    # Read original images, predict labels, and save filenames
    print("===== Original-image predictions =====")
    img_name_list_ori = os.listdir(img_folder_path)
    for img_name in img_name_list_ori:
        img_path = os.path.join(img_folder_path, img_name)
        img = Image.open(img_path)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0).to(device)

        o_prediction = model(img.to(device))
        o_label_index = torch.argmax(o_prediction, dim=1).item()
        print(f'{img_name} original: {o_label_index}')
        original.append(o_label_index)

    # Read adversarial images, predict labels, and save filenames
    print("\n===== Adversarial-image predictions =====")
    img_name_list_mod = os.listdir(img_modify_folder_path)
    for img_name in img_name_list_mod:
        img_path = os.path.join(img_modify_folder_path, img_name)
        img = Image.open(img_path).convert("RGB")
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0).to(device)

        m_prediction = model(img.to(device))
        m_label_index = torch.argmax(m_prediction, dim=1).item()
        print(f'{img_name} modification: {m_label_index}')
        modification.append(m_label_index)

    # Count successful and failed attacks
    total_num = len(original)
    success_cnt = 0
    fail_cnt = 0
    for idx in range(total_num):
        ori_label = original[idx]
        mod_label = modification[idx]
        img_name = img_name_list_ori[idx]  # Use the corresponding original filename (same as the adversarial image)
        if ori_label != mod_label:
            # Changed label = successful attack
            success_cnt += 1
            success_names.append(img_name)
        else:
            # Unchanged label = failed attack
            fail_cnt += 1
            fail_names.append(img_name)

    # Print statistics
    print("\n===== Attack summary =====")
    attack_rate = success_cnt / total_num * 100
    print(f"Total images: {total_num}")
    print(f"Successful attacks: {success_cnt}")
    print(f"Failed attacks: {fail_cnt}")
    print(f"Attack success rate: {attack_rate:.2f} %")

    # Print filenames of all failed attacks
    if fail_names:
        print("\n===== Failed-attack images =====")
        for name in fail_names:
            print(name)
    else:
        print("\nAll attacks succeeded!")

if __name__ == "__main__":
    main()