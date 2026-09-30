import numpy as np
import os
from PIL import Image

# ImageNet normalization parameters
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Original-image folder
folder_origin = r"..."
# Attack folders: (name, folder path)
attack_folders = [
    ("1", r"..."),
    ("2", r"..."),
    ("3", r"...")
]

max_possible_l2_norm = np.sqrt(3 * 255**2 * 224 * 224)

# Store L2 values for each attack folder to calculate mean and std
results = {}
for name, folder in attack_folders:
    results[name] = {
        "l2_255_list": [],
        "l2_01_list": [],
        "l2_norm_list": [],
        "max_l2_255": 0.0,
        "max_l2_01": 0.0,
        "max_l2_norm": 0.0,
    }

if not os.path.exists(folder_origin):
    print(f"Original-image folder {folder_origin} does not exist.")
else:
    # Get all PNG filenames in the original-image folder
    ori_file_list = [f for f in os.listdir(folder_origin) if f.lower().endswith(".png")]
    print(f"Original-image folder PNG files found: {len(ori_file_list)}")

    for name, folder_adv in attack_folders:
        if not os.path.exists(folder_adv):
            print(f"\n[{name}] folder {folder_adv} does not exist; skipping!")
            continue
        print(f"\n===== Processing {name} =====")
        res = results[name]
        match_cnt = 0

        for filename in ori_file_list:
            file_ori = os.path.join(folder_origin, filename)
            file_adv = os.path.join(folder_adv, filename)

            if not os.path.exists(file_adv):
                print(f"Skipping missing adversarial sample: {filename}")
                continue

            try:
                img_ori = Image.open(file_ori).convert("RGB")
                img_adv = Image.open(file_adv).convert("RGB")
                if img_ori.size != img_adv.size:
                    img_adv = img_adv.resize(img_ori.size)

                arr_ori = np.array(img_ori, dtype=np.float32)
                arr_adv = np.array(img_adv, dtype=np.float32)

                # 1. 0-255 pixel space
                delta_255 = arr_ori - arr_adv
                l2_255 = np.linalg.norm(delta_255)

                # 2. 0-1 normalized space
                ori_01 = arr_ori / 255.0
                adv_01 = arr_adv / 255.0
                delta_01 = ori_01 - adv_01
                l2_01 = np.linalg.norm(delta_01)

                # 3. ImageNet normalized space
                ori_norm = (ori_01 - mean) / std
                adv_norm = (adv_01 - mean) / std
                delta_norm = ori_norm - adv_norm
                l2_norm = np.linalg.norm(delta_norm)

                res["l2_255_list"].append(l2_255)
                res["l2_01_list"].append(l2_01)
                res["l2_norm_list"].append(l2_norm)

                res["max_l2_255"] = max(res["max_l2_255"], l2_255)
                res["max_l2_01"] = max(res["max_l2_01"], l2_01)
                res["max_l2_norm"] = max(res["max_l2_norm"], l2_norm)

                match_cnt += 1
                print(f"{filename} | 0‑255 L2:{l2_255:.2f} | 0‑1 L2:{l2_01:.4f} | Norm L2:{l2_norm:.4f}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        print(f"[{name}]Matched samples: {match_cnt}")

    # ========== Summarize all results ==========
    print("\n" + "="*70)
    print("Summary statistics (compared with original images)")
    print("="*70)

    # Combine data from all three folders
    all_l2_255 = []
    all_l2_01 = []
    all_l2_norm = []

    for name, _ in attack_folders:
        res = results[name]
        arr_255 = np.array(res["l2_255_list"])
        arr_01 = np.array(res["l2_01_list"])
        arr_norm = np.array(res["l2_norm_list"])
        n = len(arr_255)

        # Collect all data for overall statistics
        all_l2_255.extend(res["l2_255_list"])
        all_l2_01.extend(res["l2_01_list"])
        all_l2_norm.extend(res["l2_norm_list"])

        if n == 0:
            print(f"\n[{name}]No valid matched samples")
            continue

        print(f"\n[{name}]Valid samples: {n}")
        print(f"0‑255 space:  max={res['max_l2_255']:.2f}, mean={arr_255.mean():.2f}, std={arr_255.std():.2f}")
        print(f"0‑1 space:    max={res['max_l2_01']:.4f}, mean={arr_01.mean():.4f}, std={arr_01.std():.4f}")
        print(f"ImageNet normalized space:  max={res['max_l2_norm']:.4f}, mean={arr_norm.mean():.4f}, std={arr_norm.std():.4f}")

    # Report overall statistics across three folders
    print("\n" + "="*70)
    print("===== Overall statistics across three folders =====")
    print("="*70)
    arr_all_255 = np.array(all_l2_255)
    arr_all_01 = np.array(all_l2_01)
    arr_all_norm = np.array(all_l2_norm)
    total_n = len(arr_all_255)
    if total_n == 0:
        print("No valid samples in the combined set!")
    else:
        print(f"Total combined samples: {total_n}")
        print(f"0‑255 space:  mean={arr_all_255.mean():.2f}, std={arr_all_255.std():.2f}")
        print(f"0‑1 space:    mean={arr_all_01.mean():.4f}, std={arr_all_01.std():.4f}")
        print(f"ImageNet normalized space:  mean={arr_all_norm.mean():.5f}, std={arr_all_norm.std():.5f}")