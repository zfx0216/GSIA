import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.io import imread


def calculate_psnr_multi(origin_folder, attack_folders):
    results = {}
    # Read the list of original PNG files once
    if not os.path.exists(origin_folder):
        print(f"Original-image folderdoes not exist: {origin_folder}")
        return results
    img_files = [f for f in os.listdir(origin_folder) if f.lower().endswith('.png')]
    print(f"\nOriginal-image folderPNG file count: {len(img_files)}")

    # Initialize storage for each group
    for name, adv_folder in attack_folders:
        results[name] = {
            "psnr_list": [],
            "max_psnr": 0.0,
            "min_psnr": float("inf")
        }
        if not os.path.exists(adv_folder):
            print(f"\n[{name}] folder {adv_folder} does not exist; skipping!")
            continue

        print(f"\n===== Processing {name} =====")
        res = results[name]

        for image_file in img_files:
            try:
                ori_path = os.path.join(origin_folder, image_file)
                adv_path = os.path.join(adv_folder, image_file)
                if not os.path.exists(adv_path):
                    continue

                img_ori = imread(ori_path)
                img_adv = imread(adv_path)

                # Skip images with mismatched dimensions
                if img_ori.shape != img_adv.shape:
                    print(f"{image_file}: original and adversarial image dimensions differ; skipping")
                    continue

                psnr_val = psnr(img_ori, img_adv)
                res["psnr_list"].append(psnr_val)
                res["max_psnr"] = max(res["max_psnr"], psnr_val)
                res["min_psnr"] = min(res["min_psnr"], psnr_val)
                print(f"{image_file} PSNR: {psnr_val:.5f}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    # ========== Print the summary ==========
    print("\n" + "="*60)
    print("PSNR summary (compared with original images)")
    print("="*60)

    all_psnr = []  # Collect PSNR values from all three groups for overall statistics

    for name, _ in attack_folders:
        res = results[name]
        psnr_arr = np.array(res["psnr_list"])
        n = len(psnr_arr)
        all_psnr.extend(res["psnr_list"])

        if n == 0:
            print(f"\n[{name}]No valid samples")
            continue
        mean_psnr = psnr_arr.mean()
        std_psnr = psnr_arr.std()
        print(f"\n[{name}]Valid samples: {n}")
        print(f"PSNR: {mean_psnr:.5f}±{std_psnr:.5f}, max={res['max_psnr']:.5f}, min={res['min_psnr']:.5f}")

    # Output Overall statistics across three folders
    print("\n" + "="*60)
    print("===== Overall statistics across three folders =====")
    print("="*60)
    arr_all_psnr = np.array(all_psnr)
    total_n = len(arr_all_psnr)
    if total_n == 0:
        print("No valid samples in the combined set!")
    else:
        total_mean = arr_all_psnr.mean()
        total_std = arr_all_psnr.std()
        print(f"Total combined samples: {total_n}")
        print(f"Overall PSNR: {total_mean:.5f} ± {total_std:.5f}")

    return results


# -------------------------- Configure paths --------------------------
origin_folder = r"..."
attack_folders = [
    ("1", r"..."),
    ("2", r"..."),
    ("3", r"...")
]

psnr_all_results = calculate_psnr_multi(origin_folder, attack_folders)