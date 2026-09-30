import os
from skimage.metrics import structural_similarity as ssim
from skimage import io
import numpy as np


# Calculate SSIM with channel_axis support for RGB images
def calculate_ssim(img1, img2):
    # img1,img2: (H,W,3) RGB 0‑255
    return ssim(img1, img2, win_size=3, channel_axis=-1)


# Calculate SSIM for adversarial folders against original images
def calculate_ssim_multi(origin_folder, attack_folders):
    results = {}
    # Read all original PNG filenames once
    if not os.path.exists(origin_folder):
        print(f"Original-image folderdoes not exist: {origin_folder}")
        return results
    ori_file_list = [f for f in os.listdir(origin_folder) if f.lower().endswith(".png")]
    print(f"\nOriginal-image folderPNG file count: {len(ori_file_list)}")

    for name, adv_folder in attack_folders:
        results[name] = {
            "ssim_list": [],
            "max_ssim": 0.0,
            "min_ssim": 1.0
        }

        if not os.path.exists(adv_folder):
            print(f"\n[{name}] folder {adv_folder} does not exist; skipping!")
            continue
        print(f"\n===== Processing {name} =====")
        res = results[name]

        for filename in ori_file_list:
            file_ori = os.path.join(origin_folder, filename)
            file_adv = os.path.join(adv_folder, filename)
            # Silently skip missing adversarial samples
            if not os.path.exists(file_adv):
                continue
            try:
                img_ori = io.imread(file_ori)
                img_adv = io.imread(file_adv)
                # Skip images with mismatched dimensions
                if img_ori.shape[:2] != img_adv.shape[:2]:
                    print(f"{filename}：original and adversarial image dimensions differ; skipping")
                    continue

                ssim_val = calculate_ssim(img_ori, img_adv)
                res["ssim_list"].append(ssim_val)
                res["max_ssim"] = max(res["max_ssim"], ssim_val)
                res["min_ssim"] = min(res["min_ssim"], ssim_val)
                print(f"{filename} SSIM: {ssim_val:.5f}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # ========== Summary ==========
    print("\n" + "="*60)
    print("SSIM summary (compared with original images)")
    print("="*60)

    all_ssim = []  # Collect SSIM values from all three folders for overall statistics

    for name, _ in attack_folders:
        res = results[name]
        ssim_arr = np.array(res["ssim_list"])
        n = len(ssim_arr)
        all_ssim.extend(res["ssim_list"])

        if n == 0:
            print(f"\n[{name}]No valid matched samples")
            continue
        mean_ssim = ssim_arr.mean()
        std_ssim = ssim_arr.std()
        print(f"\n[{name}]Valid samples: {n}")
        print(f"SSIM: {mean_ssim:.5f} ± {std_ssim:.5f}, max={res['max_ssim']:.5f}, min={res['min_ssim']:.5f}")

    # Output overall statistics across three folders
    print("\n" + "="*60)
    print("===== Overall statistics across three folders =====")
    print("="*60)
    arr_all_ssim = np.array(all_ssim)
    total_n = len(arr_all_ssim)
    if total_n == 0:
        print("No valid samples in the combined set!")
    else:
        total_mean = arr_all_ssim.mean()
        total_std = arr_all_ssim.std()
        print(f"Total combined samples: {total_n}")
        print(f"Overall SSIM: {total_mean:.5f}±{total_std:.5f}")

    return results


# -------------------------- Configure paths --------------------------
folder_origin = r"..."
attack_folders = [
    ("1", r"..."),
    ("2", r"..."),
    ("3", r"...")
]

ssim_all_results = calculate_ssim_multi(folder_origin, attack_folders)