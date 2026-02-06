import os
import shutil
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import pickle
import pandas as pd
import random

# -------------------------
# Set base directory
# -------------------------
base_dir = "D:/Anum/MA-SAM/Processed_data_nii"

# -------------------------
# Step 1: Organize Data
# -------------------------
def organize_data():
    save_pth = os.path.join(base_dir, 'Dataset911_prostateD')
    os.makedirs(save_pth+'/imagesTr', exist_ok=True)
    os.makedirs(save_pth+'/labelsTr', exist_ok=True)

    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    for folder in subfolders:
        folder_path = os.path.join(base_dir, folder)
        for f in os.listdir(folder_path):
            if f.endswith('.nii.gz') and 'segmentation' not in f.lower():
                patient_ID = f"{folder}_{f.split('.')[0]}"
                img_path = os.path.join(folder_path, f)

                # Find mask (handle both '_segmentation' and '_Segmentation')
                mask_name = f.replace('.nii.gz', '_segmentation.nii.gz')
                mask_path = os.path.join(folder_path, mask_name)
                if not os.path.exists(mask_path):
                    mask_name = f.replace('.nii.gz', '_Segmentation.nii.gz')
                    mask_path = os.path.join(folder_path, mask_name)

                if not os.path.exists(mask_path):
                    print(f"Mask not found for {img_path}, skipping...")
                    continue

                # Load mask and make binary
                mask_obj = nib.load(mask_path)
                mask_arr = mask_obj.get_fdata()
                mask_arr[mask_arr > 1] = 1
                new_mask_obj = nib.Nifti1Image(mask_arr, mask_obj.affine, header=mask_obj.header)

                # Save organized images & masks
                nib.save(new_mask_obj, os.path.join(save_pth, 'labelsTr', patient_ID + '.nii.gz'))
                shutil.copy(img_path, os.path.join(save_pth, 'imagesTr', patient_ID + '_0000.nii.gz'))

    print("Data organization complete!")

# -------------------------
# Step 2: Convert 3D → 2D slices
# -------------------------
def get_3D_2D_all_5slice():
    save_pth = os.path.join(base_dir, 'prostateD/2D_all_5slice')
    data_pth = os.path.join(base_dir, 'Dataset911_prostateD')
    os.makedirs(save_pth, exist_ok=True)

    data_fd_list = [f for f in os.listdir(os.path.join(data_pth, 'imagesTr')) if f.endswith('.nii.gz')]
    data_fd_list.sort()

    for data_fd in data_fd_list:
        case_id = data_fd.split('_')[0] + "_" + data_fd.split('_')[1]

        case_path = os.path.join(save_pth, case_id)
        os.makedirs(case_path, exist_ok=True)
        os.makedirs(os.path.join(case_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(case_path, 'masks'), exist_ok=True)

        # Load 3D image & mask
        img_obj = nib.load(os.path.join(data_pth, 'imagesTr', data_fd))
        img_arr = np.float32(img_obj.get_fdata())
        mask_obj = nib.load(os.path.join(data_pth, 'labelsTr', data_fd.replace('_0000.nii.gz', '.nii.gz')))
        mask_arr = np.float32(mask_obj.get_fdata())

        # Intensity normalization
        high = np.quantile(img_arr, 0.99)
        low = np.min(img_arr)
        img_arr = np.clip(img_arr, low, high)
        img_arr = (img_arr - low) / (high - low + 1e-8)

        # Resize to 512x512 if needed
        h, w = img_arr.shape[:2]
        if h != 512 or w != 512:
            img_arr = zoom(img_arr, (512/h, 512/w, 1.0), order=3)
            mask_arr = zoom(mask_arr, (512/h, 512/w, 1.0), order=0)

        # Stack slices: 5-channel (±2 slices)
        img_arr = np.concatenate([img_arr[:, :, 0:1]]*2 + [img_arr] + [img_arr[:, :, -1:]]*2, axis=-1)
        mask_arr = np.concatenate([mask_arr[:, :, 0:1]]*2 + [mask_arr] + [mask_arr[:, :, -1:]]*2, axis=-1)

        for slice_idx in range(2, img_arr.shape[2]-2):
            slice_arr = img_arr[:, :, slice_idx-2:slice_idx+3]
            slice_arr = np.flip(np.rot90(slice_arr, k=1, axes=(0, 1)), axis=1)

            mask_slice = mask_arr[:, :, slice_idx-2:slice_idx+3]
            mask_slice = np.flip(np.rot90(mask_slice, k=1, axes=(0, 1)), axis=1)

            # Save as .pkl
            with open(os.path.join(case_path, 'images', f'2Dimage_{slice_idx-2:04d}.pkl'), 'wb') as f:
                pickle.dump(slice_arr, f)
            with open(os.path.join(case_path, 'masks', f'2Dmask_{slice_idx-2:04d}.pkl'), 'wb') as f:
                pickle.dump(mask_slice, f)

    print("3D → 2D slice conversion complete!")

# -------------------------
# Step 3: Generate CSVs
# -------------------------
def get_csv():
    save_pth = os.path.join(base_dir, 'prostateD/2D_all_5slice')
    all_csv = os.path.join(save_pth, 'all.csv')
    train_csv = os.path.join(save_pth, 'training.csv')
    val_csv = os.path.join(save_pth, 'validation.csv')
    test_csv = os.path.join(save_pth, 'test.csv')

    case_list = [f for f in os.listdir(save_pth) if os.path.isdir(os.path.join(save_pth, f))]
    case_list.sort()

    # Define test cases (you can change these)
    test_cases = case_list[:3]  # first 3 cases as test
    train_cases = list(set(case_list) - set(test_cases))
    val_cases = test_cases[:1]   # first test case as validation

    def make_df(cases):
        paths = []
        for case in cases:
            slice_files = os.listdir(os.path.join(save_pth, case, 'images'))
            slice_files.sort()
            for sf in slice_files:
                img_path = f"{case}/images/{sf}"
                mask_path = f"{case}/masks/{sf.replace('2Dimage', '2Dmask')}"
                paths.append((img_path, mask_path))
        df = pd.DataFrame(paths, columns=['image_pth', 'mask_pth'])
        return df

    # Save CSVs
    make_df(case_list).to_csv(all_csv, index=False)
    make_df(train_cases).to_csv(train_csv, index=False)
    make_df(val_cases).to_csv(val_csv, index=False)
    make_df(test_cases).to_csv(test_csv, index=False)

    print("CSV files generated!")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    organize_data()
    get_3D_2D_all_5slice()
    get_csv()
    print("Preprocessing complete!")