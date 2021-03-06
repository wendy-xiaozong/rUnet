from monai.transforms import LoadNifti, apply_transform
from utils.const import DIFFUSION_INPUT, DIFFUSION_LABEL, DATA_ROOT, ADNI_LIST
from utils.transforms import get_diffusion_preprocess
from nibabel.freesurfer.mghformat import MGHImage
from utils.transforms import get_train_img_transforms, get_val_img_transforms, get_label_transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

plt.rcParams["figure.figsize"] = (8.0, 8.0)  # 设置figure_size尺寸
NUM = 3

if __name__ == "__main__":
    # X = sorted(list(DIFFUSION_INPUT.glob("**/*.nii")))
    # y = sorted(list(DIFFUSION_LABEL.glob("**/*.nii")))

    # loadnifti = LoadNifti()
    # X_img, compatible_meta = loadnifti("/home/jq/Desktop/rUnet/data/ADNI/v1to2.mgz")
    # print(f"X shape: {X_img.shape}")
    # preprocess = get_diffusion_preprocess()

    # X_img = apply_transform(preprocess, X_img)

    # print(f"X shape after transform: {X_img.shape}")

    # output = MGHImage.load("/home/jq/Desktop/rUnet/data/ADNI/v1to2.mgz").get_fdata()

    # loadnifti = LoadNifti()
    # mov, compatible_meta = loadnifti(
    #     "/home/jq/Desktop/rUnet/data/ADNI/SC/002_S_1018/MPR__GradWarp/2006-11-29_10_00_05.0/S23128/ADNI_002_S_1018_MR_MPR__GradWarp_Br_20070217030756067_S23128_I40819.nii"
    # )
    # dst, compatible_meta = loadnifti(
    #     "/home/jq/Desktop/rUnet/data/ADNI/M06/002_S_1018/MPR__GradWarp/2007-07-16_06_56_24.0/S35097/ADNI_002_S_1018_MR_MPR__GradWarp_Br_20070913144018582_S35097_I72999.nii"
    # )

    # print(f"output mean:{np.mean(output)}, 50%: {np.percentile(output, 50)}, 40%: {np.percentile(output, 40)}")
    # print(f"mov mean:{np.mean(mov)}, 50%: {np.percentile(mov, 50)}, 40%: {np.percentile(mov, 40)}")
    # print(f"dst mean:{np.mean(dst)}, 50%: {np.percentile(dst, 50)}, 40%: {np.percentile(dst, 40)}")
    # print("Great!")

    y_list_all = set(list(ADNI_LIST[0].glob("**/*.nii.gz")))
    y_list_mask = set(list(ADNI_LIST[0].glob("**/*_mask.nii.gz")))
    y = sorted(list(y_list_all - y_list_mask))

    if NUM == 1:
        X_M12 = ADNI_LIST[1]
        X = sorted(list(X_M12.glob("**/*.nii.mgz")))
    elif NUM == 2:
        X_M12, X_M06 = ADNI_LIST[1], ADNI_LIST[2]
        X_M12_files, X_M06_files = sorted(list(X_M12.glob("**/*.nii.mgz"))), sorted(list(X_M06.glob("**/*.nii.mgz")))
        X = []
        for m12, m06 in zip(X_M12_files, X_M06_files):
            X.append([m12, m06])
    elif NUM == 3:
        X_M12, X_M06, X_SC = ADNI_LIST[1], ADNI_LIST[2], ADNI_LIST[3]
        X_M12_files, X_M06_files, X_SC_files = (
            sorted(list(X_M12.glob("**/*.nii.mgz"))),
            sorted(list(X_M06.glob("**/*.nii.mgz"))),
            sorted(list(X_SC.glob("**/*.nii.mgz"))),
        )
        X = []
        for m12, m06, sc in zip(X_M12_files, X_M06_files, X_SC_files):
            X.append([m12, m06, sc])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)

    for X_cur in X_train[0]:
        print(X_cur)
