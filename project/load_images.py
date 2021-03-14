from monai.transforms import LoadNifti, apply_transform
from utils.const import DIFFUSION_INPUT, DIFFUSION_LABEL, DATA_ROOT, ADNI_LIST
from nibabel.freesurfer.mghformat import MGHImage

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.transforms import get_diffusion_preprocess, get_diffusion_label_preprocess
from sklearn.model_selection import train_test_split
from utils.cropping import crop_to_nonzero
from utils.visualize_dti import log_all_info

# plt.rcParams["figure.figsize"] = (8.0, 8.0)
# NUM = 1

if __name__ == "__main__":
    # X_img, compatible_meta = loadnifti("/home/jq/Desktop/rUnet/data/ADNI/v1to2.mgz")
    # print(f"X shape: {X_img.shape}")

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

    # y_list_all = set(list(ADNI_LIST[0].glob("**/*.nii.gz")))
    # y_list_mask = set(list(ADNI_LIST[0].glob("**/*_mask.nii.gz")))
    # y = sorted(list(y_list_all - y_list_mask))

    # if NUM == 1:
    #     X_M12 = ADNI_LIST[1]
    #     X = sorted(list(X_M12.glob("**/*.nii.mgz")))
    # elif NUM == 2:
    #     X_M12, X_M06 = ADNI_LIST[1], ADNI_LIST[2]
    #     X_M12_files, X_M06_files = sorted(list(X_M12.glob("**/*.nii.mgz"))), sorted(list(X_M06.glob("**/*.nii.mgz")))
    #     X = []
    #     for m12, m06 in zip(X_M12_files, X_M06_files):
    #         X.append([m12, m06])
    # elif NUM == 3:
    #     X_M12, X_M06, X_SC = ADNI_LIST[1], ADNI_LIST[2], ADNI_LIST[3]
    #     X_M12_files, X_M06_files, X_SC_files = (
    #         sorted(list(X_M12.glob("**/*.nii.mgz"))),
    #         sorted(list(X_M06.glob("**/*.nii.mgz"))),
    #         sorted(list(X_SC.glob("**/*.nii.mgz"))),
    #     )
    #     X = []
    #     for m12, m06, sc in zip(X_M12_files, X_M06_files, X_SC_files):
    #         X.append([m12, m06, sc])

    # max_x, max_y, max_z = 0, 0, 0
    # loadnifti = LoadNifti()
    # for y_path in y:
    # for x in X[:5]:
    #     m12 = MGHImage.load(x).get_fdata()

    #     # print(f"target shape: {img.shape}")
    #     for p in range(80, 90, 1):
    #         print(f"m12 {p}%: {np.percentile(m12, p)}")

    # fig, axes = plt.subplots(nrows=5, ncols=1)
    # for i, y_path in enumerate(y[:5]):
    #     img, compatible_meta = loadnifti(y_path)
    #     mask = img != 0.0
    #     print(f"min : {np.min(img[mask])}")
    # X_transform = get_train_img_transforms()
    #     t1 = apply_transform(X_transform, img)
    #     t1 = t1[t1 != 0.0]
    #     sns.distplot(t1, kde=True, ax=axes[i])

    # fig.savefig("dist.png")

    # for subject in X:
    #     for scan in subject:
    #         img = MGHImage.load(scan).get_fdata()
    #         img[img < np.percentile(img, 85)] = 0.0
    #         img = crop_to_nonzero(img)
    #         print(f"img shape: {img.shape}")
    #         max_x, max_y, max_z = max(max_x, img.shape[0]), max(max_y, img.shape[1]), max(max_z, img.shape[2])

    # m06 = MGHImage.load(subject[1]).get_fdata()
    # img = crop_to_nonzero(m06)
    # print(f"img path: {subject[1]}, img shape: {img.shape}")
    # max_x, max_y, max_z = max(max_x, img.shape[0]), max(max_y, img.shape[1]), max(max_z, img.shape[2])

    # sc = MGHImage.load(subject[2]).get_fdata()
    # img = crop_to_nonzero(sc)
    # print(f"img path: {subject[2]}, img shape: {img.shape}")
    # max_x, max_y, max_z = max(max_x, img.shape[0]), max(max_y, img.shape[1]), max(max_z, img.shape[2])

    # print(f"max x: {max_x}")
    # print(f"max y: {max_y}")
    # print(f"max z: {max_z}")

    # X = sorted(list(DIFFUSION_INPUT.glob("**/*.nii")))
    # y = sorted(list(DIFFUSION_LABEL.glob("**/*.nii")))

    # loadnifti = LoadNifti()
    # X_transform = get_diffusion_preprocess()
    # Y_transform = get_diffusion_label_preprocess()
    # for idx, (x_path, y_path) in enumerate(zip(X, y)):
    #     x_img, compatible_meta = loadnifti(x_path)
    #     print(f"before x_img shape:{x_img.shape}")
    #     x_img = apply_transform(X_transform, x_img).numpy()
    #     print(f"after x_img shape: {x_img.shape}")

    #     y_img, compatible_meta = loadnifti(y_path)
    #     y_img = apply_transform(Y_transform, y_img).numpy()
    #     print(f"processed No. {idx} image.")
    #     np.savez(f"{idx}.npz", X=x_img, y=y_img)

    tmp = np.load("/home/jq/Desktop/rUnet/data/Diffusion/0.npz")
    y_img = tmp["y"]

    log_all_info(
        target=y_img,
        preb=y_img,
        loss=0.0,
        batch_idx=1,
        state="val",
    )
