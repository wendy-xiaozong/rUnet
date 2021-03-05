from monai.transforms import LoadNifti, apply_transform
from utils.const import DIFFUSION_INPUT, DIFFUSION_LABEL
from utils.transforms import get_diffusion_preprocess
from nibabel.freesurfer.mghformat import MGHImage

import numpy as np

if __name__ == "__main__":
    # X = sorted(list(DIFFUSION_INPUT.glob("**/*.nii")))
    # y = sorted(list(DIFFUSION_LABEL.glob("**/*.nii")))

    # loadnifti = LoadNifti()
    # X_img, compatible_meta = loadnifti("/home/jq/Desktop/rUnet/data/ADNI/v1to2.mgz")
    # print(f"X shape: {X_img.shape}")
    # preprocess = get_diffusion_preprocess()

    # X_img = apply_transform(preprocess, X_img)

    # print(f"X shape after transform: {X_img.shape}")

    output = MGHImage.load("/home/jq/Desktop/rUnet/data/ADNI/v1to2.mgz").get_fdata()

    loadnifti = LoadNifti()
    mov, compatible_meta = loadnifti(
        "/home/jq/Desktop/rUnet/data/ADNI/SC/002_S_1018/MPR__GradWarp/2006-11-29_10_00_05.0/S23128/ADNI_002_S_1018_MR_MPR__GradWarp_Br_20070217030756067_S23128_I40819.nii"
    )
    dst, compatible_meta = loadnifti(
        "/home/jq/Desktop/rUnet/data/ADNI/M06/002_S_1018/MPR__GradWarp/2007-07-16_06_56_24.0/S35097/ADNI_002_S_1018_MR_MPR__GradWarp_Br_20070913144018582_S35097_I72999.nii"
    )

    print(f"output mean:{np.mean(output)}, 50%: {np.percentile(output, 50)}, 40%: {np.percentile(output, 40)}")
    print(f"mov mean:{np.mean(mov)}, 50%: {np.percentile(mov, 50)}, 40%: {np.percentile(mov, 40)}")
    print(f"dst mean:{np.mean(dst)}, 50%: {np.percentile(dst, 50)}, 40%: {np.percentile(dst, 40)}")
    print("Great!")
