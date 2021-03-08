import numpy as np
import nibabel
from monai.transforms import LoadNifti
from nibabel.affines import apply_affine

affine = np.array(
    [
        [9.939818382263184e-01, -6.656585633754730e-02, 3.764418512582779e-02, -2.641143798828125e-01],
        [6.501126289367676e-02, 9.986859560012817e-01, 3.586662933230400e-02, 5.816650390625000e-01],
        [-3.823856636881828e-02, -3.433264791965485e-02, 1.001067638397217e00, 2.456728363037109e01],
        [0.000000000000000e00, 0.000000000000000e00, 0.000000000000000e00, 1.000000000000000e00],
    ]
)

if __name__ == "__main__":
    loadnifti = LoadNifti()
    img, compatible_meta = loadnifti(
        "/home/jq/Desktop/rUnet/data/ADNI/M06_stripping/ADNI_002_S_0938_MR_MPR__GradWarp_Br_20070713122551713_S29620_I60041_mask.nii.gz"
    )
    img_after = apply_affine(affine, img)
    nibabel.save(img_after, "tmp.nii.gz")
