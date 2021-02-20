from monai.transforms import LoadNifti, apply_transform
from utils.const import DIFFUSION_INPUT, DIFFUSION_LABEL


if __name__ == "__main__":
    X = sorted(list(DIFFUSION_INPUT.glob("**/*.nii")))

    loadnifti = LoadNifti()
    X_img, compatible_meta = loadnifti(X[0])
    print(f"X shape: {X_img.shape}")
