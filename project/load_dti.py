from monai.transforms import LoadNifti, apply_transform
from utils.const import DIFFUSION_INPUT, DIFFUSION_LABEL
from utils.transforms import get_diffusion_preprocess


if __name__ == "__main__":
    X = sorted(list(DIFFUSION_INPUT.glob("**/*.nii")))
    # y = sorted(list(DIFFUSION_LABEL.glob("**/*.nii")))

    loadnifti = LoadNifti()
    X_img, compatible_meta = loadnifti(X[0])
    print(f"X shape: {X_img.shape}")
    preprocess = get_diffusion_preprocess()

    X_img = apply_transform(preprocess, X_img)

    print(f"X shape after transform: {X_img.shape}")
