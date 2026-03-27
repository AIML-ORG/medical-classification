import cv2
import os
import albumentations as A
from tqdm import tqdm

# 1. Define your "Medical Document" Pipeline (from our previous discussion)
# Updated for albumentations 2.0 compatibility
MEDICAL_DOC_AUGMENT = A.Compose([
    # 0. FLIPS & RIGHT-ANGLE ROTATION (90/180/270/360°; 360° = identity, sampled as 0°)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=1.0),

    # 1. GEOMETRY — skew / keystone = Perspective (trapezoid from camera angle).
    #    ElasticTransform = wavy paper, not trapezoid skew.
    A.OneOf([
        A.Perspective(
            scale=(0.06, 0.14),
            fit_output=True,
            border_mode=0,
            fill=(255, 255, 255),
            p=1,
        ),
        A.ElasticTransform(alpha=12, sigma=40, approximate=True, p=1),
    ], p=0.85),

    # 2. PRINTING & SCANNING ARTIFACTS (The "Source" Simulation)
    # Using available transforms in albumentations 2.0
    A.OneOf([
        # Downscale: Simulates low-resolution printing/scanning
        A.Downscale(scale_range=(0.5, 0.9), p=1),
        # ImageCompression: Simulates JPEG compression artifacts
        A.ImageCompression(quality_range=(60, 90), p=1),
        # GaussNoise: Adds noise typical of scanned documents
        A.GaussNoise(std_range=(0.01, 0.03), p=1),
    ], p=0.3),

    # 3. LIGHTING & LENS EFFECTS
    # PlasmaShadow: Creates uneven, organic lighting across the page
    A.PlasmaShadow(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    # RandomShadow: Simulates shadows from scanning/photography
    A.RandomShadow(num_shadows_limit=(1, 3), p=0.2),

    # 4. TEXT LEGIBILITY (Blur vs Sharpen)
    A.OneOf([
        A.Sharpen(alpha=(0.2, 0.5), p=1),
        A.GaussianBlur(blur_limit=(3, 5), p=1),
    ], p=0.3),
    A.MotionBlur(blur_limit=(3, 5), allow_shifted=True, p=0.45),

    # 5. NOISE & COMPRESSION
    A.OneOf([
        A.ImageCompression(quality_range=(60, 90), p=1),
        A.GaussNoise(std_range=(0.009, 0.015), p=1),
        A.ISONoise(color_shift=(0.01, 0.03), p=1),
    ], p=0.3),

    # 6. FINAL (keep uint8 0-255 for cv2.imwrite; apply A.Normalize() in the training dataloader only)
    A.ToGray(p=1.0),
])

def augment_and_save(input_folder, output_folder, augmentations_per_image=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through every image in your original data
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    for filename in tqdm(image_files):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        
        # Save the original first (optional)
        cv2.imwrite(os.path.join(output_folder, f"orig_{filename}"), image)
        
        # Generate N augmented versions
        for i in range(augmentations_per_image):
            augmented = MEDICAL_DOC_AUGMENT(image=image)["image"]
            
            # Save with a new name
            new_name = f"aug_{i}_{filename}"
            cv2.imwrite(os.path.join(output_folder, new_name), augmented)

if __name__ == "__main__":
    augment_and_save("v3/datasets/tf-test_prescription-dpflz_v2/test/images", "v3/datasets/tf-test_prescription-dpflz_v2/test/images_augmented", augmentations_per_image=3)