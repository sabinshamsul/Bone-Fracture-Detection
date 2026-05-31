from src.preprocess import load_image, load_dataset
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# Test 1 - Load a single image

print("=" * 50)
print("Test 1: Loading a single image")
print("=" * 50)

# Point to one image from dataset
test_image_path = Path("data/FracAtlas/images/Fractured").glob("*.jpg")
first_image = next(test_image_path)

image = load_image(first_image)

if image is not None:
    print(f" Image loaded successfully")
    print(f" Shape: {image.shape}")
    print(f"Min pixel value: {image.min():.3f}")
    print(f"Max pixel value: {image.max():.3f}")
else:
    print("Image failed to load")

# Test 2 - Load full dataset

print()
print("=" * 50)
print("Test 2: Loading full dataset")
print("=" * 50)

X, y = load_dataset("data/FracAtlas")