from PIL import Image

# Path to your 16-bit PNG file
file_path = r"C:\Users\auroras\.venvMISS2\MISS2\Captured_PNG\2024\03\19\MISS2-20240319-220230.png"

# Open the image
image = Image.open(file_path)

# Print basic metadata
print(f"Format: {image.format}")
print(f"Mode: {image.mode}")  # 'I;16' for 16-bit grayscale
print(f"Size: {image.size}")
print(f"Palette: {image.palette}")

# Accessing and printing more specific metadata, like embedded text information
metadata = image.info
for key, value in metadata.items():
    print(f"{key}: {value}")

# Don't forget to close the image file
image.close()