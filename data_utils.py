import os
from PIL import Image



# Define the path to the folder containing the images
input_folder = 'datasets/train'
output_folder = 'datasets/train_rescaled'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the new size (width, height) or scale factor
scale_factor = 0.15

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpg'):
        # Construct the full file path
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open the image file
        with Image.open(input_path) as img:
            # Print the original size
            print(f"Processing {filename}, original size: {img.size}")

            # Rescale the image by new size

            # Or rescale by scale factor
            new_dimensions = (int(img.width * scale_factor), int(img.height * scale_factor))
            img_rescaled = img.resize(new_dimensions)

            # Save the rescaled image
            img_rescaled.save(output_path)

            # Print the new size
            print(f"Rescaled {filename}, new size: {img_rescaled.size}")

print("Rescaling completed.")