import imageio.v2 as imageio
from pathlib import Path

# Folder containing the PNG files
image_dir = Path(".")

# Output video filename
output_file = "movie.mp4"

# Frames per second (0.1 s per frame)
fps = 10

# Collect images in numerical order
image_files = sorted(image_dir.glob("v*.png"))

# Write video
with imageio.get_writer(output_file, fps=fps) as writer:
    for image_file in image_files:
        image = imageio.imread(image_file)
        writer.append_data(image)

print(f"Created {output_file} with {len(image_files)} frames.")
