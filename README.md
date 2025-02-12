# mosaic-image-generator

📸 Interactive Image Mosaic Generator
This project is an Interactive Image Mosaic Generator built using Gradio for the web interface. It allows users to upload images, adjust grid size, and apply different mosaic patterns to create artistic effects. It also calculates similarity metrics (SSIM and MSE) to compare the original image with the mosaic output.

🎯 Features
Image Upload: Supports image upload in various formats (JPG, PNG, etc.).
Grid Customization: Choose grid sizes between 8 and 64 to customize the mosaic.
Pattern Selection: Choose from different tile patterns:
solid: Solid color tiles based on the average color of each grid cell.
checker: Checkerboard pattern for a unique effect.
Similarity Metrics:
SSIM (Structural Similarity Index): Measures structural similarity between the original image and the mosaic.
MSE (Mean Squared Error): Measures pixel-level difference.

🛠️ Installation
Prerequisites
Make sure you have Python 3.7+ installed and the following libraries:

gradio
numpy
opencv-python
scikit-image

🚀 Usage
Clone this repository:
git clone https://github.com/yourusername/image-mosaic-generator.git
cd image-mosaic-generator
Run the file 
Access the web interface on your browser (http://localhost:7860).

📚 Code Explanation
Key Functions
preprocess_image: Resizes and converts images to a consistent format.
create_pattern_tile: Creates tiles with different patterns (solid, checker, circles, diagonal).
create_mosaic: Generates the final mosaic by segmenting the image into tiles and replacing each tile with a pattern based on its average color.
calculate_similarity: Computes SSIM and MSE to compare the original image with the mosaic.

