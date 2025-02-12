import gradio as gr
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io

def preprocess_image(image, target_size=(512, 512)):
    """Preprocess the input image to a consistent size."""
    if isinstance(image, np.ndarray):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.array(image)
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def create_color_tile(color, tile_size):
    """Create a solid color tile of specified size."""
    return np.full((tile_size, tile_size, 3), color, dtype=np.uint8)

def create_pattern_tile(color, tile_size, pattern='solid'):
    """Create a tile with different patterns."""
    if pattern == 'solid':
        return create_color_tile(color, tile_size)
    
    elif pattern == 'checker':
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        checker_size = tile_size // 4
        for i in range(0, tile_size, checker_size):
            for j in range(0, tile_size, checker_size):
                if (i // checker_size + j // checker_size) % 2 == 0:
                    tile[i:i+checker_size, j:j+checker_size] = color
        return tile
    
    elif pattern == 'circles':
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        center = tile_size // 2
        radius = tile_size // 3
        cv2.circle(tile, (center, center), radius, color, -1)
        return tile
    
    elif pattern == 'diagonal':
        tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        thickness = max(1, tile_size // 8)
        for i in range(-tile_size, tile_size, tile_size // 2):
            cv2.line(tile, (i, 0), (i + tile_size, tile_size), color, thickness)
        return tile

def get_average_color(cell):
    """Calculate the average color of a grid cell."""
    return np.mean(cell, axis=(0, 1)).astype(np.uint8)

def create_segmented_view(image, grid_size):
    """Create a visualization of the grid segmentation."""
    height, width = image.shape[:2]
    tile_size = height // grid_size
    preview = image.copy()
    
    # Draw vertical lines
    for x in range(0, width, tile_size):
        cv2.line(preview, (x, 0), (x, height), (255, 255, 255), 1)
    
    # Draw horizontal lines
    for y in range(0, height, tile_size):
        cv2.line(preview, (0, y), (width, y), (255, 255, 255), 1)
    
    return preview

def create_mosaic(image, grid_size=16, pattern='solid'):
    """Convert an image to a mosaic with specified pattern."""
    height, width = image.shape[:2]
    tile_size = height // grid_size
    
    # Create output image
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Process each grid cell
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # Get current cell
            cell = image[y:y+tile_size, x:x+tile_size]
            if cell.size == 0:
                continue
            
            # Calculate average color for the cell
            avg_color = get_average_color(cell)
            
            # Create and place the patterned tile
            tile = create_pattern_tile(avg_color, tile_size, pattern)
            
            # Handle edge cases where the cell might be smaller than tile_size
            h, w = cell.shape[:2]
            mosaic[y:y+h, x:x+w] = tile[:h, :w]
    
    return mosaic

def calculate_similarity(original, mosaic):
    """Calculate similarity between original and mosaic images."""
    # Convert images to grayscale for SSIM
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_RGB2GRAY)
    
    # Calculate SSIM
    ssim_score, _ = ssim(original_gray, mosaic_gray, full=True)
    
    # Calculate MSE
    mse = np.mean((original - mosaic) ** 2)
    
    return {
        "SSIM Score": f"{ssim_score:.4f}",
        "MSE": f"{mse:.2f}"
    }

def process_image(input_image, grid_size, pattern_type):
    """Main processing function for the Gradio interface."""
    try:
        # Preprocess the image
        processed_image = preprocess_image(input_image)
        
        # Create segmented view
        segmented = create_segmented_view(processed_image, grid_size)
        
        # Create mosaic
        mosaic = create_mosaic(processed_image, grid_size=grid_size, pattern=pattern_type)
        
        # Calculate similarity metrics
        metrics = calculate_similarity(processed_image, mosaic)
        
        # Create output string for metrics
        metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
        
        return processed_image, segmented, mosaic, metrics_str
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None, None, None, f"Error: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=8, maximum=64, value=16, step=8, label="Grid Size"),
        gr.Radio(
            choices=['solid', 'checker'],
            value='solid',
            label="Tile Pattern"
        )
    ],
    outputs=[
        gr.Image(type="pil", label="Original Image"),
        gr.Image(type="pil", label="Grid Preview"),
        gr.Image(type="pil", label="Mosaic Output"),
        gr.Textbox(label="Similarity Metrics")
    ],
    title="Interactive Image Mosaic Generator",
    description="""Upload an image and adjust the grid size to create a mosaic effect. 
                Choose different tile patterns for unique effects. The program will show:
                1. Your original image
                2. Grid segmentation preview
                3. Final mosaic result
                4. Similarity metrics comparing the original and mosaic images""",
    examples=[
        ["examples/sample1.jpg", 16, "solid"],
        ["examples/sample2.jpg", 32, "checker"]
    ],
    theme=gr.themes.Base()
)

# Launch the interface
if __name__ == "__main__":
    iface.launch(share=True)
