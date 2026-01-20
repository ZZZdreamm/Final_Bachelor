from PIL import Image
import os

    
try:
    RESAMPLING_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLING_FILTER = Image.LANCZOS

def resize_image_with_padding(image_path, desired_width, desired_height, output_path=None):
    """
    Scales the image to fit within the desired dimensions while maintaining 
    aspect ratio, and then pads any remaining space with black to meet the 
    exact desired dimensions.

    Args:
        image_path (str): The path to the image file.
        desired_width (int): The target width.
        desired_height (int): The target height.
        output_path (str, optional): Path to save the new image.
    
    Returns:
        PIL.Image object or None: The processed image object, or None on failure.
    """
    
    try:
        original_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

    original_width, original_height = original_image.size

    width_ratio = desired_width / original_width
    height_ratio = desired_height / original_height
    
    scale_factor = min(width_ratio, height_ratio)

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    if scale_factor != 1.0:
        resized_image = original_image.resize((new_width, new_height), RESAMPLING_FILTER)
    else:
        resized_image = original_image

    new_image = Image.new('RGB', (desired_width, desired_height), (0, 0, 0))

    offset_x = (desired_width - new_width) // 2
    offset_y = (desired_height - new_height) // 2
    
    new_image.paste(resized_image, (offset_x, offset_y))

    if output_path:
        try:
            ext = os.path.splitext(output_path)[1].lower()
            save_format = resized_image.format if resized_image.format and ext in ['.jpg', '.jpeg'] else 'PNG'
            new_image.save(output_path, format=save_format)
        except Exception as e:
            return None
            
    return new_image

def process_images_in_root_folder(root_dir, output_root_dir, target_width, target_height, image_extensions=('.jpg', '.jpeg', '.png')):
    """
    Traverses a root directory, finds images in subfolders, applies padding,
    and saves the results in a mirrored structure in the output directory.

    Args:
        root_dir (str): The folder to start searching for images.
        output_root_dir (str): The folder where padded images will be saved.
        target_width (int): The target width for padding.
        target_height (int): The target height for padding.
        image_extensions (tuple): Extensions to consider as images.
    """
    processed_count = 0
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        
        relative_path = os.path.relpath(dirpath, root_dir)
        
        output_dir = os.path.join(output_root_dir, relative_path)
        
        os.makedirs(output_dir, exist_ok=True)

        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                input_path = os.path.join(dirpath, filename)
                output_path = os.path.join(output_dir, filename)

                result = resize_image_with_padding(
                    image_path=input_path, 
                    desired_width=target_width, 
                    desired_height=target_height, 
                    output_path=output_path
                )

                if result:
                    processed_count += 1
    

IMAGES_FOLDER = "/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/test_data_przetworzone2"
OUTPUT_FOLDER = "/mnt/c/Users/KMult/Desktop/Praca_inzynierska/models/test_data_przetworzone2_rozmiar"
RENDER_MODELS_SIZE = (1080, 2340)
    
process_images_in_root_folder(IMAGES_FOLDER, OUTPUT_FOLDER, RENDER_MODELS_SIZE[0], RENDER_MODELS_SIZE[1])
    