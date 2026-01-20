from PIL import Image
import os

    
try:
    RESAMPLING_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLING_FILTER = Image.LANCZOS

def resize_image_with_padding(image_path, desired_width, desired_height, output_path=None):
    """
    Shrinks the image by 1.5x, centers it on a canvas of desired 
    dimensions, and pads remaining space with black.

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

    new_width = int(original_width / 1.25)
    new_height = int(original_height / 1.25)

    resized_image = original_image.resize((new_width, new_height), RESAMPLING_FILTER)

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

