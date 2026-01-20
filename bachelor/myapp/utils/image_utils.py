import io
import unicodedata
import re

from PIL import Image

def get_image_stream(image_bytes: bytes):
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def sanitize_filename(filename_base: str) -> str:
    text = filename_base.lower()

    replacements = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ż': 'z', 'ź': 'z'
    }
    for pol, eng in replacements.items():
        text = text.replace(pol, eng)

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    text = re.sub(r'[^a-zA-Z0-9]+', '_', text).strip('_')

    return text