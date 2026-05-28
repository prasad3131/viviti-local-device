#!/usr/bin/env python3
"""Generate a JPEG thumbnail. Usage: thumb.py <src> <dst> <size>"""
import sys
from PIL import Image

src, dst, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
img = Image.open(src)
img.thumbnail((size, size))
img.convert('RGB').save(dst, 'JPEG', quality=82)
