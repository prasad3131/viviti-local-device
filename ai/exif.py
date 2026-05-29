#!/usr/bin/env python3
"""
Read EXIF metadata from a photo.
Usage: python3 exif.py <photo_path>
"""
import sys, json, os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def dms_to_decimal(dms, ref):
    try:
        d, m, s = float(dms[0]), float(dms[1]), float(dms[2])
        val = d + m / 60 + s / 3600
        return round(-val if ref in ('S', 'W') else val, 6)
    except Exception:
        return None


def get_exif(photo_path):
    try:
        img = Image.open(photo_path)
        width, height = img.size
        file_size = os.path.getsize(photo_path)

        result = {
            'width': width, 'height': height, 'file_size': file_size,
            'date_taken': None, 'camera_make': None, 'camera_model': None,
            'focal_length': None, 'f_number': None, 'iso': None,
            'exposure_time': None, 'gps_lat': None, 'gps_lon': None,
        }

        exif_raw = img._getexif()
        if not exif_raw:
            print(json.dumps(result))
            return

        gps_raw = None
        for tag_id, value in exif_raw.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'DateTimeOriginal':
                result['date_taken'] = str(value)
            elif tag == 'Make':
                result['camera_make'] = str(value).strip('\x00').strip()
            elif tag == 'Model':
                result['camera_model'] = str(value).strip('\x00').strip()
            elif tag == 'FocalLength':
                try: result['focal_length'] = round(float(value), 1)
                except: pass
            elif tag == 'FNumber':
                try: result['f_number'] = round(float(value), 1)
                except: pass
            elif tag == 'ISOSpeedRatings':
                try: result['iso'] = int(value)
                except: pass
            elif tag == 'ExposureTime':
                try:
                    et = float(value)
                    result['exposure_time'] = f'1/{round(1/et)}s' if et < 1 else f'{round(et,1)}s'
                except: pass
            elif tag == 'GPSInfo':
                gps_raw = {GPSTAGS.get(k, k): v for k, v in value.items()}

        if gps_raw:
            lat = dms_to_decimal(gps_raw.get('GPSLatitude', []), gps_raw.get('GPSLatitudeRef', ''))
            lon = dms_to_decimal(gps_raw.get('GPSLongitude', []), gps_raw.get('GPSLongitudeRef', ''))
            if lat is not None and lon is not None:
                result['gps_lat'], result['gps_lon'] = lat, lon

        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({'error': str(e)}))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: exif.py <photo_path>'}))
        sys.exit(1)
    get_exif(sys.argv[1])
