#!/usr/bin/env python3
"""
Detect faces in a single photo and assign to clusters.
Usage: python3 detect_photo.py <photo_path> <db_path> <photo_dir>
"""
import sys, os, json, sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from faces import detect_faces_in, assign_cluster, init_db


def run_detect(photo_path, db_path, photo_dir):
    thumb_dir = os.path.join(os.path.dirname(db_path), 'face_thumbs')
    os.makedirs(thumb_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn)

    rel_path = os.path.relpath(photo_path, photo_dir).replace('\\', '/')
    face_list = detect_faces_in(photo_path, thumb_dir)
    results = []

    for face in face_list:
        cid = assign_cluster(conn, face['histogram'])
        row = conn.execute(
            'SELECT name, sample_thumb FROM face_clusters WHERE id=?', (cid,)
        ).fetchone()
        cluster_name = row[0] if row else None
        if row and not row[1]:
            conn.execute(
                'UPDATE face_clusters SET sample_thumb=? WHERE id=?',
                (face['thumb_path'], cid)
            )
        already = conn.execute(
            'SELECT id FROM photo_faces WHERE photo_path=? AND x=? AND y=?',
            (rel_path, face['x'], face['y'])
        ).fetchone()
        if not already:
            conn.execute(
                'INSERT INTO photo_faces (photo_path,cluster_id,histogram,x,y,w,h,thumb_path) '
                'VALUES (?,?,?,?,?,?,?,?)',
                (rel_path, cid, json.dumps(face['histogram']),
                 face['x'], face['y'], face['w'], face['h'], face['thumb_path'])
            )
        results.append({
            'x': face['x'], 'y': face['y'], 'w': face['w'], 'h': face['h'],
            'cluster_id': cid,
            'cluster_name': cluster_name,
            'thumb_filename': os.path.basename(face['thumb_path']),
        })

    conn.commit()
    conn.close()
    print(json.dumps({'faces': results}))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(json.dumps({'error': 'Usage: detect_photo.py <photo_path> <db_path> <photo_dir>'}))
        sys.exit(1)
    run_detect(sys.argv[1], sys.argv[2], sys.argv[3])
