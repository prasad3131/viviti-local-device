#!/usr/bin/env python3
"""
Detect faces in a video by sampling frames and reusing the photo face engine
(YuNet + SFace). Fully on-device. Returns the distinct people found.
Usage: python3 detect_video.py <video_path> <db_path> <photo_dir>
"""
import sys, os, json, sqlite3
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from faces import detect_faces_in, assign_cluster, init_db

SAMPLE_EVERY_SEC = 1.0   # process ~1 frame per second
MAX_FRAMES       = 40    # cap work on long videos


def run_detect_video(video_path, db_path, photo_dir):
    thumb_dir = os.path.join(os.path.dirname(db_path), 'face_thumbs')
    os.makedirs(thumb_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(json.dumps({'faces': [], 'error': 'cannot open video'}))
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    step  = max(1, int(round(fps * SAMPLE_EVERY_SEC)))

    people = {}            # cluster_id -> best { ..., score }
    frame_idx = 0
    sampled = 0

    while sampled < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            seed = f'{video_path}#{frame_idx}'
            faces = detect_faces_in(seed, thumb_dir, img=frame, lenient=True)
            used = set()                      # one cluster per face within a frame
            for face in faces:
                cid = assign_cluster(conn, face['embedding'], exclude=used)
                used.add(cid)
                row = conn.execute(
                    'SELECT name, sample_thumb FROM face_clusters WHERE id=?', (cid,)
                ).fetchone()
                cname = row[0] if row else None
                if row and not row[1]:
                    conn.execute('UPDATE face_clusters SET sample_thumb=? WHERE id=?',
                                 (face['thumb_path'], cid))
                prev = people.get(cid)
                if prev is None or face['score'] > prev['score']:
                    people[cid] = {
                        'cluster_id': cid,
                        'cluster_name': cname,
                        'thumb_filename': os.path.basename(face['thumb_path']),
                        'score': face['score'],
                    }
            sampled += 1
        frame_idx += 1
        if total and frame_idx >= total:
            break

    cap.release()
    conn.commit()
    conn.close()

    result = [{'cluster_id': p['cluster_id'], 'cluster_name': p['cluster_name'],
               'thumb_filename': p['thumb_filename']} for p in people.values()]
    print(json.dumps({'faces': result, 'frames_sampled': sampled}))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(json.dumps({'error': 'Usage: detect_video.py <video_path> <db_path> <photo_dir>'}))
        sys.exit(1)
    run_detect_video(sys.argv[1], sys.argv[2], sys.argv[3])
