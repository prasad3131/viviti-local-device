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

SAMPLE_EVERY_SEC = 1.0   # aim ~1 sample per second (fallback when frame count unknown)
MAX_FRAMES       = 18    # hard cap on detections — keeps runtime well under timeout


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

    # grab() cheaply skips frames without decoding; retrieve() decodes only the
    # ones we sample. Far faster than POS_MSEC seeking (which decodes from a
    # keyframe each time) and bounds work on high-fps slow-mo clips. Space
    # MAX_FRAMES samples across the whole video.
    if total > 0:
        step = max(1, int(total / MAX_FRAMES))
    else:
        step = max(1, int(round(fps * SAMPLE_EVERY_SEC)))

    people = {}            # cluster_id -> best { ..., score }
    sampled = 0
    read = 0

    while sampled < MAX_FRAMES:
        if not cap.grab():
            break
        read += 1
        if read % step != 0:
            continue
        ok, frame = cap.retrieve()
        if ok and frame is not None:
            seed = f'{video_path}#{read}'
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
                if prev is None:
                    people[cid] = {
                        'cluster_id': cid,
                        'cluster_name': cname,
                        'thumb_filename': os.path.basename(face['thumb_path']),
                        'score': face['score'],
                        'frames': 1,           # how many sampled frames this person is in
                    }
                else:
                    prev['frames'] += 1        # used-set => one increment per frame
                    if face['score'] > prev['score']:
                        prev['score'] = face['score']
                        prev['thumb_filename'] = os.path.basename(face['thumb_path'])
                        prev['cluster_name'] = cname
            sampled += 1

    cap.release()
    conn.commit()
    conn.close()

    # Keep real people, drop one-off false positives: a genuine subject appears in
    # many sampled frames. Named (matched to a known person) is high-confidence, so
    # a lower bar; unnamed needs to recur.
    min_unnamed = max(2, round(sampled * 0.10))
    result = []
    for p in people.values():
        keep = (p['cluster_name'] and p['frames'] >= 1) or (not p['cluster_name'] and p['frames'] >= min_unnamed)
        if keep:
            result.append({'cluster_id': p['cluster_id'], 'cluster_name': p['cluster_name'],
                           'thumb_filename': p['thumb_filename']})
    print(json.dumps({'faces': result, 'frames_sampled': sampled}))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(json.dumps({'error': 'Usage: detect_video.py <video_path> <db_path> <photo_dir>'}))
        sys.exit(1)
    run_detect_video(sys.argv[1], sys.argv[2], sys.argv[3])
