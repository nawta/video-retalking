#!/usr/bin/env python3
"""
compose_lipsync_overlay.py
================================
元フレーム (`pyframes`) に対して、LASER_ASD で得られたトラック情報と
VideoReTalking で生成した唇同期済み顔領域フレームを合成し、
フルフレームの動画を生成するユーティリティ。

使い方 (コンテナ内):
    python compose_lipsync_overlay.py \
        --asd_dir /shared_data/intermediate/asd_output \
        --lipsync_dir /shared_data/intermediate/lipsynced_output \
        --outfile /shared_data/intermediate/synced_output.mp4

依存: opencv-python, numpy, tqdm, pillow
"""
import argparse, os, cv2, pickle, numpy as np
from tqdm import tqdm

FPS = 25  # LASER_ASD で固定値


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--asd_dir', required=True,
                   help='Path to ASD output dir which contains pyframes/ and pywork/tracks_scores_speech_segmented.pckl')
    p.add_argument('--lipsync_dir', required=True,
                   help='Root directory that contains lipsynced_output/<track_segment_id>/*.png')
    p.add_argument('--s2st_dir', required=False,
                   help='Directory that contains *_translated.wav files produced by SeamlessExpressive')
    p.add_argument('--outfile', required=True,
                   help='Path to output mp4 file')
    return p.parse_args()


def load_segments(pckl_path):
    with open(pckl_path, 'rb') as f:
        data = pickle.load(f)
    segments = {}
    # data は list[dict] track_id, segment_id など
    for item in data:
        tid = item['track_id']
        sid = item.get('segment_id', 0)
        seg_key = f"{tid:06d}_{sid:02d}"
        segments[seg_key] = item
    return segments


def build_frame_mapping(segments, lipsync_dir):
    """return dict: frame_idx -> list of (face_img_path, bbox)"""
    mapping = {}
    for seg_key, item in segments.items():
        frames = item['frame']  # numpy array
        bboxes = item['bbox']  # numpy array (N,4)
        for idx_in_seg, frame_idx in enumerate(frames):
            face_img = os.path.join(lipsync_dir, seg_key, f"{idx_in_seg:06d}.png")
            if not os.path.isfile(face_img):
                continue
            bbox = bboxes[idx_in_seg]
            mapping.setdefault(int(frame_idx), []).append((face_img, bbox))
    return mapping


def main():
    args = parse_args()
    pyframes_dir = os.path.join(args.asd_dir, 'pyframes')
    pckl_path = os.path.join(args.asd_dir, 'pywork', 'tracks_scores_speech_segmented.pckl')
    assert os.path.isdir(pyframes_dir), f"pyframes not found: {pyframes_dir}"
    assert os.path.isfile(pckl_path), f"pckl missing: {pckl_path}"

    # フレーム一覧
    frame_paths = sorted([os.path.join(pyframes_dir, f) for f in os.listdir(pyframes_dir) if f.lower().endswith('.jpg')])
    total_frames = len(frame_paths)
    if total_frames == 0:
        raise RuntimeError('No frames found in pyframes')

    # セグメント情報読み込み
    segments = load_segments(pckl_path)
    mapping = build_frame_mapping(segments, args.lipsync_dir)

    # VideoWriter 設定
    sample_img = cv2.imread(frame_paths[0])
    h, w = sample_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video = args.outfile + '.tmp.mp4'
    writer = cv2.VideoWriter(temp_video, fourcc, FPS, (w, h))

    for fidx, frame_path in tqdm(enumerate(frame_paths), total=total_frames, desc='Compositing'):
        frame = cv2.imread(frame_path)
        if frame is None:
            writer.write(np.zeros((h, w, 3), dtype=np.uint8))
            continue
        overlays = mapping.get(fidx, [])
        for face_path, bbox in overlays:
            face_img = cv2.imread(face_path, cv2.IMREAD_UNCHANGED)
            if face_img is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            # 境界チェック
            x1 = max(0, min(w-1, x1)); x2 = max(1, min(w, x2))
            y1 = max(0, min(h-1, y1)); y2 = max(1, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            # リサイズ
            face_resized = cv2.resize(face_img, (x2 - x1, y2 - y1))
            if face_resized.shape[2] == 4:
                # 透過 PNG の場合 alpha ブレンド
                alpha = face_resized[:, :, 3] / 255.0
                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (1-alpha) * frame[y1:y2, x1:x2, c] + alpha * face_resized[:, :, c]
            else:
                frame[y1:y2, x1:x2] = face_resized
        writer.write(frame)
    writer.release()

    # ----------------------------------------------------------
    # 音声ミックス処理 (オプション)
    # ----------------------------------------------------------
    if args.s2st_dir and os.path.isdir(args.s2st_dir):
        audio_json = os.path.join(args.asd_dir, 'pywork', 'audio_segments_speech_segmented.json')
        if os.path.isfile(audio_json):
            import json, tempfile, subprocess, shlex

            with open(audio_json, 'r') as f:
                meta = json.load(f)

            inputs = []
            filter_parts = []
            index = 0
            for item in meta:
                base_name = os.path.splitext(os.path.basename(item['audio_file']))[0]
                translated = os.path.join(args.s2st_dir, f"{base_name}_translated.wav")
                if not os.path.isfile(translated):
                    continue
                inputs.append(translated)
                delay_ms = int(item['start_time'] * 1000)
                # FFmpeg stream index: 0 is video, so first audio input is 1, etc.
                stream_idx = len(inputs)  # since appended already
                filter_parts.append(f"[{stream_idx}:a]adelay={delay_ms}|{delay_ms}[a{index}]")
                index += 1

            if inputs:
                # build filter_complex
                mix_inputs = ';'.join(filter_parts)
                a_labels = ''.join([f"[a{i}]" for i in range(index)])
                filter_complex = f"{mix_inputs};{a_labels}amix=inputs={index}[mix]"

                # prepare command
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_video,
                ]
                for inp in inputs:
                    cmd += ['-i', inp]
                cmd += [
                    '-filter_complex', filter_complex,
                    '-map', '0:v', '-map', '[mix]',
                    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                    args.outfile,
                    '-loglevel', 'error'
                ]

                subprocess.run(cmd, check=True)
                os.remove(temp_video)
                print('Saved video with audio to', args.outfile)
                return

    # audio なしの場合: encode video only
    os.system(f"ffmpeg -y -i {temp_video} -c:v libx264 -crf 18 -pix_fmt yuv420p {args.outfile} -loglevel error")
    os.remove(temp_video)
    print('Saved silent video to', args.outfile)


if __name__ == '__main__':
    main() 
