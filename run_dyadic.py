# coding: utf-8

__author__ = 'cleardusk'
import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import imageio
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
# from utils.render_ctypes import render
from utils.functions import cv_draw_landmark, get_suffix
import time


def prepare_results(i, landmarks, bbox, pose, confidence, args):
    landmarks = list(zip(landmarks[0], landmarks[1], landmarks[2]))
    results = [i-1, bbox, landmarks, pose, confidence]
    return results
 
def main(args):
    results = []
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, gpu_id=args.gpu_id, **cfg)
        face_boxes = FaceBoxes(gpu_mode=gpu_mode)

    # Given a video path
    fn = args.video_fp.split('/')[-1]
    reader = imageio.get_reader(args.video_fp)

    fps = reader.get_meta_data()['fps']
    suffix = get_suffix(args.video_fp)
    video_wfp = os.path.join(args.output_folder, f'{args.opt}_smooth.mp4')
    if args.save_video:
        writer = imageio.get_writer(video_wfp, fps=fps)

    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()

    # run
    dense_flag = args.opt in ('2d_dense', '3d',)
    pre_ver = None
    for i, frame in tqdm(enumerate(reader)):
        if args.start > 0 and i < args.start:
            continue
        if args.end > 0 and i > args.end:
            break

        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if i == 0 or i == args.start:
            # detect
            boxes = face_boxes(frame_bgr)
            boxes = [boxes[0]]
            param_lst, roi_box_lst, poses, confidences = tddfa(frame_bgr, boxes, retrieve_pose=args.pose)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            pose = poses[0] if len(poses) > 0 else []
            conf = confidences[0]

            # refine
            param_lst, roi_box_lst, poses, confidences = tddfa(frame_bgr, [ver], crop_policy='landmark', retrieve_pose=args.pose)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # padding queue
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())

            for _ in range(n_pre):
                queue_frame.append(frame_bgr.copy())
            queue_frame.append(frame_bgr.copy())

        else:
            param_lst, roi_box_lst, poses, confidences = tddfa(frame_bgr, [pre_ver], crop_policy='landmark', retrieve_pose=args.pose)

            # todo: add confidence threshold to judge the tracking is failed
            if roi_box_lst is None or abs(roi_box_lst[0][2] - roi_box_lst[0][0]) * abs(roi_box_lst[0][3] - roi_box_lst[0][1]) < 2020:
                boxes = face_boxes(frame_bgr)
                # Check boxes are inside a region of the image
                def within_limits(shape, box):
                    PERC = 0.1
                    c1 = box[0] > PERC * shape[1] and box[1] > PERC * shape[0]
                    c2 = box[2] < (1-PERC) * shape[1] and box[3] < (1-PERC) * shape[0]
                    return c1 and c2
                    
                old = boxes
                boxes = [box for box in boxes if within_limits(frame.shape, box)]
                if len(boxes) == 0:
                    #print(f"[NO FACE] Frame {i}.")
                    if args.save_video:
                        writer.append_data(frame_bgr[:, :, ::-1])
                    if args.save_annotations:
                        results.append([i-1, -1, -1, -1, -1])
                    continue
                boxes = [boxes[0]]
                param_lst, roi_box_lst, poses, confidences = tddfa(frame_bgr, boxes, retrieve_pose=args.pose)
                if param_lst is None:
                    if args.save_video:
                        writer.append_data(frame_bgr[:, :, ::-1])
                    if args.save_annotations:
                        results.append([i-1, -1, -1, -1, -1])
                    continue
                conf = confidences[0]

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            pose = poses[0] if len(poses) > 0 else []

            queue_ver.append(ver.copy())
            queue_frame.append(frame_bgr.copy())

        pre_ver = ver  # for tracking

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)

            if args.opt == '2d_sparse':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
            elif args.opt == '2d_dense':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
            elif args.opt == '3d':
                img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
            else:
                raise ValueError(f'Unknown opt {args.opt}')

            if args.save_video:
                writer.append_data(img_draw[:, :, ::-1])  # BGR->RGB
            if args.save_annotations:
                results.append(prepare_results(i, ver_ave, roi_box_lst[0], pose, conf, args))

            queue_ver.popleft()
            queue_frame.popleft()

    # we will lost the last n_next frames, still padding
    for _ in range(n_next):
        queue_ver.append(ver.copy())
        queue_frame.append(frame_bgr.copy())  # the last frame

        ver_ave = np.mean(queue_ver, axis=0)

        if args.opt == '2d_sparse':
            img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
        elif args.opt == '2d_dense':
            img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
        elif args.opt == '3d':
            img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
        else:
            raise ValueError(f'Unknown opt {args.opt}')

        if args.save_video:
            writer.append_data(img_draw[..., ::-1])  # BGR->RGB
        if args.save_annotations:
            results.append(prepare_results(i+1, ver_ave, roi_box_lst[0], pose, conf, args))

        queue_ver.popleft()
        queue_frame.popleft()

    if args.save_video:
        writer.close()
        print(f'Dump to {video_wfp}')
    if args.save_annotations:
        annotation_file = os.path.join(args.output_folder, f"face_fiducials.npy")
        np.save(annotation_file, results)
        print(f'Dump to {annotation_file}')

def run_from_list(args):
    assert os.path.exists(args.list)

    with open(args.list, "r") as f:
        lines = f.readlines()
    
    root = lines[0].replace("\n", "")
    output = args.output_folder

    for i in range(1, len(lines)):
        t0 = time.time()
        args.video_fp = lines[i].replace("\n", "")
        args.output_folder = os.path.join(output, args.video_fp.replace(root, "").split(".")[0])

        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
        elif not args.replace and os.path.exists(os.path.join(args.output_folder, f"face_fiducials.npy")):
            print(f"[{i}/{len(lines)-1}] Already processed video -> '{args.output_folder}'")
            continue

        print(f"[{i}/{len(lines)-1}] Processing video '{args.video_fp}' -> '{args.output_folder}'")
        main(args)
        print(f"[FINISHED] Time: {time.time() - t0} s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of video of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-out', '--output_folder', type=str, required=True)
    parser.add_argument('-f', '--video_fp', type=str)
    parser.add_argument('-l', '--list', type=str, help="path lists of videos to process")
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse', choices=['2d_sparse', '2d_dense', '3d'])
    parser.add_argument('-s', '--start', default=-1, type=int, help='the started frames')
    parser.add_argument('-e', '--end', default=-1, type=int, help='the end frame')
    parser.add_argument('--onnx', action='store_true', default=False, help='if False, GPU will be used')
    parser.add_argument('--replace', action='store_true', default=False, help='if True, annotation files will be replaced')
    parser.add_argument('--save_annotations', action='store_true', default=False, help='if True, annotation results will be saved')
    parser.add_argument('--save_video', action='store_true', default=False, help='if True, final video will be saved')
    parser.add_argument('--gpu_id', default=0, type=int, help='gpu id to use')
    parser.add_argument('--pose', action='store_true', default=False, help='if True, pose will be computed')

    args = parser.parse_args()

    if args.list:
        run_from_list(args)
    else:
        main(args)
