import argparse
import os
import platform
import shutil
import time
import copy
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pandas import DataFrame
from scipy.signal import savgol_filter

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import load_classifier, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def linear_interp(x: 'np.array') -> 'np.array':
    x = DataFrame(x).interpolate(method='linear',
                                 axis=0,
                                 limit_direction='both').to_numpy()
    return x


def smoothing(arr: 'np.array([x,y,w,h,m])', fps):
    # (w,h) do not get smoothed here!
    # get odd fps
    odd = lambda x: x if x % 2 != 0 else x + 1
    x = copy.deepcopy(arr)
    x[:, :2] = savgol_filter(x[:, :2], odd(int(fps*2)), 3, mode='nearest', axis=0)
    x[:, :2] = savgol_filter(x[:, :2], odd(int(fps)), 3, axis=0)
    x[:, :2] = savgol_filter(x[:, :2], odd(int(fps/1.5)), 2, axis=0)
    x[:, -1] = savgol_filter(x[:, -1], odd(int(fps*2)), 3, axis=0)
    x[:, -1] = savgol_filter(x[:, -1], odd(int(fps)), 2, axis=0)
    x[:, -1] = savgol_filter(x[:, -1], odd(int(fps/2)), 1, axis=0)
    x[:, -1] = savgol_filter(x[:, -1], odd(int(fps/4)), 1, axis=0)
    return x.astype(int)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    if not os.path.exists(opt.smooth_txt):
        os.makedirs(opt.smooth_txt)

    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # # Second-stage classifier
    # classify = False
    # if classify:
    #     modelc = load_classifier(name='resnet101', n=2)  # initialize
    #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
    #     modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    # dataset contains all the frames (or images) of the video
    crds_crop = np.empty((0, 4))  # contains coordinates of a single bbox with the highest conf
    np_nan = np.empty([1, 4])  # for tracking
    np_nan[:] = np.nan  # for tracking
    frame_no = 0
    for path, img, im0s, vid_cap in dataset:  # im0s, img - initial, resized and padded (img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):  # only when obj is in the frame
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # these will be used for the deepsort input
                xywhs = xyxy2xywh(det[:, :4].cpu())
                confs = det[:,4].cpu()
                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confs, im0)  # this is numpy array
                ###########################################################
                # FOR NOW, WE WILL ONLY BE KEEPING THE MOST CONFIDENT VALUE
                ###########################################################
                max_conf_id = confs.argmax()
                # keeping the coordinates row with max conf (det now only keeps one row and four columns)
                det = det[max_conf_id, :].reshape(1, 6)
                to_append = xyxy2xywh(det[:, :4].cpu().numpy().reshape(1, 4).astype(int))
                if len(crds_crop) == 0:
                    crds_crop = np.append(crds_crop, to_append).reshape(-1, 4)
                else:
                    crds_crop = np.append(crds_crop, to_append, axis=0)
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    # draw_boxes(im0, bbox_xyxy, identities)  # no tracking boxes for now
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (*xyxy, conf))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s' % (names[int(cls)])
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            else:
                deepsort.increment_ages()
                if len(crds_crop) == 0:
                    crds_crop = np.append(crds_crop, np_nan).reshape(-1, 4)
                else:
                    crds_crop = np.append(crds_crop, np_nan, axis=0)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)

                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
        frame_no += 1
    ##############################################################################
    # this part should be temporary as online filtering will be implemented
    crds_crop = linear_interp(crds_crop)
    max_side_bbox = crds_crop[:, 2:].max(axis=1) * 1.2  # 20% relaxation
    # making sure that the window size does not exceed frame size
    max_side_bbox = np.where(max_side_bbox < min(w,h), max_side_bbox, min(w,h))
    crds_crop = np.c_[crds_crop, max_side_bbox]
    crds_crop = smoothing(crds_crop, fps)
    np.savetxt(os.path.join(opt.smooth_txt, os.path.basename(path)[:-4] + '_savgol_' + '.txt'),
               crds_crop, delimiter=' ')
    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/best_yolov4-p5-results.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--config_deepsort', type=str, default='deep_sort_pytorch/configs/deep_sort.yaml')
    parser.add_argument('--smooth-txt', type=str, default='inference/coordinates', help='coordinates output folder')
    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
