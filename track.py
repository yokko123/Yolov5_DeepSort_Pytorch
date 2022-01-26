# limit the number of cpus used by high performance libraries
import os
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh, \
    increment_path
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn








import streamlit as st
cnt = 0
bb_area = []

def detection(opt, stframe,kpi1_text,kpi2_text,kpi3_text):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, half = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.imgsz, opt.evaluate, opt.half
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    tracked_object = 0
    fire_count = 0
    smoke_count = 0
    no_smoke_count = 0

    tracked_object = 0
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(opt.yolo_weights, device=device, dnn=opt.dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid =check_imshow()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                # draw boxes for visualization
                prev_area = 0
                prev_distance = 0
                prev_midx = 0
                prev_midy = 0

                if len(outputs) > 0:

                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        tracked_object = id
                        # print("class: ", output[0:4])

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        if (c == 0):
                            fire_count = fire_count + 1
                            mid_x = (output[0] + output[2]) / 2
                            mid_y = (output[1] + output[3]) / 2
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            present_area = bbox_h * bbox_w

                            # f_id = id
                            # mid_x = (output[0] + output[2]) / 2
                            # mid_y = (output[1] + output[3]) / 2

                            apx_distance = round((((720 - output[3])) * 0.0025) * 4.5, 1)
                            present_distance = apx_distance - prev_distance
                            present_midx = mid_x - prev_midx
                            present_midy = mid_y - prev_midy
                            present_area = bbox_h * bbox_w

                            cv2.putText(im0, '{:.1f}'.format(present_area), (int(mid_x), int(mid_y)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(im0, 'Distancce = {:.1f}'.format(present_distance), (int(mid_x+20), int(mid_y+20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(im0, 'Center(x,y) = %s , %s'%(mid_x,mid_y), (int(mid_x+40), int(mid_y+40)),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



                            prev_area = present_area
                            prev_midy = mid_y
                            prev_midx = mid_x
                            prev_distance = apx_distance

                        elif c == 1:
                            smoke_count = smoke_count + 1
                        elif c== 2:
                            no_smoke_count = no_smoke_count + 1
                            #print(bb_area[len(bb_area)-1])
                            #
                            # cv2.putText(im0, "FIRE", (100, 100), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                            # cv2.putText(im0, "checking if the fire is valid or not", (100, 150), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
                            # diff = cv2.absdiff(frame2, im0)
                            # gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                            # blur = cv2.GaussianBlur(gray, (5, 5), 0)
                            # _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                            # dilated = cv2.dilate(thresh, None, iterations=3)
                            # contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            #
                            # for contour in contours:
                            #     (x, y, w, h) = cv2.boundingRect(contour)
                            #
                            #     if cv2.contourArea(contour) < 2500:
                            #         continue
                            #     cv2.rectangle(im0, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            #     cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            #                 1, (0, 0, 255), 3)
                            #
                            # # cv2.drawContours(frame1, contours, -1, (0,255,0), 2)
                            # #cv2.imshow('output', im0)
                            # frame1 = frame2
                            # ret, frame2 = cap.read()

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1,
                                                               -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results

            im0 = annotator.result()

            fps = 0


            fps = (fps + (1. / (time.time() - t1))) / 2
            stframe.image(im0, channels='BGR', use_column_width=True)
            kpi1_text.write(f"<h5 style= 'text-align:left; color: red'>{'{:.1f}'.format(fps)}</h5>",
            unsafe_allow_html=True)
            kpi2_text.write(f"<h5 style= 'text-align:left; color: red'>{tracked_object}</h5>",
                            unsafe_allow_html=True)
            kpi3_text.write(f"<h5 style= 'text-align:left; color: red'></h5>")
            if show_vid:
                cv2.namedWindow(p, cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow(p,600,600)
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
    # Print results

    print(fire_count, smoke_count, no_smoke_count)
    st.empty()
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--yolo_weights', nargs='+', type=str, default='fire.pt', help='model.pt path(s)')
#     parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
#     # file/folder, 0 for webcam
#     parser.add_argument('--source', type=str, default='0', help='source')
#     parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
#     parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
#     parser.add_argument('--save-vid', action='store_true', default=True, help='save video tracking results')
#     parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
#     # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--evaluate', action='store_true', help='augmented inference')
#     parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
#     parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
#     parser.add_argument('--visualize', action='store_true', help='visualize features')
#     parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#
#     with torch.no_grad():
#         detect(opt)
