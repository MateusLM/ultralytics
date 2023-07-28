#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import argparse
import ast
import configparser
import gzip
import os
import os.path as osp
import pickle
import sys
from copy import copy, deepcopy
from pathlib import Path

import cv2
import torch
#from loguru import logger

#from byte_tracker import BYTETracker_Counter
#from drawWidget import DrawWidget
#from excel import Excel
#from file_operations import get_depth, search_files_in_directory, get_data_from_xml, \
#    convert_xml_regions_to_bounding_box, get_ignored_regions
from ultralytics.cfg.app.interface import SimpleApp, InfoWindow
#from models.experimental import attempt_load
#from nn import attempt_load_one_weight
#from predictor import Predictor
#from time_operations import load_excel_time, convert_seconds_to_string, search_for_valid_filenames
#from tracking_classifying import CLS_TRACK
#from utils.general import check_img_size
#from utils.torch_utils import TracedModel, select_device
#from video import Video
#from yolox.exp import get_exp
#from yolox.utils import fuse_model, get_model_info

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
#from yolox.tracking_utils.timer import Timer


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument("-type", default="video", help="demo type, eg. image, video and webcam"
                        )

    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/default/yolox_m.py",
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )

    # tracking args
    # track_thresh was 0.5 .changed to 0.2
    # track_buffer was 30 changed to 50
    # match_thresh was 0.8
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

    parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        action="store_true",
        help="Show video frame by frame and outputs information",
    )

    parser.add_argument(
        "--no_count",
        dest="count",
        default=True,
        action="store_false",
        help="Disables Counting (with Lines)",
    )

    parser.add_argument(
        "--discontinuous",
        dest="continuous",
        default=True,
        action="store_false",
        help="Videos aren't one followed by another",
    )

    parser.add_argument(
        "--no_use_region",
        dest="use_region",
        default=True,
        action="store_false",
        help="Do not use lines",
    )

    parser.add_argument(
        "--use_line",
        dest="use_line",
        default=False,
        action="store_false",
        help="Use lines instead of regions",
    )

    parser.add_argument("--draw_option", type=str, default="regions", help="select type of drawing")

    parser.add_argument(
        "--server",
        dest="server",
        default=False,
        action="store_true",
        help="server mode (no interface, only terminal output)",
    )

    parser.add_argument(
        "--no_excel",
        dest="excel",
        default=True,
        action="store_false",
        help="Do not create excel"
    )
    parser.add_argument('--use_ignoring_regions', default=False,
                        action="store_true", help="Regions to ignore ")

    # datetime.strptime("00:00:00", '%H:%M:%S')
    parser.add_argument('--filenames', nargs='+', type=str, default=[], help="files path")
    parser.add_argument("--video_time", nargs='+', type=str, default=[], help="init time video")
    parser.add_argument("--excel_time", nargs='+', type=str, default=[], help="init time video in excel")
    parser.add_argument("--start_excel_time", nargs='+', type=str, default=[], help="start project time")
    parser.add_argument("--end_excel_time", nargs='+', type=str, default=[], help="end project time")
    parser.add_argument("--interval_counting", type=str, default="15", help="init time video")
    parser.add_argument("--net_model_name", type=str, default="yolov8n.pt", help="init time video")
    parser.add_argument(
        "--interface",
        default=True,
        action="store_true",
        help="show interface",
    )
    parser.add_argument(
        "--load_line", nargs='+', type=str, default=[], help="load previous drawn stuff", )
    parser.add_argument(
        "--show",
        action="store_true",
        help="show video detection",
    )
    parser.add_argument('--is_directory', type=bool, default=False, help="choose folder instead of files")

    # Yolov7
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')

    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def add_tracks(tracks, new_tracks):
    for new_track in new_tracks:
        track_id = new_track.track_id
        index = next((i for i, t in enumerate(tracks) if t.track_id == track_id), None)
        if index is not None:
            tracks[index] = new_track
        else:
            tracks.append(new_track)

    return tracks


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},{c},-1,-1\n'
    # print(filename, results)
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores, categories in results:
            for tlwh, track_id, score, category in zip(tlwhs, track_ids, scores, categories):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                # print(tlwh, category)
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2), c=category)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def load_line(load_line):
    #print('load_lines/' + load_line.split("_")[0] + "/" + load_line, )
    with open("SOFT_outputs/" + load_line.split("_")[0] + '/load_draw/' + load_line, 'rb') as object_file:
        lines = pickle.load(object_file)
    # TODO DELETE
    # TODO: REMOVE
    #for l in lines:
        #l.intersection_points = {"Entry": [], "Exit": []}
        #l.dir_l = {"Entry": [], "Exit": []}
    return lines


def draw_lines(cls_track, tracks, i_site, draw_option):
    drawWidget = DrawWidget("first_frame", args)
    tracks = cls_track.preProcessing(tracks)
    lines = drawWidget.get_properties(tracks, i_site)

    args.load_line[i_site] = args.object_name[i_site] + "." + draw_option
    #print("Create object")
    proj = args.load_line[i_site].split("_")[0]
    with open("SOFT_outputs/" + proj + '/load_draw/' + args.load_line[i_site], 'wb') as object_file:
        print("WRITING OBJECT")
        print("SOFT_outputs/" + proj + '/load_draw/' + args.load_line[i_site])
        pickle.dump(lines,
                    object_file)


def save_results(save_folder, filename, results, draw_option):
    # print("saving...", results)
    filename = osp.basename(filename).split(".")[0]
    ext = ".txt"
    if draw_option:
        name = "_" + draw_option
    else:
        name = ""
    save_path = osp.join(save_folder, filename + name + ext)
    write_results(save_path, results)
    logger.info(f"save results to {save_path}")


def export_associated_tracks(esc=False):
    if not args.debug and args.excel:
        if esc:
            print("EXPORTING TRACKS SO FAR")
        else:
            print("Export all associated movements")

        path = "SOFT_outputs/" + args.object_name[args.line_index].split("_")[0] + '/tracks/' + args.object_name[
            args.line_index] + "_" + \
               args.net_model_name.split(".")[0] + "_" + args.interval_counting + "_" + args.draw_option + ".tracks"
        for mov in args.associated_movements:
            print(mov.name, mov.tracks)
        with gzip.open(path, 'wb') as object_file:
            print()

            print("dumping...")

            pickle.dump(args.associated_movements, object_file)
            print("finished")


def append_associated_tracks(tracker):
    if not args.debug and args.excel:
        for mov in range(len(tracker.movements)):
            for t_new in tracker.movements[mov].tracks:
                # if is_new_track_associated(t, aux[i].tracks):
                print("adding", t_new, t_new.entry, t_new.exit)
                args.associated_movements[mov].tracks.append(t_new)


def imageflow_demo(predictor, vis_folder, args):
    timer_tot = Timer()
    timer_tot.tic()

    vid_writer, counter, excel, tracker, info_window, cls_track, lines = None, None, None, None, None, None, None

    i_line = 0

    if args.use_ignoring_regions and args.draw_option:
        ignoring_regions = []
        for filename in args.filenames:
            filename, ext = os.path.splitext(os.path.basename(filename[0]))
            # print(filename)
            xml_filename = "./YOLO_outputs/gt/" + filename + ".xml"
            data = get_data_from_xml(xml_filename)
            regions = get_ignored_regions(data) if data else []

            ignoring_regions.append(regions)
        args.ignoring_regions = ignoring_regions
        print("CREATED")
    test = False
    if args.draw_option:
        for i_site in range(len(args.filenames)):
            tracks = []
            if args.is_project:
                args.f = args.filenames[i_site]
            else:
                args.f = [args.filenames[i_site]]

            args.num_videos = len(args.f)

            for i, filename in enumerate(args.f):
                if not args.load_line[i_site]:
                    cap = cv2.VideoCapture(filename if args.type == "video" else args.camid)
                    video = Video(args, cap, filename)
                    args.track_thresh = 0.2 if not video.is_good_quality() else args.track_thresh
                    args.track_buffer = 55 if not video.is_good_quality() else args.track_buffer
                    args.filename_id = i

                    if args.server:
                        sys.exit("Server Mode: Create line object to run the software")

                    if args.continuous:
                        args.line_index = i_site
                        if i == 0:
                            args.n_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) * len(args.f)
                            tracker = BYTETracker_Counter(args, frame_rate=video.fps_fixed)
                            cls_track = CLS_TRACK(args, None, predictor, tracker, None, None, video, None)
                        cls_track.video = video
                    else:
                        args.line_index = i
                        args.n_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        tracker = BYTETracker_Counter(args, frame_rate=video.fps_fixed)
                        cls_track = CLS_TRACK(args, None, predictor, tracker, None, None, video, None)

                    cont = False
                    if not test:
                        cont, new_tracks = cls_track.main(cap)

                        add_tracks(tracks, new_tracks)

                    if not args.continuous:
                        draw_lines(cls_track, tracks, args.line_index, args.draw_option)
                        tracks = []
                        tracker = None
                        cls_track = None
                    else:

                        if not cont or i == len(args.f) - 1:
                            draw_lines(cls_track, tracks, args.line_index, args.draw_option)
                            break
                else:
                    print(filename, "already has a line")
                    break
                i_line += 1
    else:
        print("evaluate without lines -> Bytetrack")
    break_out_flag = False
    for i_site in range(len(args.filenames)):
        if break_out_flag:
            break
        print("NEW SITE", i_site)
        results, tracks = [], []
        if args.is_project:
            args.f = args.filenames[i_site]
        else:
            args.f = args.filenames

        args.num_videos = len(args.f)
        for i, filename in enumerate(args.f):
            args.filename_id = i
            cap = cv2.VideoCapture(filename if args.type == "video" else args.camid)

            if args.continuous:
                args.n_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) * len(args.f)
                args.line_index = i_site
            else:
                args.n_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                args.line_index = i

            video = Video(args, cap, filename)
            args.track_thresh = 0.2 if not video.is_good_quality() else args.track_thresh
            args.track_buffer = 55 if not video.is_good_quality() else args.track_buffer
            print("NEW VIDEO")
            if args.type == "video" and args.save_result:
                if args.continuous:
                    filename_output = args.f[0]
                else:
                    filename_output = args.f[i]
                args.save_folder = osp.join(vis_folder, osp.basename(filename_output).split(".")[0])
                os.makedirs(args.save_folder, exist_ok=True)
                if args.draw_option:
                    name = "_" + args.draw_option
                else:
                    name = ""
                save_path = osp.join(args.save_folder, osp.basename(filename_output).split(".")[0] + name + ".mp4" )

                logger.info(f"video save_path is {save_path}")
                vid_writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*"mp4v"), video.fps_fixed, (int(video.width), int(video.height))
                )

            if args.filename_id == 0 or not args.continuous:
                if not args.draw_option:
                    print("comes here, new tracker, new cls_track object")
                    tracker = BYTETracker_Counter(args, frame_rate=video.fps_fixed)
                    cls_track = CLS_TRACK(args, None, predictor, tracker, None, vid_writer, video, None)
                else:
                    lines = load_line(args.load_line[args.line_index])

                    ROI_coordinates = video.get_ROI(lines, offset=200, has_ROI=False)
                    print(args.video_time)
                    frame_id = video.get_properties(args.video_time[args.line_index], onlyFrame=True)
                    tracker = BYTETracker_Counter(args, lines, ROI_coordinates, frame_id=frame_id,
                                                  frame_rate=video.fps_fixed)

                    counter = tracker.init_counter()
                    args.associated_movements = deepcopy(tracker.movements)
                    if args.excel:
                        excel = Excel(args)
                        excel.init_excel(counter, args.line_index)
                    else:
                        excel = None
                    if args.interface and args.draw_option:
                        info_window = InfoWindow(
                            [0, video, args],
                            counter, i_site)
                    cls_track = CLS_TRACK(args, counter, predictor, tracker, excel, vid_writer, video, info_window,
                                          lines=lines, i_site=i_site)
            if args.continuous:
                # UPDATE VIDEO
                cls_track.video = video
                if args.interface and not cls_track.tracker.is_preprocessing:
                    info_window.video = video
                args.track_thresh = 0.2 if not video.is_good_quality() else args.track_thresh

            cont, results_aux = cls_track.main(cap)
            if not cont:
                break_out_flag = True
                export_associated_tracks(esc=break_out_flag)
                break

            if not args.continuous:
                if not args.debug and args.excel:
                    print("append associated")
                    append_associated_tracks(tracker)
                    # export_associated_tracks(args.line_index, tracker)
                    tracker.reset_movements()
                if args.save_result:
                    save_results(args.save_folder, filename_output, results_aux, args.draw_option)
            else:
                results.extend(results_aux)
                if i == len(args.f) - 1:
                    if not args.debug and args.excel:
                        print("append associated")
                        append_associated_tracks(tracker)

                        tracker.reset_movements()
                        tracker = None
                    if args.save_result:
                        save_results(args.save_folder, filename_output, results, args.draw_option)
        if not break_out_flag:
            export_associated_tracks(esc=break_out_flag)
    dur = timer_tot.toc()
    print(convert_seconds_to_string(round(dur, 0)))


def main(args, exp=None):
    torch.cuda.empty_cache()
    print(args.net_model_name)

    if not exp:
        print("DONT HAVE EXP")
        file_name = osp.join("./YOLO_outputs", args.net_model_name.split(".")[0])
    else:
        print("HAVE EXP")
        file_name = osp.join(exp.output_dir, exp.exp_name)
    print(file_name)
    os.makedirs(file_name, exist_ok=True)
    vis_folder = file_name
    if args.trt:
        args.device = "gpu"
    if not args.yolov7:
        args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    else:
        args.device = select_device("" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if exp:
        if args.conf is not None:
            exp.test_conf = args.conf
        if args.nms is not None:
            exp.nmsthre = args.nms
        if args.tsize is not None:
            exp.test_size = (args.tsize, args.tsize)
        args.img_size = (args.img_size, args.img_size)

    if args.yolov7 or args.yolor:
        # Load model
        """if args.update:  # update all models (to fix SourceChangeWarning)
            for args.net_model_name in ['yolov7.pt']:
                strip_optimizer(args.net_model_name)"""
        weights, imgsz, device, trace, half, source = args.net_model_name, args.img_size, args.device, not args.no_trace, args.fp16, \
                                                      args.filenames[0]

        if "6" in args.net_model_name:
            imgsz = 1280
        print(imgsz)

        model = attempt_load(weights, map_location=device)  # load FP32 model
        args.stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=args.stride)  # check img_size

        if not "-" in args.net_model_name or "_" in args.net_model_name:
            args.net_model_name = args.net_model_name.split(".")[0] + "_default." + args.net_model_name.split(".")[1]
        else:
            args.net_model_name = args.net_model_name.replace("-", "_")

        if trace:
            model = TracedModel(model, device, imgsz)

        if half:
            model.half()  # to FP16

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        # if not args.net_model_name.startswith("yolox"):
        # model = torch.hub.load('ultralytics/yolov5', args.net_model_name, force_reload=True)

        """"# Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once"""
    elif args.yolov8:
        print(args.net_model_name)
        weights, imgsz, device, trace, half, source = args.net_model_name, args.img_size, args.device, not args.no_trace, args.fp16, \
                                                      args.filenames[0]
        model, ckpt = attempt_load_one_weight(weights)
        args.stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=args.stride)  # check img_size

        if not "-" in args.net_model_name or "_" in args.net_model_name:
            args.net_model_name = args.net_model_name.split(".")[0] + "_default." + args.net_model_name.split(".")[1]
        else:
            args.net_model_name = args.net_model_name.replace("-", "_")

        #if trace:
            #model = TracedModel(model, device, imgsz)

        #if half:
            #model.half()  # to FP16

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    else:
        model = exp.get_model().to(args.device)
        # if not args.net_model_name.startswith("yolox"):
        # model = torch.hub.load('ultralytics/yolov5', args.net_model_name, force_reload=True)
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        model.eval()

        if not args.trt and args.net_model_name.startswith("yolox") or args.net_model_name.startswith("yolov"):
            ckpt_file = args.net_model_name
            logger.info("loading checkpoint")
            ckpt = torch.load(ckpt_file, map_location="cpu")
            # load the model state dict
            model.load_state_dict(ckpt["model"])
            logger.info("loaded checkpoint done.")

        print(torch.device("cuda"))

        # model = get_best_model(model, args)
        print(args.fp16, args.fuse)

        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)
            print("YE")

        if args.device == torch.device("cuda"):
            if torch.cuda.is_available():
                model.cuda()
                print("Cuda")

            if args.fp16:
                model.half()  # to FP16
                print("fp16")

        print(args.trt)
    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(file_name, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, trt_file=trt_file, decoder=decoder, args=args)
    # current_time = time.localtime()
    if args.type == "image":
        exit(1)
        # image_demo(predictor, vis_folder, current_time, args)
    elif args.type == "video" or args.type == "webcam":
        imageflow_demo(predictor, vis_folder, args)


if __name__ == "__main__":
    args = make_parser().parse_args()

    if args.server:
        args.interface = False
        args.show = False
        args.debug = False

        if not args.filenames:
            sys.exit("Specify the folder/filename to run!")

    # TODO: FIX
    if args.filenames:
        if os.path.isdir(args.filenames[0]):
            print("here")
            directory = args.filenames
            args.filenames = search_files_in_directory(directory)
            args.dir_depth = get_depth(directory, args.filenames)
            proj = os.path.basename(Path(args.filenames[0]).parents[args.dir_depth]).split(" ")[0]
            site = os.path.basename(Path(args.filenames[0]).parents[args.dir_depth - 1])
            hour = os.path.basename(args.filenames[0]).split(".")[0].split("_")[1]
            args.object_name = proj + "_" + site + "_" + hour
        else:
            args.object_name = os.path.basename(args.filenames[0]).split(".")[0]
        if not args.interface and os.path.isfile("SOFT_outputs/"+ proj + "/load_draw/" + args.object_name + ".lines"):
            args.load_line = args.object_name + ".lines"

    if args.interface and not args.filenames:
        interface = SimpleApp("700x700", "App", args)
        interface.mainloop()
        # TODO: Choose how many classes I want my program to classify -> expand to minibus, taxi, h2,h3+

        print("oi")
        args.filenames, args.video_time, args.excel_time, args.start_excel_time, args.end_excel_time, args.load_line, args.interval_counting, args.net_model_name, args.device, args.debug, args.show, args.dir_depth, args.draw_option, args.continuous, args.save_result, args.excel, args.is_project, args.object_name = interface.record
        config = configparser.ConfigParser()
        config.read("cfg/labels/config_classes.ini")
        sector = config.get("correspondence", args.net_model_name.split(".")[0])
        args.CLASSES = ast.literal_eval(config.get(sector, "CLASSES"))
        args.categories = ast.literal_eval(config.get(sector, "RELEVANT"))
        print(args.excel_time)
        # identify = args.categories
        # args.categories = [identify[i] for i in range(len(identify)) if identify_bool[i]]
        """for i_site in range(len(args.filenames)):
            if load_excel_time(args.filenames[i_site][0], args.excel_time[i_site], interface=False) != args.excel_time[i_site]:
                args.filenames = search_for_valid_filenames(args.filenames[i_site], args.excel_time[i_site])
            args.excel_time = load_excel_time(args.filenames[i_site][0], args.excel_time[i_site], interface=False)"""

    if args.net_model_name.startswith("yolov7"):
        args.yolov7 = True
        args.yolor = False
        args.yolov8 = False
        exp = None
    elif args.net_model_name.startswith("yolor"):
        args.yolor = True
        args.yolov7 = False
        args.yolov8 = False
        exp = None
    elif args.net_model_name.startswith("yolov8"):
        args.yolov8 = True
        args.yolov7 = False
        args.yolor = False
        exp = None

    else:
        if args.net_model_name.startswith("yolox"):
            args.exp_file = "exps/default/yolox_" + args.net_model_name.split("_")[1] + ".py"
        else:
            args.exp_file = "exps/default/" + args.net_model_name[:-1] + ".py"
        exp = get_exp(args.exp_file, args.name)

    if args.debug:
        args.show = True
        args.excel = False
    if not args.draw_option:
        args.excel = False

    main(args, exp=exp)


def app():
    args, _ = make_parser().parse_known_args()

    interface = SimpleApp("700x700", "App", args)
    interface.mainloop()
