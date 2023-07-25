import cv2
import numpy as np

from global_variables import BUFFER_LINE
from math_operations import predict_point_pos, calculate_direction_pt, add_degree


def plot_tracking(image, ROI_coordinates, online_targets, CLASSES, lines=None, frame_id=1, fps=0., ids2=None,
                  ignoring_regions=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    # top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    # text_scale = max(1, image.shape[1] / 1600.)
    # text_thickness = 2
    # line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 1

    # radius = max(5, int(im_w / 140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(online_targets)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
    # first_time = True
    if ignoring_regions:
        for region in ignoring_regions:
            x1, y1, w, h = region.tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=(0, 0, 0), thickness=-1)

    for i, t in enumerate(online_targets):
        # print(t.tlwh)

        x1, y1, w, h = t.tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        # print(intbox)
        obj_id = int(t.track_id)
        score_id = t.score
        id_text = '{}'.format(int(obj_id))
        score_text = '{:.2f}'.format(score_id)

        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        alpha = 1

        tracking_line = list(map(t.tlbr_to_bottom_center_point, t.tracking_tlbr[-BUFFER_LINE:]))
        pts = np.array(tracking_line).astype(int)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(im, [pts], False, t.color, 2)
        # print(t, t.category)

        if t.exit:
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=t.color, thickness=line_thickness)
            cv2.putText(im, id_text + " " + CLASSES[int(t.category)] + " " + score_text,
                        (intbox[0], intbox[1]),
                        cv2.FONT_HERSHEY_PLAIN, text_scale,
                        (255, 0, 0),
                        thickness=text_thickness)

        elif t.entry:
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=t.color, thickness=line_thickness)

            cv2.putText(im, id_text + " " + CLASSES[int(t.category)] + " " + score_text,
                        (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale,
                        (0, 255, 0),
                        thickness=text_thickness)
        else:
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=t.color, thickness=line_thickness)
            cv2.putText(im, id_text + " " + CLASSES[int(t.category)] + " " + score_text,
                        (intbox[0], intbox[1]),
                        cv2.FONT_HERSHEY_PLAIN, text_scale,
                        (0, 0, 255),
                        thickness=text_thickness)

    # exit(1)

    if lines:
        for l in lines:
            im = draw_line_number_letters(im, l)

        cv2.rectangle(im, ROI_coordinates[0:2], ROI_coordinates[2:4], (0, 215, 255), 2)

    return im


def plot_tracking_1(image, ROI_coordinates, online_targets, CLASSES, lines=None, frame_id=1, fps=0., ids2=None,
                    ignoring_regions=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    # top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    # text_scale = max(1, image.shape[1] / 1600.)
    # text_thickness = 2
    # line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 1

    # radius = max(5, int(im_w / 140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(online_targets)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
    # first_time = True
    if ignoring_regions:
        for region in ignoring_regions:
            x1, y1, w, h = region.tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=(0, 0, 0), thickness=-1)

    for i, t in enumerate(online_targets):
        # print(t.tlwh)

        x1, y1, w, h = t.tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        # print(intbox)
        obj_id = int(t.track_id)
        score_id = t.score
        id_text = '{}'.format(int(obj_id))
        score_text = '{:.2f}'.format(score_id)

        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        alpha = 1

        tracking_line = list(map(t.tlbr_to_bottom_center_point, t.tracking_tlbr[-BUFFER_LINE:]))
        pts = np.array(tracking_line).astype(int)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(im, [pts], False, t.color, 2)
        # print(t, t.category)

        if t.exit:
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=t.color, thickness=line_thickness)
            cv2.putText(im, id_text + " " + CLASSES[int(t.category)] + " " + score_text,
                        (intbox[0], intbox[1]),
                        cv2.FONT_HERSHEY_PLAIN, text_scale,
                        (255, 0, 0),
                        thickness=text_thickness)

        elif t.entry:
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=t.color, thickness=line_thickness)

            cv2.putText(im, id_text + " " + CLASSES[int(t.category)] + " " + score_text,
                        (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale,
                        (0, 255, 0),
                        thickness=text_thickness)
        else:
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=t.color, thickness=line_thickness)
            cv2.putText(im, id_text + " " + CLASSES[int(t.category)] + " " + score_text,
                        (intbox[0], intbox[1]),
                        cv2.FONT_HERSHEY_PLAIN, text_scale,
                        (0, 0, 255),
                        thickness=text_thickness)

    # exit(1)

    if lines:
        for l in lines:
            im = draw_region_number_letters(im, l)

        # cv2.rectangle(im, ROI_coordinates[0:2], ROI_coordinates[2:4], (0, 215, 255), 2)

    return im


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_bounding_box(im, t, lines, index, start_plotting=False):
    if start_plotting:
        if index <= len(t.tracking_tlbr) - 1:
            tlbr = t.tracking_tlbr[index]
            intbox = tuple(map(int, (tlbr[0], tlbr[1], tlbr[2], tlbr[3])))
            cv2.rectangle(im, intbox[0:2], intbox[2:4], color=t.color, thickness=2)
            tracking_line = t.show_tracking_line[:index]
            pts = np.array(tracking_line).astype(int)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(im, [pts], False, t.color, 2)
    for l in lines:
        im = draw_region_number_letters(im, l)
    return im


def draw_line_number_letters(frame, line):
    # Draw entry line and letters
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.5
    font_thickness = 2
    arrow_thickness = 2

    color_entry = (0, 255, 0)
    color_line = (255, 255, 255)
    color_exit = (255, 0, 0)
    cv2.line(frame, line.left_point, line.right_point, color_line, 2)

    cv2.putText(frame, line.name("Entry"),
                (line.right_point[0], line.right_point[1] - 15),
                font,
                font_scale, color_entry, font_thickness)

    cv2.putText(frame, line.name("Exit"),
                (line.left_point[0], line.left_point[1] + 15),
                font,
                font_scale, color_exit, font_thickness)

    cv2.arrowedLine(frame, line.left_point, predict_point_pos(line.left_point, 25, add_degree(
        calculate_direction_pt(line.right_point, line.left_point), 90), integer=True),
                    color_exit, arrow_thickness)

    cv2.arrowedLine(frame, line.right_point, predict_point_pos(line.right_point, 25, add_degree(
        calculate_direction_pt(line.left_point, line.right_point), 90), integer=True),
                    color_entry, arrow_thickness)
    return frame


def draw_region_number_letters(frame, region):
    # print("COME HERE")
    # Draw entry region and letters
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.5
    font_thickness = 4
    arrow_thickness = 4
    letters_thickness = 3

    # Define the width and height of the rectangle

    color_entry = (0, 255, 0)
    color_region = (255, 255, 255)
    color_exit = (255, 0, 0)

    # cv2.region(frame, region.left_point, region.right_point, color_region, 2)

    points = np.array(region.coordinates)
    #print(region.coords)
    points = points.reshape((-1, 1, 2))
    cv2.polylines(frame, [points], True, color_region, 3)

    # print(region.left_point, region.right_point)
    tl, tr, br, bl = region.tl, region.tr, region.br, region.bl
    arrow_size = region.min_dist / 2

    # Define the size and position of the subregions

    exit_center = region.exit_center

    entry_center = region.entry_center

    exit_arrow_pred_point = predict_point_pos(exit_center, arrow_size, add_degree(
        calculate_direction_pt(tr, tl), 90), integer=True)
    entry_arrow_pred_point = predict_point_pos(entry_center, arrow_size, add_degree(
        calculate_direction_pt(bl, br), 90), integer=True)

    # Draw the blue arrow and letter on the left subregion

    cv2.arrowedLine(frame, (int(exit_center[0]), int(exit_center[1])), exit_arrow_pred_point, (255, 0, 0), arrow_thickness)
    cv2.putText(frame, region.name("Exit"), (int(exit_center[0]), int(exit_center[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_exit, letters_thickness)

    # Draw the green letter and arrow on the right subregion

    cv2.arrowedLine(frame, (int(entry_center[0]), int(entry_center[1])), entry_arrow_pred_point,
                    (0, 255, 0), arrow_thickness)
    cv2.putText(frame, region.name("Entry"), (int(entry_center[0]), int(entry_center[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_entry, letters_thickness)

    """# Draw the white line in the center of the rectangle
    cv2.line(frame, (exit_subregion_center[0], tl[1]), (exit_subregion_center[0], br[1]),
             (255, 255, 255), 1)

    cv2.putText(frame, region.name("Entry"),
                (tr[0], tr[1] - 15),
                font,
                font_scale, color_entry, font_thickness)

    cv2.putText(frame, region.name("Exit"),
                (bl[0], bl[1] + 15),
                font,
                font_scale, color_exit, font_thickness)

    cv2.arrowedLine(frame, tr, predict_point_pos(tr, 25, add_degree(
        calculate_direction_pt(tl, tr), 90), integer=True),
                    color_entry, arrow_thickness)


    cv2.arrowedLine(frame, bl, predict_point_pos(bl, 25, add_degree(
        calculate_direction_pt(br, bl), 90), integer=True),
                    color_exit, arrow_thickness)"""

    return frame


def write_ground_truth_on_frame(frame, targets, ignored_regions):
    for target in targets:
        # print(t.tlwh)
        for region in ignored_regions:
            x1, y1, w, h = region.tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            cv2.rectangle(frame, intbox[0:2], intbox[2:4], color=(0, 0, 0), thickness=-1)

        box = target.find("box")

        x1, y1, w, h = float(box.get("left")), float(box.get("top")), float(box.get("width")), float(box.get("height"))

        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        cv2.rectangle(frame, intbox[0:2], intbox[2:4], color=get_color(abs(int(target.get("id")))), thickness=3)
        obj_id = target.get("id")
        text_thickness = 2
        cv2.putText(frame, obj_id,
                    (intbox[0], intbox[1]),
                    cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255),
                    thickness=2)



"""def plot_tracking(image, args, online_targets, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    # text_scale = max(1, image.shape[1] / 1600.)
    # text_thickness = 2
    # line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 1

    radius = max(5, int(im_w / 140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(online_targets)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
    first_time = True
    for i, t in enumerate(online_targets):
        x1, y1, w, h = t.tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(t.track_id)
        score_id = t.score
        id_text = '{}'.format(int(obj_id))
        score_text = '{:.2f}'.format(score_id)
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))
        alpha = 1
        if not t.entry and args.load_line != "None":
            if first_time:
                overlay = im.copy()
                first_time = False
            else:
                overlay = overlay.copy()

            cv2.rectangle(overlay, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
            cv2.putText(overlay, id_text + " " + CLASSES[int(t.category)] + " " + score_text,
                        (intbox[0], intbox[1]),
                        cv2.FONT_HERSHEY_PLAIN, text_scale,
                        (0, 0, 255),
                        thickness=text_thickness)

            cv2.addWeighted(overlay, alpha, im, 1 - alpha,
                            0, im)
        else:
            if not t.exit and args.load_line != "None":
                cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)

                cv2.putText(im, id_text + " " + CLASSES[int(t.category)] + " " + score_text,
                            (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale,
                            (0, 255, 0),
                            thickness=text_thickness)
            else:
                if first_time:
                    overlay = im.copy()
                    first_time = False
                else:
                    overlay = overlay.copy()

                cv2.rectangle(overlay, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
                cv2.putText(overlay, id_text + " " + CLASSES[int(t.category)] + " " + score_text,
                            (intbox[0], intbox[1]),
                            cv2.FONT_HERSHEY_PLAIN, text_scale,
                            (255, 0, 0),
                            thickness=text_thickness)
                cv2.addWeighted(overlay, alpha, im, 1 - alpha,
                                0, im)
    if args.load_line != "None":
        lines, ROI_coordinates = args.lines, args.ROI_coordinates

        for l in lines:
            im = draw_line_number_letters(im, l)

        cv2.rectangle(im, ROI_coordinates[0:2], ROI_coordinates[2:4], (0, 215, 255), 2)
    return im


"""

"""def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img"""
