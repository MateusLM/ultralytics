import math
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
#from pyfastcopy import copyfile


def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print("Directory '%s' can not be created")


def write_to_txt(save_folder, filename, counter, draw_option, fps):
    filename = os.path.basename(filename).split(".")[0]
    if draw_option:
        name = "_" + draw_option
    else:
        name = ""
    ext = ".txt"
    save_path = os.path.join(save_folder, filename + "_count" + name + ext)
    with open(save_path, "w") as file:
        if draw_option:
            for movement in counter:
                tracks = [*counter[movement].values()]
                tracks = [item for sublist in tracks for item in sublist]
                if len(movement) != 1:
                    file.write("Movement: {} | Track ID: {} \n".format(movement, tracks))
        file.write("FPS: {}".format(fps))


def create_folders(proj):
    print()
    create_folder("SOFT_outputs/" + proj + '/load_draw/' )
    create_folder("SOFT_outputs/" + proj + '/photos/')
    create_folder("SOFT_outputs/" + proj + '/tracks/')
    create_folder("SOFT_outputs/" + proj + '/excel/')


def get_data_from_xml(filename):
    try:
        with open(filename, 'r') as f:
            data = f.read()
    except FileNotFoundError:
        data = None
        print('File does not exist')

    return data


def get_ignored_regions(data):
    from ignoring_region import IgnoringRegion
    root = ET.fromstring(data)
    boxes_ignored_region_list = root.findall("./ignored_region/box")
    ignoring_regions = []
    for box in boxes_ignored_region_list:
        x1, y1, w, h = float(box.get("left")), float(box.get("top")), float(box.get("width")), float(box.get("height"))
        ignoring_regions.append(IgnoringRegion([x1, y1, w, h]))

    return ignoring_regions


def convert_xml_regions_to_bounding_box(regions):
    bbregions = []
    for box in regions:
        x1, y1, w, h = float(box.get("left")), float(box.get("top")), float(box.get("width")), float(box.get("height"))
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        bbregions.append(intbox)
    return bbregions


def get_ground_truth(data, frame_id):
    root = ET.fromstring(data)
    frame = root.find(".//frame[@num=" + '"' + str(frame_id) + '"]')
    if frame:
        targets = frame.findall('./target_list/target')
    else:
        print(frame_id)
        exit(1)
        return []
    return targets


def create_xml_half_resolution(filename):
    data = get_data_from_xml(filename)

    root = ET.fromstring(data)

    print(root)
    boxes_ignored_region_list = root.findall("./ignored_region/box")
    for box in boxes_ignored_region_list:
        box.attrib["left"], box.attrib["top"], box.attrib["width"], box.attrib["height"] = str(
            float(box.get("left")) / 2), str(
            float(box.get("top")) / 2), str(float(box.get("width")) / 2), str(float(box.get("height")) / 2)

    frames_targets_list = root.findall('./frame/target_list')
    print(len(frames_targets_list))

    for frame_targets_list in frames_targets_list:
        targets = frame_targets_list.findall('./target')
        for target in targets:
            box = target.find("./box")
            box.attrib["left"], box.attrib["top"], box.attrib["width"], box.attrib["height"] = str(
                float(box.get("left")) / 2), str(
                float(box.get("top")) / 2), str(float(box.get("width")) / 2), str(float(box.get("height")) / 2)
            attribute = target.find("./attribute")
            attribute.attrib["speed"] = str(float(attribute.get("speed")) / 2)

            region_overlap_box = target.findall("./occlusion/region_overlap")
            if region_overlap_box:
                for box in region_overlap_box:
                    box.attrib["left"], box.attrib["top"], box.attrib["width"], box.attrib["height"] = str(
                        float(box.get("left")) / 2), str(
                        float(box.get("top")) / 2), str(float(box.get("width")) / 2), str(float(box.get("height")) / 2)

    tree_as_str = ET.tostring(root, encoding='utf8', method='xml')

    return BeautifulSoup(tree_as_str, "xml").prettify()


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    print('save results to {}'.format(filename))


def not_in_use(filename):
    try:
        with open(filename, "r") as file:
            # Print the success message
            print("File has opened for reading.")
    # Raise error if the file is opened before
    except IOError:
        print("File has opened already.")


def search_sites_in_files(files):
    sites = []
    for file in files:
        site = os.path.basename(Path(file).parents[1])
        if site not in sites:
            sites.append(site)
    return sites


def search_first_file(directory, video=True):
    if video:
        for r, d, f in os.walk(directory):
            for file in f:
                if file.endswith(".mp4") or file.endswith(".flv") or file.endswith(
                        ".avi") or file.endswith(
                    ".3gp") or file.endswith(".MPG"):
                    print(os.path.join(r, file))
                    return [os.path.join(r, file)]


def search_files_in_directory(directory, video=True):
    if video:
        files = [os.path.join(r, file) for r, d, f in os.walk(directory) for file in
                 f if
                 file.endswith(".mp4") or file.endswith(".flv") or file.endswith(
                     ".avi") or file.endswith(
                     ".3gp") or file.endswith(".MPG")]
    else:
        files = [os.path.join(r, file) for r, d, f in os.walk(directory) for file in
                 f if
                 file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(
                     ".png")]

    files.sort()
    return files


def search_images_in_directories(directories):
    print("search images...")

    files = [os.path.join(r, file) for dir in directories for r, d, f in os.walk(dir) for file in
             f if
             file.endswith(".jpg") or file.endswith(".png")]
    print("sorting...")
    files.sort()
    print("finish!")
    return files


def search_txt_in_directories(directories):
    print("search annotations...")
    files = [os.path.join(r, file) for dir in directories for r, d, f in os.walk(dir) for file in
             f if
             file.endswith(".txt")]
    print("sorting...")
    files.sort()
    print("finish!")
    return files


def get_depth(directory_main, filenames):
    startinglevel = directory_main[0].count(os.sep)
    """if len(directory_main) != 1:
        startinglevel -= 1"""
    toplevel = max([filename.count(os.sep) for filename in filenames])
    return toplevel - startinglevel


def txt_has_categories(annot_p, categories_relevant_index):
    with open(annot_p, 'r') as fp:
        # read an store all lines into list
        lines = fp.readlines()

    for l in lines:
        info = l.split()
        if int(info[0]) in categories_relevant_index:
            return True
    return False


def txt_count_categories(annot_p, counter, categories):
    with open(annot_p, 'r') as fp:
        # read an store all lines into list
        lines = fp.readlines()

    for l in lines:
        info = l.split()
        category = categories[int(info[0])]
        if category not in counter:
            counter[category] = 1
        else:
            counter[category] += 1

    return counter


def write_yaml_file(yaml_final_path, correspondence):
    with open(yaml_final_path, "w") as yaml_f:
        contents = {}
        categories = [name if correspondence[name] in correspondence.values() else -1 for name in
                      correspondence]
        contents["names"] = categories
        contents["nc"] = len(categories)
        contents["train"] = "./images/train/"
        contents["val"] = "./images/val/"
        print("DUMPING", contents)
        yaml.dump(contents, yaml_f)
        return contents


def create_video_from_images(filenames):
    import cv2

    output_folder = str(Path(filenames[0]).parents[1])
    create_folder(output_folder)
    output_video_file = output_folder + "/" + os.path.basename(str(Path(filenames[0]).parents[0])) + ".mp4"
    if not os.path.exists(output_video_file):

        cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        frame = cv2.imread(filenames[0])
        size = list(frame.shape)
        del size[2]
        size.reverse()

        out = cv2.VideoWriter(output_video_file, cv2_fourcc, 60, size)

        for i, filename in enumerate(filenames):
            print("Frame", i, "of", len(filenames))
            img = cv2.imread(filename)
            out.write(img)

        out.release()
    return output_video_file


def update_cat_annot(annot_p, categories, correspondence, change=None):
    print("new")
    print(correspondence, categories, change)
    # Write file
    with open(annot_p, 'r') as fp:
        # read and store all lines into list
        lines = fp.readlines()
    with open(annot_p, 'w') as fp:
        print("Writing...")
        # iterate each line
        for number, line in enumerate(lines):
            lst = line.split()
            # print(lst[0])
            if not change:
                category = categories[int(lst[0])]
                if int(lst[0]) != correspondence[category]:
                    print("Changing line", number + 1)

                    print("From", int(lst[0]), ":", category, "to", correspondence[category])
                    lst[0] = str(correspondence[category])

                    fp.write(" ".join(lst) + "\n")

                else:
                    fp.write(" ".join(lst) + "\n")
            else:
                if int(lst[0]) in change:
                    print("From", int(lst[0]), ":", correspondence[int(lst[0])], "to", change[int(lst[0])], ":",
                          categories[change[int(lst[0])]])
                    lst[0] = str(change[int(lst[0])])
                    fp.write(" ".join(lst) + "\n")
                else:
                    fp.write(" ".join(lst) + "\n")


def create_train_val_test_files(base_outdir, files, train_p=0.7, val_p=0.2, size=1, balanced=False):
    if train_p + val_p >= 1 or size > 1:
        exit(1)

    imgs, annots = files
    #list_shuffled = list(zip(imgs, annots))
    #random.shuffle(list_shuffled)
    #imgs, annots = zip(*list_shuffled)
    imgs, annots = imgs[: math.floor(size * len(imgs))], annots[: math.floor(size * len(imgs))]

    if size != 1:
        if balanced:
            add_str = "_balanced"
        else:
            add_str = ""
        base_outdir = base_outdir + "data_" + str(size) + "_coco" + add_str + "/"
    else:
        if balanced:
            add_str = "_balanced"
        else:
            add_str = ""
        base_outdir = base_outdir + "data" + add_str + "/"
    # imgs_final_path = base_outdir + "images/"
    # annots_final_path = base_outdir + "labels/"
    # create_folder(imgs_final_path)
    # create_folder(annots_final_path)
    for folder in ['train', 'val', 'test']:
        print(folder)
        target_dir = base_outdir + folder
        img_dir = target_dir + "/" + "images/"
        annot_dir = target_dir + "/" + "labels/"
        print(img_dir)
        create_folder(img_dir)
        create_folder(annot_dir)

        if folder == 'train':
            images_to_pass, annots_to_pass = imgs[: math.floor(train_p * len(imgs))], annots[: math.floor(
                train_p * len(imgs))]
        elif folder == 'val':
            images_to_pass, annots_to_pass = imgs[math.floor(train_p * len(imgs)): math.floor(
                (train_p + val_p) * len(imgs))], annots[math.floor(train_p * len(imgs)): math.floor(
                (train_p + val_p) * len(imgs))]
        else:
            images_to_pass, annots_to_pass = imgs[math.floor((train_p + val_p) * len(imgs)):], annots[math.floor(
                (train_p + val_p) * len(imgs)):]
        for i, [img_path, annot_path] in enumerate(zip(images_to_pass, annots_to_pass)):
            print(folder, "File", i, " of ", len(images_to_pass))
            copy_file(img_path, img_dir + "/" + os.path.basename(img_path))
            copy_file(annot_path, annot_dir + "/" + os.path.basename(annot_path))

    return base_outdir


def copy_file(old_path, new_path):
    copyfile(old_path, new_path)
