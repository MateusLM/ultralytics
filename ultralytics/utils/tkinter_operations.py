import os.path
from copy import copy
from pathlib import Path
from tkinter import *
from tkinter.messagebox import askyesno

from tkfilebrowser import askopendirnames


def yes_no_box():
    return askyesno(title='Counting or Tracking',
                    message='You want to proceed to counting? (if answer is no, Bytetrack (without counting) will run '
                            'as default) ')


def open_directories(init_dir):
    master = Tk()
    master.withdraw()
    master.update()
    directories = askopendirnames(
        initialdir=init_dir,
        title="Select dataset folder")
    master.destroy()

    return directories


def choose_delete_or_change_cat():
    def sel():
        pass

    def destroy():
        root.quit()
        root.destroy()

    root = Tk()
    var = IntVar(root)
    var.set(1)
    Radiobutton(root, text="Change Category", variable=var, value=1,
                command=sel).pack(anchor=W)

    Radiobutton(root, text="Delete Category", variable=var, value=2,
                command=sel).pack(anchor=W)
    label = Label(root)
    label.pack()
    root.protocol("WM_DELETE_WINDOW", destroy)

    B = Button(root, text="OK", command=destroy)

    B.pack()

    root.mainloop()
    return var.get()


def choose_type_changes():
    def sel():
        pass

    def destroy():
        root.quit()
        root.destroy()

    root = Tk()
    var = IntVar(root)
    var.set(1)
    Radiobutton(root, text="Small Changes (Images 1 by 1)", variable=var, value=1,
                command=sel).pack(anchor=W)

    Radiobutton(root, text="Big Changes (Change all images at once)", variable=var, value=2,
                command=sel).pack(anchor=W)
    label = Label(root)
    label.pack()
    root.protocol("WM_DELETE_WINDOW", destroy)

    B = Button(root, text="OK", command=destroy)

    B.pack()

    root.mainloop()
    return var.get()


def change_cat_name(categories):
    def update_cats(categories):
        for i, cat in enumerate(vars):
            if cat.get() != "":
                categories[i] = cat.get()

        root.quit()
        root.destroy()

    root = Tk()
    vars = []
    for _ in categories:
        vars.append(StringVar(root))

    for i, cat in enumerate(categories):
        Label(root, text=cat).grid(row=i)

    Button(root, text="Enter", command=lambda: update_cats(categories), activebackground='green',
           justify='center').grid(row=i + 1, column=1)
    for i, var in enumerate(vars):
        Entry(root, textvariable=var).grid(row=i, column=1)

    root.mainloop()
    return categories


def update_excel_properties(filenames, excel_time, start_excel_time, end_excel_time, text_sites, is_project,
                            tkinter=True, changed=False):
    from time_operations import load_excel_time
    def destroy():
        root.quit()
        root.destroy()

    print("tkinter", tkinter)
    if not changed:
        excel_time.clear()
        start_excel_time.clear()
        end_excel_time.clear()
    root = None
    if tkinter:
        root = Tk()
        root.title('Excel Properties')

    text_sites_groups = [text_sites[i:i + 10] for i in range(0, len(text_sites), 10)]
    current_column = 0

    for group in text_sites_groups:
        i_tkinter = 0
        for i_site in range(len(group)):

            if is_project:

                start_excel_time_str, end_excel_time_str = load_excel_time(filenames[i_site][0]), load_excel_time(
                    filenames[i_site][-1])
                print(start_excel_time_str, end_excel_time_str)
            else:
                start_excel_time_str, end_excel_time_str = load_excel_time(filenames[i_site]), load_excel_time(
                    filenames[i_site])

            if tkinter:
                Label(root, text=group[i_site]).grid(row=i_tkinter, column=current_column)
                Label(root, text="Excel Time").grid(row=i_tkinter + 1, column=current_column)
                excel_time_aux = [StringVar(root) for _ in start_excel_time_str]
                [time_.set(start_excel_time_str[i]) for i, time_ in enumerate(excel_time_aux)]

                for i in range(len(excel_time_aux)):
                    Entry(root, textvariable=excel_time_aux[i], width=2,
                          state=DISABLED if start_excel_time_str == end_excel_time_str else NORMAL).grid(
                        row=i_tkinter + 1, column=i + current_column + 1)
                Label(root, text="of " + ":".join(end_excel_time_str)).grid(row=i_tkinter + 1,
                                                                            column=current_column + 4)
                excel_time.append(excel_time_aux)
            else:
                excel_time.append(":".join(start_excel_time_str))

            start_excel_time_str = "06:00:00".split(":")
            if tkinter:
                Label(root, text="Start Excel Time").grid(row=i_tkinter + 2, column=current_column)
                start_excel_time_aux = [StringVar(root) for _ in start_excel_time_str]
                [time_.set(start_excel_time_str[i]) for i, time_ in enumerate(start_excel_time_aux)]

                for i in range(len(start_excel_time_aux)):
                    Entry(root, textvariable=start_excel_time_aux[i], width=2).grid(row=i_tkinter + 2,
                                                                                    column=current_column + i + 1)

                start_excel_time.append(start_excel_time_aux)
            else:
                start_excel_time.append(":".join(start_excel_time_str))

            end_excel_time_str = "06:00:00".split(":")
            if tkinter:
                Label(root, text="End Excel Time").grid(row=i_tkinter + 3, column=current_column)

                end_excel_time_aux = [StringVar(root) for _ in end_excel_time_str]
                [time_.set(end_excel_time_str[i]) for i, time_ in enumerate(end_excel_time_aux)]

                for i in range(len(end_excel_time_aux)):
                    Entry(root, textvariable=end_excel_time_aux[i], width=2).grid(row=i_tkinter + 3,
                                                                                  column=current_column + i + 1)

                end_excel_time.append(end_excel_time_aux)
            else:
                end_excel_time.append(":".join(end_excel_time_str))

            # old_excel_time = excel_time[:]
            # old_start_excel_time = start_excel_time[:]
            # old_end_excel_time = end_excel_time[:]
            i_tkinter += 4
        current_column += 5
    if tkinter:
        Button(root, text="Enter", command=destroy, activebackground='green', justify='center').grid(
            row=0, column=current_column)
        root.mainloop()
    if tkinter:
        excel_time[:] = [":".join([str(var.get()) for var in sublist]) for sublist in excel_time]

        start_excel_time[:] = [":".join([str(var.get()) for var in sublist]) for sublist in start_excel_time]
        end_excel_time[:] = [":".join([str(var.get()) for var in sublist]) for sublist in end_excel_time]

        # print(old_excel_time, old_start_excel_time, old_end_excel_time)


def update_start_video_time(start_video_time, duration, text_sites, tkinter=True):
    from time_operations import convert_seconds_to_string
    def destroy():
        root.quit()
        root.destroy()

    print("tkinter", tkinter)

    start_video_time.clear()
    root = None
    if tkinter:
        root = Tk()
        root.title('Start Video Time')

    text_sites_groups = [text_sites[i:i + 10] for i in range(0, len(text_sites), 10)]
    current_column = 0
    print("TEXT SITES", text_sites_groups)
    for group in text_sites_groups:
        print("NEXT GROUP", current_column)
        for i_site in range(len(group)):
            video_time_lst = "00:00:00".split(":")

            if tkinter:
                start_video_time_aux = [StringVar(root) for _ in video_time_lst]
                [time_.set(video_time_lst[i]) for i, time_ in enumerate(start_video_time_aux)]
                start_video_time.append(start_video_time_aux)
            else:
                start_video_time.append(":".join(video_time_lst))
        if tkinter:
            i_tkinter = 0
            for i_site in range(len(group)):
                Label(root, text=group[i_site]).grid(row=i_tkinter, column=current_column)
                for i in range(len(start_video_time[i_site])):
                    Entry(root, textvariable=start_video_time[i_site][i], width=2).grid(row=i_tkinter + 1,
                                                                                        column=current_column + i + 1)
                Label(root, text="of " + convert_seconds_to_string(int(duration[i_site]))).grid(row=i_tkinter + 1,
                                                                                                column=current_column + 4)
                i_tkinter += 2
            current_column += 6  # increment the column after each group

    if tkinter:
        Button(root, text="Enter", command=destroy, activebackground='green', justify='center').grid(
            row=0, column=current_column)
        root.mainloop()
        start_video_time[:] = [":".join([str(var.get()) for var in sublist]) for sublist in start_video_time]


def update_load_line(load_line, filenames, dir_depth, continuous, is_project, object_name, draw_option, tkinter=True):
    def destroy():
        root.quit()
        root.destroy()

    print("tkinter", tkinter)
    load_line.clear()
    object_name.clear()

    i_site_line = 0
    root = None
    if tkinter:
        root = Tk()
        root.title('Loading...')
    # print(filenames)
    for i_site, file_site in enumerate(filenames):
        if is_project:
            filename = file_site[0]
            proj = os.path.basename(Path(filename).parents[dir_depth[i_site]]).split(" ")[0]
            p = os.path.basename(Path(filename).parents[dir_depth[i_site] - 1])
            obj = proj + "_" + p + "_" + os.path.basename(filename).split(".")[0].split("_")[
                1] + "." + draw_option.get()

            # print(os.listdir(os.getcwd() + "/load_lines/" + proj))
            list_drawn = [file for file in os.listdir(os.getcwd() + "/SOFT_outputs/" + proj + "/load_draw/") if
                          file.endswith('.' + draw_option.get())]

            print(list_drawn, draw_option.get(), obj)
            if tkinter:
                has_line = False

                if list_drawn:
                    for line_obj in list_drawn:
                        if line_obj == obj:
                            has_line = True
                            load_line.append(StringVar(root))
                            load_line[i_site_line].set(obj)
                            break

                if not has_line or not list_drawn:
                    load_line.append(StringVar(root))
                    load_line[i_site_line].set("None")
            else:
                load_line.append(obj) if obj in list_drawn else load_line.append(
                    None)
            object_name.append(proj + "_" + p + "_" + os.path.basename(filename).split(".")[0].split("_")[1])
            i_site_line += 1
        else:

            if all(depth == 0 for depth in dir_depth):
                # print(file_site)

                obj = "Default_" + os.path.basename(file_site).split(".")[0] + "." + draw_option.get()
                list_drawn = [file for file in os.listdir(os.getcwd() + "/SOFT_outputs/Default/load_draw/") if
                              file.endswith('.' + draw_option.get())]

                if tkinter:
                    has_line = False
                    for line_obj in list_drawn:
                        print(line_obj, obj, line_obj == obj)
                        if line_obj == obj:
                            has_line = True
                            load_line.append(StringVar(root))
                            load_line[i_site_line].set(obj)
                            break

                    if not has_line or not list_drawn:
                        load_line.append(StringVar(root))
                        load_line[i_site_line].set("None")

                else:
                    load_line.append(obj) if obj in list_drawn else load_line.append(None)
                object_name.append("Default_" + os.path.basename(file_site).split(".")[0])
            else:
                # TODO: FIX dir_depth
                # print(file_site, dir_depth)
                proj = os.path.basename(Path(file_site).parents[dir_depth[i_site]]).split(" ")[0]
                p = os.path.basename(Path(file_site).parents[dir_depth[i_site] - 1])

                obj = proj + "_" + p + "_" + os.path.basename(file_site).split(".")[0].split("_")[
                    1] + "." + draw_option.get()

                list_drawn = [file for file in os.listdir(os.getcwd() + "/SOFT_outputs/" + proj + "/load_draw/") if
                              file.endswith('.' + draw_option.get())]
                if tkinter:
                    load_line.append(StringVar(root))
                    load_line[i_site].set(obj) if obj in list_drawn else load_line[
                        i_site].set("None")
                else:
                    load_line.append(obj) if obj in list_drawn else load_line.append(None)
                object_name.append(
                    proj + "_" + p + "_" + os.path.basename(file_site).split(".")[0].split("_")[1])
            if continuous:
                break
    if tkinter:

        site_groups = [filenames[i:i + 10] for i in range(0, len(filenames), 10)]
        current_column = 0
        i_line = 0

        for site_group in site_groups:
            for i_site, site_files in enumerate(site_group):
                if is_project:
                    site_label = os.path.basename(Path(site_files[0]).parents[dir_depth[i_site] - 1])
                    Label(root, text=site_label).grid(row=i_site + 1, column=current_column)
                else:
                    site_label = os.path.basename(site_files)
                    Label(root, text=site_label).grid(row=i_site + 1, column=current_column)

                options = [load_line[i_line].get()]
                if load_line[i_line].get() != "None":
                    options.append("None")
                OptionMenu(root, load_line[i_line], *options).grid(row=i_site + 1, column=current_column + 1)
                i_line += 1
            current_column += 2
        Button(root, text="Enter", command=destroy, activebackground='green', justify='center').grid(
            row=0, column=current_column)
        root.mainloop()
    if tkinter:
        aux = copy(load_line)
        for i, var in enumerate(load_line):
            aux[i] = None if (var.get() == "None" or var.get() == "") else var.get()
        load_line[:] = aux
