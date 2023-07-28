import os
from pathlib import Path
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilenames

from tkfilebrowser import askopendirnames

from ultralytics.utils.file_operations import get_depth, create_video_from_images, search_files_in_directory, create_folders, \
    search_first_file
from ultralytics.utils.time_operations import convert_seconds_to_string, convert_string_to_seconds, \
    search_for_valid_filenames
# TODO:ADD real life time to interface
from ultralytics.utils.tkinter_operations import update_load_line, update_start_video_time, update_excel_properties


class SimpleApp(Tk):
    def __init__(self, geometry, title, args):
        Tk.__init__(self)
        #self.filenames = args.filenames
        self.device = None
        self.load_line = []
        self.start_excel_time = []
        self.end_excel_time = []
        self.video_time = []
        self.excel_time = []
        self.text_sites = []
        self.net_model_name = None
        self.interval_counting = None

        self.cls_bools = None

        self.debug = None
        self.show = None
        self.draw_option = None
        self.continuous = None
        self.excel = None
        self.save_result = None

        self.is_project = False
        self.object_name = []
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.geometry(geometry)
        self.title(title)
        self.args = args
        self._frame = None
        self.record = None
        self.changed = False
        self.images = \
            False
        self.state = 2

        """if self.filenames:
            self.duration, _ = self.get_video_properties(index=0)
            self.state = 3

        else:
            self.state = 2"""
        self.switch_frame()

    def reset(self):
        self.load_line = []
        self.start_excel_time = []
        self.end_excel_time = []
        self.video_time = []
        self.excel_time = []
        self.text_sites = []

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self._frame.destroy()
            exit(1)

    def create_next_back(self):

        if self.state != 3:
            b_next = Button(self, text="Next", command=lambda: self.go_next())
        else:
            b_next = Button(self, text="Start", command=lambda: self.go_next())
        b_next.place(rely=1.0, relx=1.0, x=-10, y=-10, anchor=SE)

        b_back = Button(self, text="Back", command=lambda: self.go_back())
        b_back.place(rely=1.0, relx=0, x=10, y=-10, anchor=SW)
        if self.state == 1:
            b_next["state"] = NORMAL
            b_back["state"] = DISABLED

        if self.state == 2:
            b_next["state"] = DISABLED
            b_back["state"] = NORMAL

    def switch_frame(self):
        """Destroys current frame and replaces it with a new one."""
        print(self.state)
        if self.state == 1:
            new_frame = StartPage(self)
        elif self.state == 2:
            new_frame = ChooseVideoPage(self)
        elif self.state == 3:
            new_frame = ProgramSettingsPage(self)
        else:
            tkinter = False
            if not self.load_line and self.draw_option.get():
                update_load_line(self.load_line, self.filenames, self.dir_depth, self.continuous.get(), self.is_project,
                                 self.object_name, self.draw_option, tkinter=tkinter)

            if not self.excel_time and self.excel.get():
                update_excel_properties(self.filenames, self.excel_time,
                                        self.start_excel_time, self.end_excel_time,
                                        self.text_sites, self.is_project, tkinter=tkinter)
            if not self.video_time:
                update_start_video_time(self.video_time, self.duration, self.text_sites, tkinter=tkinter)

            # print(self.excel_time)
            # print(self.filenames)
            if self.is_project and not self.images:
                for i in range(len(self.filenames)):
                    self.filenames[i] = search_for_valid_filenames(self.filenames[i], self.excel_time[i])

            self.record = [self.filenames,
                           self.video_time,
                           self.excel_time,
                           self.start_excel_time,
                           self.end_excel_time,
                           self.load_line, self.interval_counting.get(),
                           self.net_model_name.get(),
                           self.device.get(), self.debug.get(), self.show.get(), self.dir_depth,
                           self.draw_option.get() if self.draw_option.get() != "None" else None,
                           self.continuous.get(), self.save_result.get(),
                           self.excel.get(), self.is_project, self.object_name]  # [i.get() for i in self.cls_bools],
            self.destroy()
            return

        if self._frame is not None:
            print("KILL")
            self._frame.destroy()

        self._frame = new_frame

        self._frame.pack()

    def go_next(self):
        self.state += 1
        self.switch_frame()

    def go_back(self):
        self.state -= 1
        self.reset()
        self.switch_frame()

    def open_image_files_directory(self):
        init_dir = "/media/planeamusafrente/Local Disk Planeamus/Videos Projetos" if os.name == "posix" else os.getcwd() + "/videos"
        self.directory = askopendirnames(
            initialdir=init_dir,
            title="Select image folder/s")
        if self.directory:
            self.filenames = []
            create_folders("DETRAC-Images")
            self.is_project = True
            self.images = True
            for i, directory in enumerate(self.directory):
                print(f"Folder images {i + 1} of {len(self.directory)}")
                images_path = search_files_in_directory(directory, video=False)
                # print(images_path)

                filename = create_video_from_images(images_path)

                self.filenames.append([filename])

            if self.filenames:
                self.n_sites = len(self.filenames)
                # ("NSITES", self.n_sites)
                # print(self.directory, self.filenames)
                self.dir_depth = [1 for site in self.filenames]
                self.duration, _ = self.get_video_properties(index=0)

                self.state += 1
        self.switch_frame()

    def open_video_files_path(self):
        video_formats = [".mp4", ".flv", ".avi", ".3gp", ".MPG"]
        init_dir = "/media/planeamusafrente/Local Disk Planeamus/Videos Projetos" if os.name == "posix" else os.getcwd() + "/videos"
        self.filenames = askopenfilenames(filetypes=[("all video format", f) for f in video_formats],
                                          initialdir=init_dir, title="Select file/s")

        if self.filenames:
            create_folders("Default")
            self.dir_depth = [0]
            self.n_sites = len(self.filenames)
            print(self.n_sites)
            self.duration, _ = self.get_video_properties(index=0)

            self.state += 1
            self.switch_frame()

    def open_video_files_directory(self):
        if os.name == "posix":
            init_dir = "/media/planeamusafrente/Local Disk Planeamus/Videos Projetos"
        elif os.name == "nt":
            init_dir = os.getcwd() + "/videos"
        self.directory = askopendirnames(
            initialdir=init_dir,
            title="Select folder/s")
        self.filenames = []
        if self.directory:
            print(self.directory)
            self.dir_depth = [get_depth(self.directory, search_first_file(dir)) for dir in self.directory]

            self.is_project = any(depth >= 2 for depth in self.dir_depth)

            for i, dir in enumerate(self.directory):
                files = search_files_in_directory(dir)
                folder_path = \
                    os.path.basename(Path(files[0]).parents[self.dir_depth[i] if self.dir_depth[i] != 3 else 2]).split(
                        " ")[0]
                create_folders(folder_path)
                # sites = search_sites_in_files(files)
                # dir_files_aux = []
                # for site in sites:
                # site_files = [file for file in files if site in file]
                # dir_files_aux.append(site_files)
                self.filenames.append(files)
        if any(self.filenames):
            self.n_sites = len(self.filenames)
            self.duration, _ = self.get_video_properties(index=0)
            self.state += 1

        self.switch_frame()

    def get_video_properties(self, index=0):
        from moviepy.editor import VideoFileClip
        # print(self.filenames)
        duration, frame_count = [], []
        for i_site in range(self.n_sites):
            if self.is_project:
                filename = self.filenames[i_site][index]
            else:
                filename = self.filenames[i_site]
            # print(filename)
            clip = VideoFileClip(filename)
            duration_aux = clip.duration
            frame_count_aux = int(clip.fps * duration_aux)
            # print(duration_aux, frame_count_aux)

            duration.append(duration_aux)
            frame_count.append(frame_count_aux)

        # print("CHECKKK", duration, frame_count)

        return duration, frame_count


class StartPage(Frame):
    def __init__(self, master):
        Frame.__init__(self, master)
        l = Label(self, text="Welcome to the automatic vehicle classification")
        l.pack(side="top", fill="x", pady=10)
        master.create_next_back()


class ChooseVideoPage(Frame):
    def __init__(self, master):
        print("CHEGUEI")
        Frame.__init__(self, master)
        master.create_next_back()
        button = Button(self, text="Select one or more videos", command=lambda: master.open_video_files_path())
        button.pack(fill="none", pady=50)  # change
        # button.pack()

        button = Button(self, text="Select one or more folders",
                        command=lambda: master.open_video_files_directory())
        button.pack(fill="none", pady=50)  # change
        # button.pack()
        button = Button(self, text="Select one or more folders containing multiple images",
                        command=lambda: master.open_image_files_directory())
        button.pack(fill="none", pady=50)  # change
        # button.pack()


class ProgramSettingsPage(Frame):

    def __init__(self, master):
        Frame.__init__(self, master)
        self.draw_option_menu = None
        master.create_next_back()
        # print(master.filenames)

        self.update_frame(master, init=True)

    def update_frame(self, master, init=True):

        if not init:
            # master.args.use_region = master.use_line.get()

            master.args.excel = master.excel.get()
            if master.debug.get():
                master.args.show = True

            if master.excel.get() and master.draw_option.get() == "None":
                master.args.draw_option = "regions"
            else:
                master.args.draw_option = master.draw_option.get()



            if master.excel_time:
                master.args.excel_time = [master.excel_time[i_site] for i_site in range(master.n_sites)]
                master.args.start_excel_time = [master.start_excel_time[i_site] for i_site in range(master.n_sites)]
                master.args.end_excel_time = [master.end_excel_time[i_site] for i_site in range(master.n_sites)]
            master.text_sites = []

            master.args.debug = master.debug.get()

        row = 0
        if master.is_project:
            txt = "Site name/s: "
        else:
            txt = "Video name/s: "

        # print(master.filenames, master.dir_depth)
        for i, depth in enumerate(master.dir_depth):

            if depth >= 2:
                master.text_sites.append(os.path.basename(Path(master.filenames[i][0]).parents[depth - 1]))

            else:
                if depth == 1:
                    master.text_sites.append(os.path.basename(master.filenames[i][0]))
                else:
                    master.text_sites.append(os.path.basename(master.filenames[i]))

        # print(master.text_sites)
        text_sites_label_print = ", ".join(master.text_sites[:3])
        text_sites_label_print += "..." if len(master.text_sites) > 3 else ""
        # print(master.text_sites)
        Label(self, text=txt + text_sites_label_print + " " + str(len(master.filenames))).grid(row=row, column=0)
        row += 1

        Label(self, text="Start Video Time").grid(row=row, column=0)

        # print("Create Button")
        Button(self, text="Check Start Video Time",
               command=lambda: update_start_video_time(master.video_time, master.duration, master.text_sites)).grid(
            row=row, column=1)

        row += 1

        Label(self, text="Use Excel").grid(row=row, column=0)
        master.excel = BooleanVar()
        Checkbutton(self, variable=master.excel, command=lambda: self.update_frame(master, init=False)).grid(row=row,
                                                                                                             column=1)
        master.excel.set(master.args.excel)
        row += 1

        Label(self, text="Excel Properties").grid(row=row, column=0)

        # print("Create Button")
        Button(self, text="Check Excel Properties", state=NORMAL if master.excel.get() else DISABLED,
               command=lambda: update_excel_properties(master.filenames, master.excel_time,
                                                       master.start_excel_time, master.end_excel_time,
                                                       master.text_sites, master.is_project,
                                                       changed=master.changed)).grid(
            row=row, column=1)

        row += 1

        Label(self, text="Export Interval").grid(row=row, column=0)
        master.interval_counting = StringVar()
        Entry(self, textvariable=master.interval_counting, width=2,
              state=NORMAL if master.excel.get() else DISABLED).grid(row=row, column=1)
        master.interval_counting.set(master.args.interval_counting)

        row += 1

        Label(self, text="Are the videos in the selection continuous?").grid(row=row, column=0)
        master.continuous = BooleanVar()
        if master.is_project:
            master.args.continuous = True
            state = DISABLED
        else:
            state = NORMAL
        Checkbutton(self, variable=master.continuous, state=state).grid(row=row, column=1)
        master.continuous.set(master.args.continuous)
        row += 1

        Label(self, text="Choose Draw Type").grid(row=row, column=0)

        if master.excel.get():
            draw_type_options = ["regions", "lines"]
        else:
            draw_type_options = ["regions", "lines", "None"]
        master.draw_option = StringVar()
        self.draw_option_menu = OptionMenu(self, master.draw_option, *draw_type_options,
                                           command=lambda event: self.update_frame(master, init=False))
        # if not master.excel.get():
        # self.draw_option_menu.configure(state='disabled')
        self.draw_option_menu.grid(row=row, column=1)
        master.draw_option.set(master.args.draw_option)

        row += 1

        Label(self, text=master.draw_option.get().title() + " Load").grid(row=row, column=0)
        # print(master.continuous.get())
        Button(self, text="Load " + master.draw_option.get().title() + " for videos",
               state=NORMAL if master.draw_option.get() != "None" else DISABLED,
               command=lambda: update_load_line(master.load_line, master.filenames,
                                                master.dir_depth, master.continuous.get(), master.is_project,
                                                master.object_name, master.draw_option)).grid(
            row=row, column=1)

        row += 1
        name_model_options = [name for name in os.listdir(".") if
                              name.endswith(".pth") or name.endswith(".pt")]
        # print(name_model_options)

        if master.args.net_model_name in name_model_options:

            Label(self, text="Network Model Name").grid(row=row, column=0)
            master.net_model_name = StringVar()
            OptionMenu(self, master.net_model_name, *name_model_options).grid(row=row, column=1)
            master.net_model_name.set(master.args.net_model_name)
        else:
            print(master.args.net_model_name + "is not a valid name.\nPlease specify a valid net model name")
            exit(1)

        row += 1

        device_options = ["cpu", "gpu"]
        Label(self, text="Device").grid(row=row, column=0)
        master.device = StringVar()
        OptionMenu(self, master.device, *device_options).grid(row=row, column=1)

        if master.args.device:
            if master.args.device in device_options:
                master.device.set(master.args.device)
            else:
                print(master.args.net_model_name + "is not a valid name.\nPlease specify a valid device option")
                exit(1)
        else:
            master.device = "cpu"

        row += 1

        Label(self, text="Save Result").grid(row=row, column=0)
        master.save_result = BooleanVar()
        Checkbutton(self, variable=master.save_result).grid(row=row, column=1)
        master.save_result.set(master.args.save_result)

        row += 1
        Label(self, text="Debug").grid(row=row, column=0)
        master.debug = BooleanVar()
        Checkbutton(self, variable=master.debug, command=lambda: self.update_frame(master, init=False)).grid(row=row,
                                                                                                             column=1)
        master.debug.set(master.args.debug)

        row += 1

        Label(self, text="Show").grid(row=row, column=0)
        master.show = BooleanVar()
        Checkbutton(self, variable=master.show, state=DISABLED if master.debug.get() else NORMAL).grid(row=row,
                                                                                                       column=1)
        master.show.set(master.args.show)


class InfoWindow(Tk):

    def __init__(self, video_info_text, count_info_text, i_site):
        def disable_event():
            print("Can't delete info window")
            pass

        Tk.__init__(self)
        self.protocol("WM_DELETE_WINDOW", disable_event)
        self.title(" Info ")

        self.frame_id, self.video, self.args = video_info_text

        print("PRINTING")
        print(video_info_text)
        self.movs = {k: v for k, v in count_info_text.items() if len(k) == 2}
        window_size = 30 + len(video_info_text) * 25 + 30 + 30 * len(self.movs)
        self.geometry("400x" + str(window_size))
        self.i_site = i_site
        if self.args.excel:
            self.init_seconds = convert_string_to_seconds(self.args.excel_time[self.i_site])

        self.build_info_window()

    def build_info_window(self):
        video_info_l = Label(self, text="Video Info", font=('TkDefaultFont', 15))
        video_info_text = ["Video Time:", "Real Time:", "FPS:", "Est. Time:"]
        count_info_l = Label(self, text="Counting Info", font=('TkDefaultFont', 15))
        movs_text = []
        categories_text = []
        first_time = True
        for mov in self.movs.keys():
            movs_text.append("MOV " + mov + ": ")
            if first_time:
                for category in self.movs[mov].keys():
                    categories_text.append(category)
                first_time = False
        y = 0
        video_info_l.pack()
        y += 25
        for i in range(len(video_info_text)):
            if video_info_text[i] == "Video Time:":
                info_num_video = str(self.args.filename_id + 1) + "/" + str(self.args.num_videos)
                self.time_text = StringVar()
                self.time_text.set(
                    str(self.args.video_time[self.i_site]) + " of " + str(
                        self.video.total_time) + " (" + convert_seconds_to_string(
                        convert_string_to_seconds(self.video.total_time) - convert_string_to_seconds(
                            self.args.video_time[self.i_site])) + " left)" + " " + info_num_video)
                self.time_text_label = Label(self, textvariable=self.time_text)

                self.time_text_label.place(x=100, y=y)
            elif video_info_text[i] == "Real Time:":
                self.real_time_text = StringVar()
                if self.args.excel:
                    seconds = convert_string_to_seconds(self.args.excel_time[self.i_site]) + convert_string_to_seconds(
                        self.args.video_time[self.i_site])
                else:
                    seconds = convert_string_to_seconds(self.args.video_time[self.i_site])
                real_time_str = convert_seconds_to_string(seconds)
                time_to_export_sec = (float(self.args.interval_counting) * 60) - (seconds + 1) % (
                        float(self.args.interval_counting) * 60)
                self.real_time_text.set(real_time_str + " (" + convert_seconds_to_string(time_to_export_sec,
                                                                                         minute=True) + " left to export)")
                self.real_time_text_label = Label(self, textvariable=self.real_time_text)
                self.real_time_text_label.place(x=100, y=y)
            elif video_info_text[i] == "FPS:":
                self.fps_text = StringVar()
                fps_str = str(self.video.fps_real_time) + " (" + str(self.frame_id) + "/" + str(
                    self.video.total_frames) + ") frames"
                self.fps_text.set(fps_str)
                self.fps_text_label = Label(self, textvariable=self.fps_text)
                self.fps_text_label.place(x=100, y=y)
            elif video_info_text[i] == "Est. Time:":
                self.estimated_text = StringVar()
                estimated_str = ""
                self.estimated_text.set(estimated_str)
                self.estimated_text_label = Label(self, textvariable=self.estimated_text)
                self.estimated_text_label.place(x=100, y=y)
            l = Label(self, text=video_info_text[i])
            l.place(x=0, y=y)

            y += 20
        count_info_l.place(x=130, y=y)
        y += 50
        x = 100
        first_time = True
        self.mov_counter = {}
        for mov in self.movs:
            self.mov_counter[mov] = {}
            for category in categories_text:
                if first_time:
                    l = Label(self, text=category)
                    l.place(x=x, y=y - 20)

                self.mov_x = StringVar()
                self.mov_x.set(0)
                self.mov_x_label = Label(self, textvariable=self.mov_x)
                self.mov_x_label.place(x=x + 10, y=y)
                self.mov_counter[mov][category] = self.mov_x
                x += 70

            l = Label(self, text=mov)
            l.place(x=0, y=y)
            y += 25
            x = 100
            first_time = False

    def update_info(self, info):
        # [frame_id, online_time, int(1. / self.timer.average_time), video.video_time[self.i_site](),self.n_frames_total]
        video_info, count_info = info

        frame_number, time, fps_real_time, time_video, n_frames_so_far = video_info
        info_num_video = str(self.args.filename_id + 1) + "/" + str(self.args.num_videos)
        info_estimated_time = convert_seconds_to_string(
            round(1 / fps_real_time * (self.args.n_total_frames - n_frames_so_far), 0))
        if self.args.excel:
            seconds = convert_string_to_seconds(self.args.excel_time[self.i_site]) + time
        else:
            seconds = time

        # print(time , seconds, time_video, convert_seconds_to_string(seconds))
        real_time_str = convert_seconds_to_string(seconds)

        if self.args.draw_option and self.args.excel:

            if self.args.filename_id == self.args.num_videos:
                # print(float(self.args.interval_counting) * 60 - (seconds + 1) % (float(self.args.interval_counting) * 60))
                time_to_export = " (" + convert_seconds_to_string(
                    int(min(convert_string_to_seconds(self.video.total_time) - time_video,
                            (float(self.args.interval_counting) * 60) - (seconds + 1) % (
                                    float(self.args.interval_counting) * 60))), minute=True) + " to export)"
            else:
                time_to_export = " (" + convert_seconds_to_string(
                    int((float(self.args.interval_counting) * 60) - (seconds + 1) % (
                            float(self.args.interval_counting) * 60)), minute=True) + " to export)"
        else:
            time_to_export = "(enable excel to export)"
            # print(float(self.args.interval_counting) * 60 - (seconds + 1) % (float(self.args.interval_counting) * 60))

        # print(time_to_export_sec)
        real_time_str = real_time_str + time_to_export
        fps_str = str(fps_real_time) + " (" + str(frame_number) + "/" + str(self.video.total_frames) + ") frames"
        time_str = str(convert_seconds_to_string(time_video)) + " of " + str(
            self.video.total_time) + " (" + convert_seconds_to_string(
            convert_string_to_seconds(self.video.total_time) - time_video) + " left) " + info_num_video

        self.fps_text.set(fps_str)
        self.time_text.set(time_str)
        self.real_time_text.set(real_time_str)
        self.estimated_text.set(info_estimated_time)

        mov_counter_aux = {k: v for k, v in count_info.items() if len(k) == 2}
        for mov in mov_counter_aux:

            for category in mov_counter_aux[mov]:
                self.mov_counter[mov][category].set(len(mov_counter_aux[mov][category]))
        self.update()
