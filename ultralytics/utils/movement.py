from ultralytics.utils.math_operations import calculate_distance, sum_line_distances


class Movement(object):
    def __init__(self, l1, l2):
        self.activated = False
        self.name = l1.name("Entry") + l2.name("Exit")
        self.lines = {"Entry": l1, "Exit": l2}
        self.total_distances = []
        if hasattr(l1, "center_point"):
            self.mean_distance = calculate_distance(l1.center_point, l2.center_point)
        else:
            self.mean_distance = calculate_distance(l1.mid_point, l2.mid_point)
        self._dir = {
            "Entry": l1._dir["Entry"],
            "Exit": l2._dir["Exit"]}
        self.dir_l = {"Entry": [],
                      "Exit": []}
        self.intersection_points = {"Entry": [], "Exit": []}
        #self._avg_point = {"Entry": l1.mid_point, "Exit": l2.mid_point}
        self.stop = False
        self.tracks = []


    def update_distance(self, track):
        entry_cross_index, exit_cross_index = track.entry[1], track.exit[1]
        track_distance = sum_line_distances(track, entry_cross_index, exit_cross_index)
        self.total_distances.append(track_distance)

    """def update_avg_point(self, track):
        if not self.stop:
            if len(self.intersection_points["Entry"]) < MAX_BUFFER_MOVS:
                entry_point = track.tlbr_to_center_point(track.tracking_tlbr[track.entry[1]])
                exit_point = track.tlbr_to_center_point(track.tracking_tlbr[track.exit[1]])
                self.intersection_points["Entry"].append(entry_point)
                self.intersection_points["Exit"].append(exit_point)
                self._avg_point["Entry"] = np.average(self.intersection_points["Entry"], axis=0)
                self._avg_point["Exit"] = np.average(self.intersection_points["Exit"], axis=0)

            else:
                self.stop = True
                #print("Enough of intersection_points")"""



    """def update_dir(self, track):
        if not self.stop:
            if len(self.dir_l["Entry"]) < MAX_BUFFER_MOVS:
                entry_dir, exit_dir = track.entry[3], track.exit[3]
                self.dir_l["Entry"].append(entry_dir)
                self.dir_l["Exit"].append(exit_dir)
                self._dir["Entry"] = average_dir(self.dir_l["Entry"], range=True, r=90, only_dir=True)
                self._dir["Exit"] = average_dir(self.dir_l["Exit"], range=True, r=90, only_dir=True)
            else:
                self.stop = True
                #print("Enough of dirs movs")"""

    """def avg_point(self, mode):
        return self._avg_point[mode]

    def dir(self, mode):
        return self._dir[mode]"""


    """def mean_distance(self):
        return np.mean(self.total_distances)"""

    def update(self, track):
        if track.associated:
            print(track.track_id, "IS ASSOCIATED")
            #print("append to associated tracks")
            if not track.info_video["Start"]:
                print(track.track_id, "ERROR")
                exit(1)
            self.tracks.append(track)
            print(self.name, "+1", len(self.tracks))

        else:
            print(track.track_id, "solid")
            pass
            #self.activated = True
            # self.update_distance(track)
            # self.update_avg_point(track)
            # self.update_dir(track)






    def __repr__(self):
        return 'MOV_{}({})'.format(self.name, self.mean_distance)
