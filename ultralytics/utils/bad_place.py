from math_operations import get_avg_point


class BasePlace(object):
    _count = 0
    start_frame = 0
    frame_id = 0

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BasePlace._count += 1
        return BasePlace._count


class BadPlace(BasePlace):
    def __init__(self, t, frame_id, length=1):
        self.center_point = list(map(t.tlbr_to_center_point, t.tracking_tlbr))
        self.length = length
        self.list = list(map(t.tlbr_to_center_point, t.tracking_tlbr))
        self.avg_point = get_avg_point(self.list)
        self.place_id = self.next_id()
        self.frame_id = frame_id
        self.start_frame = frame_id

    def update(self, t, frame_id, length=1):
        # print(self.place_id, "UPDATING")
        if len(self.list) < 3600:
            self.list.extend(list(map(t.tlbr_to_center_point, t.tracking_tlbr)))
            self.length += length
            self.frame_id = frame_id
            self.avg_point = get_avg_point(self.list)

    def __repr__(self):
        return 'Place_{}_({}):{})'.format(self.place_id, self.avg_point, self.length)
