import os
from datetime import datetime, timedelta


def convert_string_to_date(string, plus_day=False):
    if plus_day:
        d = datetime.strptime(string, '%H:%M:%S') + timedelta(days=1)
    else:
        d = datetime.strptime(string, '%H:%M:%S')
    return d


def convert_miliseconds_to_seconds(miliseconds):
    return int(round(miliseconds / 1000, 0))


def convert_seconds_to_miliseconds(seconds):
    return int(seconds * 1000)


def convert_date_to_seconds(date_time):
    a_timedelta = date_time - datetime(1900, 1, 1)
    seconds = a_timedelta.total_seconds()
    return round(seconds, 0)


def convert_string_to_seconds(string):
    return convert_date_to_seconds(convert_string_to_date(string))


def convert_seconds_to_date(n):
    return convert_string_to_date(convert_seconds_to_string(n))


def convert_seconds_to_string(n, minute=False):
    if os.name == "posix":
        n = n - 3600
    if minute:
        return datetime.fromtimestamp(n).strftime('%M:%S')

    return datetime.fromtimestamp(n).strftime('%H:%M:%S')


def load_excel_time(filename):


    try:
        initial_time = os.path.basename(filename).split(".")[0].split("_")[1]

        # xiaomi cameras
        if len(initial_time) == 10:
            init_time_sec = int(initial_time)
            #print("XIaomi")
            initial_time = convert_seconds_to_string(init_time_sec + 3600)
            init_time = [x for x in initial_time.split(":")]
        # orvibo cameras
        elif len(initial_time) == 6:
            init_time = [initial_time[i:i + 2] for i in range(0, len(initial_time), 2)]
        elif len(initial_time) == 4:
            init_time = [initial_time[i:i + 2] for i in range(0, len(initial_time), 2)]
            init_time.append("00")
        else:
            init_time = ["06", "00", "00"]

        return init_time
    except:
        return ["06", "00", "00"]


def search_for_valid_filenames(filenames, excel_time):
    time = datetime.strptime(excel_time, '%H:%M:%S')
    print(filenames)
    for i, f in enumerate(filenames):
        try:
            initial_time = os.path.basename(f).split(".")[0].split("_")[1]
            print(initial_time)
            if len(initial_time) == 10:
                init_time_sec = int(initial_time)
                #print("XIaomi")
                initial_time = convert_seconds_to_string(init_time_sec + 3600)
                init_time = [x for x in initial_time.split(":")]
            # orvibo cameras
            elif len(initial_time) == 6:
                init_time = [initial_time[i:i + 2] for i in range(0, len(initial_time), 2)]
            elif len(initial_time) == 4:
                init_time = [initial_time[i:i + 2] for i in range(0, len(initial_time), 2)]
                init_time.append("00")
            else:
                init_time = ["06", "00", "00"]
        except:
            print("ERROR")
            #init_time = ["06", "00", "00"]
            exit(1)

        f_time = datetime.strptime(":".join(init_time), '%H:%M:%S')
        #print(f_time, time, f_time > time)
        if f_time > time:
            #print(f_time, time)
            break

    if i == len(filenames) - 1:
        print("INVALID HOUR")
        exit(1)

    filenames = filenames[i - 1:]
    filenames.sort()
    return filenames
