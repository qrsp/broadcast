import argparse
import concurrent.futures
import functools
import logging
import logging.handlers
import multiprocessing
import os
import re
import shlex
import subprocess
import traceback
from logging.handlers import QueueHandler
from shutil import which

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
from sklearn.cluster import KMeans


def handle_exceptions(f):
    @functools.wraps(f)
    def wrapper(*args, **kw):
        try:
            return f(*args, **kw)
        except Exception:
            traceback.print_exc()

    return wrapper


def logger_process(queue):
    logger = logging.getLogger(os.path.basename(__file__))

    formatter = logging.Formatter("%(levelname)s: %(message)s")
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    file_log = logging.FileHandler("{}".format("_compress.log"), "w", encoding="utf-8")
    file_log.setLevel(logging.DEBUG)
    file_log.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_log)

    while True:
        message = queue.get()
        # check for shutdown
        if message is None:
            break
        # log the message
        logger.handle(message)


def classify(data, labels):
    result = []
    # [ [], [], []... ]
    for _ in range(labels.max() + 1):
        result.append([])

    for id, label in enumerate(labels):
        result[label].append(data[id][0])

    # [ label_0[ 0db, ... ], label_1, ... ]
    return [np.array(x, dtype=float) for x in result]


def autopct_func(pct, iterator):
    return "{:.1f}%\n({:d})".format(pct, next(iterator))


def analysis(filepath, max_db, min_db):
    logger = logging.getLogger(os.path.basename(__file__))
    data_dir = "data"
    filename = re.sub(r".*/(.*)\.\w+", r"\1", filepath)

    statsfile = f"{data_dir}/{filename}.csv"

    if not os.path.isfile(statsfile):
        # Peak_level, RMS_peak
        command = f'ffprobe -v error -f lavfi -i "amovie={filepath},asetnsamples=44100,astats=metadata=1:reset=1" -show_entries frame_tags=lavfi.astats.Overall.Peak_level,lavfi.astats.Overall.RMS_peak -of csv=p=0'  # noqa: E501
        args = shlex.split(command)
        logger.debug(f"[ffmpeg] {command}")

        with open(statsfile, "w", encoding="utf-8") as f:
            process = subprocess.Popen(
                args, encoding="utf-8", stdout=f, stderr=subprocess.PIPE
            )
            _, stderr = process.communicate()
            if stderr:
                logger.debug(f"[ffmpeg] stderr: {stderr}")

    total = []
    with open(statsfile, "r", encoding="utf-8") as f:
        for line in f:
            peak, rms = line.split(",")
            peak, rms = float(peak), float(rms)
            # drop -inf and other extreme values
            if peak > min_db and peak < max_db:
                total.append([peak, rms])
        total = np.array(total, dtype=float)

    kmeans = KMeans(n_clusters=4, random_state=0).fit(total)
    labels = kmeans.labels_

    # pet chart
    result = classify(total, labels)

    # [ label, max, min, num ]
    overview = []
    for label, r in enumerate(result):
        overview.append([label, max(r), min(r), len(r)])

    overview.sort(key=lambda x: x[1], reverse=True)

    quantity = []
    db_range = []
    for id, t in enumerate(overview):
        quantity.append(t[3])
        db_range.append(f"[{t[1]:.2f}, {t[2]:.2f}]")

    iterator = iter(quantity)
    plt.pie(quantity, labels=db_range, autopct=lambda pct: autopct_func(pct, iterator))
    plt.savefig(f"{data_dir}/{filename}-pie.png", dpi=150)
    plt.clf()

    # histograms chart
    # let color is the same between with pie chart and histograms chart
    # change labels sequence: large db ~ small db
    # labels_map[old] = new
    labels_map = {}
    for id, i in enumerate(overview):
        labels_map[i[0]] = id

    new_labels = []
    for label in labels:
        new_labels.append(labels_map[label])

    new_result = classify(total, np.array(new_labels))

    plt.hist(new_result, bins="auto", stacked=True)
    plt.savefig(f"data/{filename}-histograms.png", dpi=150)
    plt.clf()

    return new_result


@handle_exceptions
def main(
    queue,
    filepath,
    output_dir,
    max_db,
    min_db,
    coverage,
    noise_reduce_db,
    goal_db,
):
    logger = logging.getLogger(os.path.basename(__file__))
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)

    filename = re.sub(r".*/(.*)\.\w+", r"\1", filepath)
    logger.info(filepath)

    result = analysis(filepath, max_db, min_db)

    # human voice
    human_voice = np.array([])
    human_voice = np.append(human_voice, result[0])
    human_voice = np.append(human_voice, result[1])
    human_voice.sort()

    whisper_db = human_voice[int(human_voice.size * (1 - coverage))]
    logger.info(f"whisper_db: {whisper_db}")

    # noise reduction
    x = result[-1].max()
    y = result[-1].mean()

    new_min_db = -1 * (((x - (y - noise_reduce_db)) * (x - min_db)) / (x - y) - x)

    logger.info(f"max noise: {x}, mean noise: {y}, new min db: {new_min_db}")

    gain = goal_db - whisper_db if whisper_db < goal_db else 0
    output = f"{output_dir}/{filename}-normalized.m4a"
    command = f'ffmpeg -i "{filepath}" -filter_complex "compand=attacks=0:points={min_db}/{new_min_db:.2f}|{x:.2f}/{x:.2f}|20/{whisper_db:.2f}:gain={gain:.2f}" -c:a aac "{output}" -y'  # noqa: E501
    logger.debug(f"[ffmpeg] {command}")
    args = shlex.split(command)

    process = subprocess.Popen(
        args, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    analysis(filepath, max_db, min_db)


def multiple_manager(filenames):
    with multiprocessing.Manager() as manager:
        queue = manager.Queue()
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as el:
            el.submit(logger_process, queue)
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as e:
                for filename in filenames:
                    e.submit(
                        main,
                        queue,
                        filename,
                        args.output,
                        args.max,
                        args.min,
                        args.coverage,
                        args.noise_reduce_db,
                        args.goal_db,
                    )

            queue.put_nowait(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress audio volume.")
    parser.add_argument("-i", "--input", type=str, nargs="?", default="input")
    parser.add_argument("-o", "--output", type=str, nargs="?", default="output")
    parser.add_argument("--max", type=int, nargs="?", default=-2)
    parser.add_argument("--min", type=int, nargs="?", default=-50)
    parser.add_argument("-c", "--coverage", type=float, nargs="?", default=1)
    parser.add_argument("-rdb", "--noise_reduce_db", type=int, nargs="?", default=5)
    parser.add_argument("-gdb", "--goal_db", type=int, nargs="?", default=-9)
    parser.add_argument("-p", "--preset", type=int, nargs="?")
    args = parser.parse_args()

    if not which("ffmpeg") or not which("ffprobe"):
        print("Can't find ffmpeg or ffprobe.")
        exit(1)

    if args.preset:
        presets = {
            # more central, but has loud background noise
            1: (1, 5),
            # less central, but has less background noise
            2: (0.75, 7),
            3: (0.65, 8),
        }
        args.coverage, args.noise_reduce_db = presets[args.preset]

    mplstyle.use(["ggplot", "fast"])

    if os.path.isdir(args.input):
        filenames = []
        with os.scandir(args.input) as it:
            for entry in it:
                if not entry.name.startswith(".") and entry.is_file():
                    filenames.append(entry.path)

        multiple_manager(filenames)
    else:
        multiple_manager([args.input])

    process = subprocess.Popen(
        ["notify", "Broadcast", "\n".join(filenames)], encoding="utf-8"
    )
