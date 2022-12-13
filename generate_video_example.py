import subprocess
from util.basic import map_async, load_json
import os
import numpy as np
from PIL import Image
import os.path as osp
from util.data.datapipe import *
from torchdata.datapipes.iter import IterableWrapper
from torchdata.dataloader2 import DataLoader2
from torchdata.dataloader2.reading_service import PrototypeMultiProcessingReadingService
from tqdm import tqdm
import time
from util.basic import get_logger

video_ids = load_json("./video_lg10.json")
subprocess.run("rm -rf ./videos", shell=True)
os.makedirs("./videos", exist_ok=True)


def get_video_generator(video_ids):
    # quiet = False
    quiet = True
    dataset_dp = IterableWrapper([dict(video_id=video_id) for video_id in video_ids])
    num_videos = len(dataset_dp)
    dataset_dp = prepare_for_dataloader(dataset_dp, shuffle=False)
    dataset_dp = dataset_dp.download_youtube(cache_dir="./videos",
                                             video_format="worst[height>=224][ext=mp4]",
                                             from_key="video_id",
                                             quiet=quiet)
    dataset_dp = dataset_dp.rename(["video_id.vid_path"], ["video_path"])
    dataset_dp = dataset_dp.load_frames_decord(
        from_key="video_path",
        width=224,
        height=224,
        #    stride=5,
        to_array=True).rename(["video_path.frame_arr"], ["video_array"])
    # dataset_dp = dataset_dp.load_frames_ffmpeg(from_key="video_path",
    #                                            to_images=True,
    #                                            to_array=True,
    #                                            frame_scale=224,
    #                                            fps=10,
    #                                            quiet=quiet)
    dataset_dp = dataset_dp.collect(["video_id", "video_array"])
    return dataset_dp.set_length(num_videos)


logger = get_logger(output_dir="./")

load_frame_dp = get_video_generator(video_ids)
reading_service = PrototypeMultiProcessingReadingService(num_workers=3, prefetch_worker=2)
# reading_service = None
loader = DataLoader2(load_frame_dp, reading_service=reading_service)

sum_frame = 0
video_cnt = 0

from kn_util.debug import explore_content as EC

st = time.time()
for x in loader:
    # video_id = x['video_id']
    # frame_cnt = x['video_array'].shape[0]
    logger.info("\n" + EC(x, print_str=False))
    # logger.info(f"{video_id}\t{frame_cnt}")
    # sum_frame += frame_cnt
    # video_cnt += 1

# logger.info("=============DONE===========")
# logger.info(f"eta {st - time.time()}")
# logger.info(f"avg_frame_cnt {sum_frame / video_cnt}")