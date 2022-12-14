# stream_video_generator

A minimal code example powered by [torchdata](https://github.com/pytorch/data) for runtime video frames generation.

The dataloader will download and decode videos into RGB frames or numpy arrays in asynchronous and streaming style.

Large num_workers and prefetch_workers are not recommended, since too many processing videos will be kept in memory at the moment.

**2022.12.14 Update**

Add function of **downloading videos to buffer directly**, instead of dumping videos to hard drive first.

Using this function by set to_file=False, to_buffer=True in YoutubeDownloader
