# stream_video_generator

A minimal code example powered by [torchdata](https://github.com/pytorch/data) for runtime video frames generation.

The dataloader will download and decode videos into RGB frames or numpy arrays in asynchronous and streaming style.

Large num_workers and prefetch_workers are not recommended, since too many processing videos will be kept in memory at the moment.
