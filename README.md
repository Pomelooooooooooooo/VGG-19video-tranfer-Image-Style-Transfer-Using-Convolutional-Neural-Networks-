It is a video style transfer program based on the code provided by the paper Image Style Transfer Using Convolutional Neural Networks

The version of the library I use to run the code is:
numpy=1.26.4
torch=2.5.1 
torchvision=0.20.1
pillow=10.3.0

Command-line arguments required：
* `--content, -c`: The path to the content video.
* `--style, -s`: The path to the style image.
* `--epoch, -e`: The number of epoch. (Default: 300)
* `-content_weight, -c_w`: The weight of the content loss. (Default: 1)
* `-style_weight, -s_w`: The weight of the style loss. (Default: 1000)
* `--initialize_noise, -i_n`: If you use this option, the transferred image is initialized with white noise. If not, it is initialized with the grayscale content image.
* `--cuda`: If you have an available GPU, you should use this option.

When running, make sure that there is a folder indicated by the path in the working directory

* `frames_save_path = 'videos'
* `video_save_path = 'transed_videos'

这是自己尝试改代码的一个初步结果，代码仍有很多需要优化的地方
大二软工柚子竹
