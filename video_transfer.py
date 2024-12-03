from __future__ import print_function
from torch.autograd import Variable
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import cv2
import numpy as np
import argparse
import copy
from torchvision.transforms import ToPILImage
parser = argparse.ArgumentParser()
parser.add_argument('--content', '-c', type=str, required=True, help='The path to the Content image')
parser.add_argument('--style', '-s', type=str, required=True, help='The path to the style image')
parser.add_argument('--epoch', '-e', type=int, default=300, help='The number of epoch')
parser.add_argument('--content_weight', '-c_w', type=int, default=1, help='The weight of content loss')
parser.add_argument('--style_weight', '-s_w', type=int, default=1000, help='The weight of style loss')
parser.add_argument('--initialize_noise', '-i_n', action='store_true', help='Initialize with white noise?')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and args.cuda
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

# desired size of the output image
imsize = 512 if use_cuda else 128  # use small size if no gpu



input_size = Image.open(args.content).size



cnn = models.vgg19(pretrained=True).features

# move it to the GPU if possible:
if use_cuda:
    cnn = cnn.cuda()

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# Loss模块
class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

# Utils模块
def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()])
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image


def image_loader_gray(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()])
    image = Image.open(image_name).convert('RGB')  # Convert to RGB to maintain 3 channels
    image = loader(image).unsqueeze(0)
    return image

def save_image(tensor, size, input_size, fname, out_dir):
    unloader = transforms.ToPILImage()  # reconvert into PIL image

    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(size)
    image = unloader(image).resize(input_size)
 #确保目标目录存在
    #out_dir = 'transed_videos'
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, fname)
    image.save(out_path)

# def save_image(tensor, size, input_size, fname,out_dir):
#     unloader = transforms.ToPILImage()
#     image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
#     image = image.view(size)
#     image = unloader(image).resize(input_size)
#     # 确保目标目录存在
#     #out_dir = 'transed_videos'
#     os.makedirs(out_dir, exist_ok=True)
#
#     out_path = os.path.join(out_dir, fname)
#     image.save(out_path)
imsize = 256

style_img = image_loader(args.style, imsize).type(dtype)
content_img = image_loader_gray(args.content, imsize).type(dtype)

if args.initialize_noise:
    input_img = torch.randn(content_img.size()).type(dtype)
else:
    input_img = content_img
# content_layers_default = ['conv_4']
# style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
def get_style_model_and_losses(cnn, style_img, content_img, style_weight=1000, content_weight=1, content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300, style_weight=1000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
              #  print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.data[0], content_score.data[0]))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_param.data.clamp_(0, 1)

    return input_param.data
# 视频帧提取和风格迁移处理
def video2frame(videos_path, frames_save_path, time_interval):
    if not os.path.exists(frames_save_path):
        os.makedirs(frames_save_path)

    vidcap = cv2.VideoCapture(videos_path)
    if not vidcap.isOpened():
        print(f"无法打开视频文件：{videos_path}")
        return

    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break  # 如果没有成功读取到帧，退出循环

        count += 1
        if count % time_interval == 0:
            frame_name = os.path.join(frames_save_path, f"frame{count}.jpg")
            if image is not None and image.size > 0:
                cv2.imwrite(frame_name, image)
            else:
                print(f"无法保存空帧：{frame_name}")

    vidcap.release()
    print(f"提取 {count} 帧完成。")
    return count

def apply_style_transfer_to_frames(frames_dir, output_dir, style_img):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for frame_name in os.listdir(frames_dir):
        frame_path = os.path.join(frames_dir, frame_name)
        content_img = image_loader_gray(frame_path, imsize).type(dtype)
        if args.initialize_noise:
            input_img = Variable(torch.randn(content_img.data.size())).type(dtype)
        else:
            input_img = image_loader_gray(frame_path, imsize).type(dtype)

        input_param, optimizer = get_input_param_optimizer(input_img)
        output = run_style_transfer(cnn, content_img, style_img, input_img, args.epoch, args.style_weight, args.content_weight)
        output_name = os.path.splitext(frame_name)[0] + '.jpg'
        save_image(output, size=input_img.data.size()[1:], input_size=input_size, fname=output_name, out_dir=output_dir)

def frame2video(im_dir, video_dir, fps):
    im_list = sorted(os.listdir(im_dir), key=lambda x: int(x.split('.')[0].split('frame')[-1]))
    img = Image.open(os.path.join(im_dir, im_list[0]))
    img_size = img.size
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    for i in im_list:
        im_name = os.path.join(im_dir, i)
        frame = cv2.imread(im_name)
        videoWriter.write(frame)
    videoWriter.release()
    print('Video creation finished.')


def wait_for_frames(frames_dir, expected_count):
    while True:
        frame_count = len(os.listdir(frames_dir))
        if frame_count >= expected_count:
            break
        print(f"等待转换更多帧... 已转换 {frame_count}/{expected_count} 帧")
        time.sleep(5)  # 每5秒检查一次
    print("所有帧都已准备好，开始构建视频...")
# 主程序
# if __name__ == "__main__":
#     # 参数设置
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--content', '-c', type=str, required=True, help='The path to the Content image')
#     parser.add_argument('--style', '-s', type=str, required=True, help='The path to the style image')
#     parser.add_argument('--epoch', '-e', type=int, default=300, help='The number of epoch')
#     parser.add_argument('--content_weight', '-c_w', type=int, default=1, help='The weight of content loss')
#     parser.add_argument('--style_weight', '-s_w', type=int, default=1000, help='The weight of style loss')
#     parser.add_argument('--initialize_noise', '-i_n', action='store_true', help='Initialize with white noise?')
#     parser.add_argument('--cuda', action='store_true', help='use cuda?')
# args = parser.parse_args()
# use_cuda = torch.cuda.is_available() and args.cuda
# dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
#
# imsize = 128
# style_img = image_loader(args.style, imsize).type(dtype)
# content_img = image_loader_gray(args.content, imsize).type(dtype)
#
# if args.initialize_noise:
#     input_img = torch.randn(content_img.size()).type(dtype)
# else:
#     input_img = content_img
#
# cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
# if use_cuda:
#     cnn = cnn.cuda()


# 构建模型
# print('Building the style transfer model...')
# model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img, args.style_weight, args.content_weight)

# 1. 视频转帧
videos_path = args.content
frames_save_path = 'videos'
video_save_path = 'transed_videos'
frame_count = video2frame(videos_path, frames_save_path, 1)


# 2. 风格迁移到每一帧
apply_style_transfer_to_frames(frames_save_path, video_save_path, style_img)
wait_for_frames(video_save_path, frame_count)

# 3. 帧合成视频
frame2video(video_save_path, r'transferred/video.mp4', 30)