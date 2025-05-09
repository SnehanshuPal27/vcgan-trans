# Test the model's second stage
import os
import argparse
import cv2
import numpy as np

def getImage(imgpath, opt):
    # Read the images
    grayimg = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    H, W = grayimg.shape
    grayimg = cv2.resize(grayimg, (opt.crop_size_w, opt.crop_size_h))
    grayimg = np.expand_dims(grayimg, 2)
    grayimg = np.concatenate((grayimg, grayimg, grayimg), axis = 2)
    # To PyTorch Tensor
    grayimg = torch.from_numpy(grayimg.transpose(2, 0, 1).astype(np.float32)).contiguous()
    grayimg = grayimg.unsqueeze(0).cuda()
    # Normalized to [-1, 1]
    grayimg = (grayimg - 128) / 128
    return grayimg, H, W

def load_model(opt):
    model = network.SecondStageNet(opt)
    pretrained_dict = torch.load(opt.load_name)
    # Get the dict from processing network
    process_dict = model.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    model.load_state_dict(process_dict)
    model = model.cuda()
    return model

def check_path(path):
    lastlen = len(path.split('/')[-1])
    path = path[:(-lastlen)]
    if not os.path.exists(path):
        os.makedirs(path)

def define_imglist(opt):
    # wholepathlist contains: base_path + class_name + image_name, while the input is base_path
    classlist = utils.text_readlines(opt.class_txt)
    wholepathlist = utils.text_readlines(opt.imagelist_txt)
    # classlist contains all class_names
    # imglist contains all class_names + image_names
    # imglist first dimension: class_names
    # imglist second dimension: image_names, for the curent class
    imglist = [list() for i in range(len(classlist))]
    for i, classname in enumerate(classlist):
        for j, imgname in enumerate(wholepathlist):
            if imgname.split('/')[-2] == classname:
                imglist[i].append(imgname)
    return imglist

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # testing parameters
    parser.add_argument('--videopath', type = str, \
        default = '/fab3/btech/2022/snehanshu.pal22b/VCGAN/davis/DAVIS/JPEGImages/480p', \
            help = 'testing video folder path')
    parser.add_argument('--savepath', type = str, \
        default = './DAVIS', \
            help = 'saving folder path')
    parser.add_argument('--class_txt', type = str, default = './txt/DAVIS_test_class.txt', help = 'DAVIS / videvo classes')
    parser.add_argument('--imagelist_txt', type = str, default = './txt/DAVIS_test_imagelist.txt', help = 'DAVIS / videvo image full paths')
    parser.add_argument('--load_name', type = str, \
        default = './models/Second_Stage_epoch10_bs1_256p.pth', \
            help = 'load the trained pth model with certain epoch')
    parser.add_argument('--pwcnet_path', type = str, default = './trained_models/pwcNet-default.pytorch', help = 'the path that contains the PWCNet model')
    parser.add_argument('--crop_size_h', type = int, default =256, help = 'single patch size')
    parser.add_argument('--crop_size_w', type = int, default = 256, help = 'single patch size')
    parser.add_argument('--comparison', type = bool, default = True, help = 'compare with original RGB image or not')
    # GPU parameters
    parser.add_argument('--test_gpu', type = str, default = '0', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    # network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'in channel for U-Net encoder')
    parser.add_argument('--start_channels', type = int, default = 32, help = 'start channel for U-Net encoder')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'out channel for U-Net decoder')
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'activation function for generator')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'activation function for discriminator')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'intialization type for generator and discriminator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the standard deviation if Gaussian normalization')
    opt = parser.parse_args()
    
    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.test_gpu
    print('Single-GPU mode, %s GPU is used' % (opt.test_gpu))
    
    # ----------------------------------------
    #                  Testing
    # ----------------------------------------
    import torch
    import network
    import pwcnet
    import utils

    # 2-dimensional list contains folder name + image name
    imglist = define_imglist(opt)
    print(imglist)
    
    # Get model
    model = load_model(opt)
    flownet = utils.create_pwcnet(opt).cuda()

    # Loop all the paths
    for i in range(len(imglist)):
        # compute the number of images in current class
        img_num = len(imglist[i])
        print('This category contains %d images' % (img_num))
        for j in range(img_num):
            print('Now it is %d-th category and %d-th frame' % (i, j))
            readpath = os.path.join(opt.videopath, imglist[i][j])
            savepath = os.path.join(opt.savepath, imglist[i][j])
            check_path(savepath)
            img, ori_H, ori_W = getImage(readpath, opt)
            # first frame
            if j == 0:
                out = model(img, img)
            # following frames
            else:
                # o_t_last_2_t range is [-20, +20]
                o_t_last_2_t = pwcnet.PWCEstimate(flownet, img, last_img)
                warp_last_out = pwcnet.PWCNetBackward((last_out + 1) / 2, o_t_last_2_t)
                warp_last_out = warp_last_out * 2 - 1
                out = model(img, warp_last_out)
            # save as last output
            last_out = out.detach()
            # save the generated frame
            out = out.squeeze(0).cpu().detach().numpy().reshape([3, opt.crop_size_h, opt.crop_size_w])
            out = out.transpose(1, 2, 0)
            out = (out * 0.5 + 0.5) * 255
            out = out.astype(np.uint8)
            out = out[:, :, ::-1]
            out = cv2.resize(out, (ori_W, ori_H))
            cv2.imwrite(savepath, out)
            # save as last output
            last_img = img
