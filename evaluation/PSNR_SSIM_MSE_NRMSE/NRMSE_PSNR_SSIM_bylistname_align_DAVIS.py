import argparse
import numpy as np
import os
import cv2
from skimage import io
from skimage import measure
from skimage import transform
from skimage import color

def resize_image(image_path, size=(64, 64)):
    """
    Resize the image to the given size.
    """
    image = io.imread(image_path)
    resized_image = transform.resize(image, size)
    return resized_image

# Compute the mean-squared error between two images
def MSE(srcpath, dstpath, gray2rgb = False, scale = True):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    if gray2rgb:
        dst = np.concatenate((dst, dst, dst), axis = 2)
    if scale:
        resize_h = scr.shape[0]
        resize_w = scr.shape[1]
        dst = cv2.resize(dst, (resize_w, resize_h))
    else:
        resize_h = dst.shape[0]
        resize_w = dst.shape[1]
        scr = cv2.resize(scr, (resize_w, resize_h))
    mse = measure.compare_mse(scr, dst)
    return mse

# Compute the normalized root mean-squared error (NRMSE) between two images
# def NRMSE(srcpath, dstpath, gray2rgb = False, scale = True, mse_type = 'Euclidean'):
#     scr = io.imread(srcpath)
#     dst = io.imread(dstpath)
#     if gray2rgb:
#         dst = np.expand_dims(dst, axis = 2)
#         dst = np.concatenate((dst, dst, dst), axis = 2)
#     if scale:
#         resize_h = scr.shape[0]
#         resize_w = scr.shape[1]
#         dst = cv2.resize(dst, (resize_w, resize_h))
#     else:
#         resize_h = dst.shape[0]
#         resize_w = dst.shape[1]
#         scr = cv2.resize(scr, (resize_w, resize_h))
#     nrmse = measure.compare_nrmse(scr, dst, norm_type = mse_type)
#     return nrmse

# added this because syntax got deprecated of compare_nrmse function
from skimage import io, metrics
import cv2
import numpy as np

def NRMSE(srcpath, dstpath, gray2rgb=False, scale=True, mse_type='Euclidean'):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    if gray2rgb:
        dst = np.expand_dims(dst, axis=2)
        dst = np.concatenate((dst, dst, dst), axis=2)
    if scale:
        resize_h = scr.shape[0]
        resize_w = scr.shape[1]
        dst = cv2.resize(dst, (resize_w, resize_h))
    else:
        resize_h = dst.shape[0]
        resize_w = dst.shape[1]
        scr = cv2.resize(scr, (resize_w, resize_h))
    nrmse = metrics.normalized_root_mse(scr, dst, normalization=mse_type)
    return nrmse

# Compute the peak signal to noise ratio (PSNR) for an image
# def PSNR(srcpath, dstpath, gray2rgb = False, scale = True):
#     scr = io.imread(srcpath)
#     dst = io.imread(dstpath)
#     if gray2rgb:
#         dst = np.expand_dims(dst, axis = 2)
#         dst = np.concatenate((dst, dst, dst), axis = 2)
#     if scale:
#         resize_h = scr.shape[0]
#         resize_w = scr.shape[1]
#         dst = cv2.resize(dst, (resize_w, resize_h))
#     else:
#         resize_h = dst.shape[0]
#         resize_w = dst.shape[1]
#         scr = cv2.resize(scr, (resize_w, resize_h))
#     psnr = measure.compare_psnr(scr, dst)
#     return psnr

from skimage import io, metrics
import cv2
import numpy as np

def PSNR(srcpath, dstpath, gray2rgb=False, scale=True):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    if gray2rgb:
        dst = np.expand_dims(dst, axis=2)
        dst = np.concatenate((dst, dst, dst), axis=2)
    if scale:
        resize_h = scr.shape[0]
        resize_w = scr.shape[1]
        dst = cv2.resize(dst, (resize_w, resize_h))
    else:
        resize_h = dst.shape[0]
        resize_w = dst.shape[1]
        scr = cv2.resize(scr, (resize_w, resize_h))
    psnr = metrics.peak_signal_noise_ratio(scr, dst)
    return psnr

# Compute the mean structural similarity index between two images
# def SSIM(srcpath, dstpath, gray2rgb = False, scale = True, RGBinput = True):
#     scr = io.imread(srcpath)
#     dst = io.imread(dstpath)
#     if gray2rgb:
#         dst = np.expand_dims(dst, axis = 2)
#         dst = np.concatenate((dst, dst, dst), axis = 2)
#     if scale:
#         resize_h = scr.shape[0]
#         resize_w = scr.shape[1]
#         dst = cv2.resize(dst, (resize_w, resize_h))
#     else:
#         resize_h = dst.shape[0]
#         resize_w = dst.shape[1]
#         scr = cv2.resize(scr, (resize_w, resize_h))
#     ssim = measure.compare_ssim(scr, dst, multichannel = RGBinput)
#     return ssim

def SSIM(srcpath, dstpath, gray2rgb=False, scale=True, RGBinput=True):
    scr = io.imread(srcpath)
    dst = io.imread(dstpath)
    print(scr.shape)
    print(dst.shape)
    if gray2rgb:
        dst = np.expand_dims(dst, axis=2)
        dst = np.concatenate((dst, dst, dst), axis=2)
    if scale:
        resize_h = scr.shape[0]
        resize_w = scr.shape[1]
        dst = cv2.resize(dst, (resize_w, resize_h))
    else:
        resize_h = dst.shape[0]
        resize_w = dst.shape[1]
        scr = cv2.resize(scr, (resize_w, resize_h))
    ssim = metrics.structural_similarity(scr, dst, multichannel=RGBinput, win_size=7, channel_axis=-1)
    return ssim

# read a txt expect EOF
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

# save a list to a txt
def text_save(content, filename, mode = 'a'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

# save current results to a conclusion file
def save_report(content, filename, mode = 'a'):
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    file.write(str(content) + '\n')
    file.close()

# Traditional indexes accuracy for dataset
def Dset_Acuuracy(imglist, refpath, basepath, gray2rgb = False, scale = True):
    # Define the list saving the accuracy
    nrmselist = []
    psnrlist = []
    ssimlist = []
    nrmseratio = 0
    psnrratio = 0
    ssimratio = 0
    print('The number of images: %d' % len(imglist))
    print('The reference path: %s' % refpath)
    print('The base path: %s' % basepath)
    # Compute the accuracy
    for i in range(len(imglist)):
        # Full imgpath
        imgname = imglist[i]# + '.JPEG'
        refimgpath = os.path.join(refpath,imgname)
        imgpath = os.path.join(basepath,imgname)
        print(refimgpath)
        print(imgpath)
        # refimgpath = os.path.join(refpath, 'DAVIS', imgname)
        # imgpath = os.path.join(basepath, 'DAVIS', imgname)
        # Compute the traditional indexes
        nrmse = NRMSE(refimgpath, imgpath, gray2rgb, scale, 'Euclidean')
        psnr = PSNR(refimgpath, imgpath, gray2rgb, scale)
        ssim = SSIM(refimgpath, imgpath, gray2rgb, scale, True)
        nrmselist.append(nrmse)
        psnrlist.append(psnr)
        ssimlist.append(ssim)
        nrmseratio = nrmseratio + nrmse
        psnrratio = psnrratio + psnr
        ssimratio = ssimratio + ssim
        print('The %dth image: nrmse: %f, psnr: %f, ssim: %f' % (i, nrmse, psnr, ssim))
    nrmseratio = nrmseratio / len(imglist)
    psnrratio = psnrratio / len(imglist)
    ssimratio = ssimratio / len(imglist)

    return nrmselist, psnrlist, ssimlist, nrmseratio, psnrratio, ssimratio
    
if __name__ == "__main__":
    
    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--imglist', type = str, default = '/fab3/btech/2022/snehanshu.pal22b/ImageNetSmall/ModifiedVCGAN/VCGAN/evaluation/PSNR_SSIM_MSE_NRMSE/DAVIS_test_imagelist_without_first_frame.txt', help = 'define imglist txt path')
    parser.add_argument('--refpath', type = str, \
        default = '/fab3/btech/2022/snehanshu.pal22b/VCGAN/davis/DAVIS/JPEGImages/480p', \
            help = 'define reference path')
    parser.add_argument('--basepath', type = str, \
        default = '/fab3/btech/2022/snehanshu.pal22b/ImageNetSmall/ModifiedVCGAN/VCGAN/train/DAVIS', \
            help = 'define imgpath')
    parser.add_argument('--gray2rgb', type = bool, default = False, help = 'whether there is an input is grayscale')
    parser.add_argument('--scale', type = tuple, default = True, help = 'whether the input needs resize')
    parser.add_argument('--savelist', type = bool, default = True, help = 'whether the results should be saved')
    parser.add_argument('--savereport', type = bool, default = True, help = 'whether the results should be saved')
    opt = parser.parse_args()
    # print(opt)

    # Read all names
    imglist = text_readlines(opt.imglist)
    print('The number of images: %d' % len(imglist))
    nrmselist, psnrlist, ssimlist, nrmseratio, psnrratio, ssimratio = Dset_Acuuracy(imglist, opt.refpath, opt.basepath, gray2rgb = opt.gray2rgb, scale = opt.scale)

    print(opt.basepath)
    print('The overall results: nrmse: %f, psnr: %f, ssim: %f' % (nrmseratio, psnrratio, ssimratio))

    # Save the files
    if opt.savelist:
        text_save(nrmselist, "./nrmselist.txt")
        text_save(psnrlist, "./psnrlist.txt")
        text_save(ssimlist, "./ssimlist.txt")

    # Save current results to a txt by adding lines
    if opt.savelist:
        content = '%s \t nrmse: %f, psnr: %f, ssim: %f' % (os.path.split(opt.basepath)[-2] + '/' + os.path.split(opt.basepath)[-1], nrmseratio, psnrratio, ssimratio)
        save_report(content, "./report.txt")
    