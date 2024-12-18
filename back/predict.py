import numpy as np
import skimage
import matplotlib.pyplot as plt
from preclassify import dicomp, hcluster
import torch
from Net import DDNet, MRC
import skimage.io
from PIL import Image

def image_padding(data,r):
    if len(data.shape)==3:
        data_new=np.lib.pad(data,((r,r),(r,r),(0,0)),'constant',constant_values=0)
        return data_new
    if len(data.shape)==2:
        data_new=np.lib.pad(data,r,'constant',constant_values=0)
        return data_new

def createTestingCubes(X, patch_size):
    # 给 X 做 padding
    margin = int((patch_size - 1) / 2)
    zeroPaddedX = image_padding(X, margin)
    patchesData = np.zeros( (X.shape[0]*X.shape[1], patch_size, patch_size, X.shape[2]) )
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchIndex = patchIndex + 1
    return patchesData

def predict(im1_path, im2_path, net_path, out_path):

    """
    获取预测结果。
    :param im1_path: 第一张图片的位置。
    :param im2_path: 第二张图片的位置。
    :param net_path: 模型的位置。
    :param out_path: 输出图片的位置。
    :return:    结果图片数组。
    """
    net = torch.load(net_path)

    #读取图像文件
    image1 = Image.open(im1_path)
    image2 = Image.open(im2_path)
    channel1 = image1.mode
    channel2 = image2.mode
    print(channel1,channel2)

    im_1_o = skimage.io.imread(im1_path)
    im_2_o = skimage.io.imread(im2_path)
    if channel1 == 'L':
        im1 = skimage.color.gray2rgb(im_1_o)[:,:,0].astype(np.float32)
    else:
        im1= im_1_o[:,:,0].astype(np.float32)
    if channel2 == 'L':
        im2 = skimage.color.gray2rgb(im_2_o)[:,:,0].astype(np.float32)
    else:
        im2 = im_2_o[:,:,0].astype(np.float32)
    # important parameter
    patch_size = 7
    # tranform image to float32
    im_di = dicomp(im1, im2)
    ylen, xlen = im_di.shape
    pix_vec = im_di.reshape([ylen*xlen, 1])

    # hiearchical FCM clustering
    # in the preclassification map, 
    # pixels with high probability to be unchanged are labeled as 1
    # pixels with high probability to be changed are labeled as 2
    # pixels with uncertainty are labeled as 1.5
    preclassify_lab = hcluster(pix_vec, im_di)
    print('... ... hiearchical clustering finished !!!')

    mdata = np.zeros([im1.shape[0], im1.shape[1], 3], dtype=np.float32)
    mdata[:,:,0] = im1
    mdata[:,:,1] = im2
    mdata[:,:,2] = im_di

    x_test = createTestingCubes(mdata, patch_size)
    x_test = x_test.transpose(0, 3, 1, 2)
    print('... x test shape: ', x_test.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 逐像素预测类别
    net.eval()
    outputs = np.zeros((ylen, xlen))
    for i in range(ylen):
        for j in range(xlen):
            if preclassify_lab[i, j] != 1.5 :
                outputs[i, j] = preclassify_lab[i, j]
            else:
                img_patch = x_test[i*xlen+j, :, :, :]
                img_patch = img_patch.reshape(1, img_patch.shape[0], img_patch.shape[1], img_patch.shape[2])
                img_patch = torch.FloatTensor(img_patch).to(device)
                prediction = net(img_patch)

                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i, j] = prediction+1
        if (i+1) % 50 == 0:
            print('... ... row', i+1, ' handling ... ...')

    plt.imsave(out_path, outputs-1, cmap='gray')
    # 输出一个表示图片的数组
    return outputs-1

