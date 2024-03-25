import os
import glob
import csv
import cv2
import numpy as np
from skimage.metrics import mean_squared_error as mse, peak_signal_noise_ratio as psnr, structural_similarity as ssim


def compute_metrics(img1, img2):
    mse_val = mse(img1, img2)
    psnr_val = psnr(img1, img2)
    ssim_val = ssim(img1, img2, multichannel=True)
    
    return mse_val, psnr_val, ssim_val



mse_list = []
psnr_list = []
ssim_list = []

for i in range(0, 5000):

    rec_img_path = '../test_res1/Galaxy10-DEC-cont/transformer/mlp_8_layer/image_loss-NF_128/'+str(i)+'/recon_image.png'
    image_img_path = '../test_res1/Galaxy10-DEC-cont/transformer/mlp_8_layer/image_loss-NF_128/'+str(i)+'/image.png'

    asamples_mean_img = cv2.imread(rec_img_path, cv2.COLOR_RGB2GRAY)
 
    image_img = cv2.imread(image_img_path, cv2.COLOR_RGB2GRAY)

    mse_val, psnr_val, ssim_val = compute_metrics(asamples_mean_img, image_img)

    mse_list.append(mse_val)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

mse_mean = np.mean(mse_list)
mse_std = np.std(mse_list)
psnr_mean = np.mean(psnr_list)
psnr_std = np.std(psnr_list)
ssim_mean = np.mean(ssim_list)
ssim_std = np.std(ssim_list)

print("PSNR", psnr_mean, psnr_std)
print("SSIM", ssim_mean, ssim_std)


def LFD(recon_freq, real_freq):

    tmp = (recon_freq - real_freq) ** 2

    freq_distance = tmp[:,0,:,:] + tmp[:,1,:,:]

    LFD = np.log(freq_distance + 1)
    return LFD



data_list_1 = []
data_list_2 = []

for i in range(0, 5000):
    folder_name = "../test_res1/Galaxy10-DEC-cont/transformer/mlp_8_layer/image_loss-NF_128/"+str(i)
    file_path = os.path.join(folder_name, "GT_vis.npy")

    data = np.load(file_path)
    data_list_1.append(data)

    folder_name = "../test_res1/Galaxy10-DEC-cont/transformer/mlp_8_layer/image_loss-NF_128/"+str(i)
    file_path = os.path.join(folder_name, "recon_vis.npy")
    data = np.load(file_path)
    data_list_2.append(data)

result_1 = np.stack(data_list_1, axis=0)
result_2 = np.stack(data_list_2, axis=0)


res = LFD(result_1, result_2)
res_vector = np.mean(res, axis=(1, 2))

mean = np.mean(res_vector)
std_dev = np.std(res_vector)

print("LFD_mean:", mean)
print("LFD_std:", std_dev)
