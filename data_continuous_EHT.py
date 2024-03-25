from torch.utils.data import Dataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
from data_ehtim_cont import *
import torch


def load_h5_uvvis(fpath):
    print('--loading h5 file for eht sparse and dense {u,v,vis_re,vis_im} dataset...')
    with h5py.File(fpath, 'r') as F:
        u_sparse = np.array(F['u_sparse'])
        v_sparse = np.array(F['v_sparse'])
        vis_re_sparse = np.array(F['vis_re_sparse'])
        vis_im_sparse = np.array(F['vis_im_sparse'])
        u_dense = np.array(F['u_dense'])
        v_dense = np.array(F['v_dense'])
        vis_re_dense = np.array(F['vis_re_dense'])
        vis_im_dense = np.array(F['vis_im_dense'])
    print('Done--')
    return u_sparse, v_sparse, vis_re_sparse, vis_im_sparse, u_dense, v_dense, vis_re_dense, vis_im_dense


def load_h5_uvvis_cont(fpath):
    print('--loading h5 file for eht continuous {u,v,vis_re,vis_im} dataset...')
    with h5py.File(fpath, 'r') as F:
        u_cont = np.array(F['u_cont'])
        v_cont = np.array(F['v_cont'])
        vis_re_cont = np.array(F['vis_re_cont'])
        vis_im_cont = np.array(F['vis_im_cont'])
    print('Done--')
    return u_cont, v_cont, vis_re_cont, vis_im_cont


class EHTIM_Dataset(Dataset):
    '''
    EHT-imaged dataset (load precomputed)
    ''' 
    def __init__(self,  
            dset_name = 'Galaxy10', # 'MNIST'
            data_path = '../data/eht_grid_128FC_200im_Galaxy10_DECals_full.h5', 
            data_path_imgs = '../data/Galaxy10_DECals.h5', 
            data_path_cont = '../data/eht_cont_200im_Galaxy10_DECals_full.h5',
            img_res = 200,
            pre_normalize = False,
            ):

        # get spectral data
        u_sparse, v_sparse, vis_re_sparse, vis_im_sparse, u_dense, v_dense, vis_re_dense, vis_im_dense = load_h5_uvvis(data_path)
        print(u_sparse.shape, v_sparse.shape, vis_re_sparse.shape, vis_im_sparse.shape, u_dense.shape, v_dense.shape, vis_re_dense.shape, vis_im_dense.shape)

        uv_sparse = np.stack((u_sparse.flatten(), v_sparse.flatten()), axis=1)
        uv_dense = np.stack((u_dense.flatten(), v_dense.flatten()), axis=1)
        fourier_resolution = int(len(uv_dense)**(0.5))
        self.fourier_res = fourier_resolution

        # rescale uv to (-0.5, 0.5)
        max_base = np.max(uv_sparse)
        uv_dense_scaled = np.rint((uv_dense+max_base) / max_base * (fourier_resolution-1)/2) / (fourier_resolution-1) - 0.5
        self.uv_dense = uv_dense_scaled
        self.vis_re_dense = vis_re_dense
        self.vis_im_dense = vis_im_dense
        # TODO: double check un-scaling if continuous (originally scaled with sparse) 
        # should be ok bc dataset generation was scaled to max baseline, so np.max(uv_sparse)=np.max(uv_cont)
            
        # use sparse continuous data
        if data_path_cont:
            print('using sparse continuous visibility data..')
            u_cont, v_cont, vis_re_cont, vis_im_cont = load_h5_uvvis_cont(data_path_cont)
            uv_cont = np.stack((u_cont.flatten(), v_cont.flatten()), axis=1)
            uv_cont_scaled = np.rint((uv_cont+max_base) / max_base * (fourier_resolution-1)/2) / (fourier_resolution-1) - 0.5
            self.uv_sparse = uv_cont_scaled
            self.vis_re_sparse = vis_re_cont
            self.vis_im_sparse = vis_im_cont
            
        # use sparse grid data
        else:
            print('using sparse grid visibility data..')
            uv_sparse_scaled = np.rint((uv_sparse+max_base) / max_base * (fourier_resolution-1)/2) / (fourier_resolution-1) - 0.5
            self.uv_sparse = uv_sparse_scaled
            self.vis_re_sparse = vis_re_sparse
            self.vis_im_sparse = vis_im_sparse
        
        # load GT images
        self.img_res = img_res 
        
        if dset_name == 'MNIST':
            if data_path_imgs:
                from torchvision.datasets import MNIST
                from torchvision import transforms

                transform = transforms.Compose([transforms.Resize((img_res, img_res)),
                                                transforms.ToTensor(), 
                                                transforms.Normalize((0.1307,), (0.3081,)),
                                                ])
                self.img_dataset = MNIST('', train=True, download=True, transform=transform)
            else:  # if loading img data is not necessary
                self.img_dataset = None

        elif dset_name == 'Galaxy10' or 'Galaxy10_DECals':
            if data_path_imgs:
                self.img_dataset = Galaxy10_Dataset(data_path_imgs, None)
            else:  # if loading img data is not necessary
                self.img_dataset = None

        else:
            print('[ MNIST | Galaxy10 | Galaxy10_DECals ]')
            raise NotImplementedError
            
        # pre-normalize data? (disable for phase loss)
        self.pre_normalize = pre_normalize
            

    def __getitem__(self, idx):
        vis_dense = self.vis_re_dense[:,idx] + 1j*self.vis_im_dense[:,idx]
        vis_real = self.vis_re_sparse[:,idx].astype(np.float32)
        vis_imag = self.vis_im_sparse[:,idx].astype(np.float32)
        if self.pre_normalize == True:
            padding = 50 ## TODO make this actual hyperparam
            real_min, real_max= np.amin(vis_real)-padding, np.amax(vis_real)+padding
            imag_min, imag_max= np.amin(vis_imag)-padding, np.amax(vis_imag)+padding
            vis_real_normed = (vis_real - real_min) / (real_max - real_min)
            vis_imag_normed = (vis_imag - imag_min) / (imag_max - imag_min)
            vis_sparse = np.stack([vis_real_normed, vis_imag_normed], axis=1) 
        else:
            vis_sparse = np.stack([vis_real, vis_imag], axis=1) 

        if self.img_dataset:
            img, label = self.img_dataset[idx]
            img_res_initial = int(torch.numel(img)**(0.5))
            img = img.reshape((img_res_initial,img_res_initial))
            if img_res_initial != self.img_res:
                img = upscale_tensor(img, final_res=self.img_res, method='cubic')
                img = torch.from_numpy(img)
        else:
            img = torch.from_numpy(np.zeros((self.img_res,self.img_res)))
            label = None
                
        return self.uv_sparse.astype(np.float32), self.uv_dense.astype(np.float32), vis_sparse.astype(np.float32), vis_dense, img, label

    def __len__(self):
        return len(self.vis_re_sparse[0,:])


if __name__ == "__main__":
    
    fourier_resolution = 64
    dset_name = 'Galaxy10' #'MNIST'
    idx = 123
    
    data_path =f'../data/eht_grid_128FC_200im_Galaxy10_DECals_full.h5'

    #spectral_dataset = EHTIM_Dataset(data_path)
    #uv_sparse, uv_dense, vis_sparse, vis_dense = spectral_dataset[idx]
    
    im_data_path = '../data/Galaxy10_DECals.h5'
    spectral_dataset = EHTIM_Dataset(dset_name = dset_name,
                                     data_path = data_path,
                                     data_path_imgs = im_data_path,
                                     img_res = 200
                                    )
    uv_sparse, uv_dense, vis_sparse, vis_dense, img = spectral_dataset[idx]
    print(uv_sparse.shape, uv_dense.shape, vis_sparse.shape, vis_dense.shape, img.shape)
    
    # plot data
    vis_amp_sparse = np.linalg.norm(vis_sparse, axis=1)
    vis_amp_dense = np.abs(vis_dense)

    print(uv_sparse.shape)
    plt.scatter(uv_sparse[:,0], uv_sparse[:,1], c=vis_amp_sparse)
    plt.savefig('ehtim_sparse.png')
    print(uv_dense.shape)
    print(uv_dense)
    print(vis_amp_dense.shape)
    print(vis_amp_dense)
    plt.scatter(uv_dense[:,0], uv_dense[:,1], c=vis_amp_dense)
    plt.savefig('ehtim_dense.png')
    
    plt.imshow(img)
    plt.savefig('ehtim_gt_img.png')
    
#    obs_meta = spectral_dataset.get_metadata(idx, dset_name)
#    plt.imshow(obs_meta['gt_img'])
#    plt.savefig('ehtim_gt_img.png')
