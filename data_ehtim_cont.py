import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.colors as colors

import numpy as np
from numpy.fft import fft2, fftshift
from scipy import interpolate

from torchvision import transforms
from PIL import Image

import ehtim as eh
import ehtim.const_def as ehc

import h5py
import torch
from torch.utils.data import Dataset
from skimage import color

# CLEAN tests
from ehtim.imaging.clean import *
# import gICLEAN

import socket
hostname= socket.gethostname()

from scipy.signal import convolve2d

torch.manual_seed(0)
np.random.seed(0)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def preprocess_ehtim(img):
    # load the image (path or data) into eht obs format

    if (torch.is_tensor(img) or isinstance(img, np.ndarray)): # if img arr or tensor
        image = img.numpy()
    else:
        return 'img type not recognized'

    if image.ndim==3 and image.shape[-1]==3:
        image = rgb2gray( image )

    return image


def eht_createImg(image, normalize=False, pulse=ehc.PULSE_DEFAULT, obs_type='eht'):
    '''
    image - np array
    '''

    filename = './code/avery_m87_2_eofn.txt' # 200x200
    if obs_type == 'dense':     # synthetic dense array
        meta_file =f'data/eht-imaging/array_test_dense.txt' 
    elif obs_type == 'sparse':  # synthetic sparse array
        meta_file =f'data/eht-imaging/array_test_sparse.txt'
    else:                       # EHT array from CHIRP Supplement
        meta_file ='./code/EHT2017.txt' 

    assert image.shape[0]==200

    # Read the header
    file = open(filename)
    src = ' '.join(file.readline().split()[2:])
    ra = file.readline().split()
    ra = float(ra[2]) + float(ra[4]) / 60.0 + float(ra[6]) / 3600.0
    dec = file.readline().split()
    dec = np.sign(float(dec[2])) * (abs(float(dec[2])) +
                                    float(dec[4]) / 60.0 + float(dec[6]) / 3600.0)
    mjd_float = float(file.readline().split()[2])
    mjd = int(mjd_float)
    time = (mjd_float - mjd) * 24
    rf = float(file.readline().split()[2]) * 1e9
    xdim = file.readline().split()
    xdim_p = int(xdim[2])
    psize_x = float(xdim[4]) * ehc.RADPERAS / xdim_p
    ydim = file.readline().split()
    ydim_p = int(ydim[2])
    psize_y = float(ydim[4]) * ehc.RADPERAS / ydim_p
    file.close()

    if normalize:
        img = image / np.sqrt((image**2).sum())
    else:
        img = image

    # load the image
    eht_image= eh.image.Image(
            img,
            psize_x, ra, dec,
            rf=rf, source=src, mjd=mjd, time=time, pulse=pulse,
            polrep='stokes', pol_prim='I')

    # load meta
    eht_meta = eh.array.load_txt(meta_file)
    return eht_image, eht_meta


def upscale_tensor(x, final_res=256, method='nearest'):
    init_res = x.shape[0]
    xy_idx_dense = np.mgrid[:init_res,:init_res]
    x_idx_dense = xy_idx_dense[0].flatten()
    y_idx_dense = xy_idx_dense[1].flatten() 
    
    # meshgrid from 0..(final_res-1)/final_res with final_res number of entries 
    U, V = torch.meshgrid(torch.arange(final_res), torch.arange(final_res))
    U, V = U/float(final_res), V/float(final_res)
    
    # now it's a meshgrid from 0..(1-1/final_res)*init_res = init_res - init_res/final_res
    U, V = init_res*U, init_res*V 
    
    upscaled = interpolate.griddata((x_idx_dense, y_idx_dense), x.flatten(), (U, V), method=method, fill_value=-0.5) 
   
    return upscaled


def obs_with_eht(img_path, obs_type='eht', eht_npix=200):

    image = preprocess_ehtim(img_path)
    eht_im, eht_meta = eht_createImg(image, normalize=True, obs_type=obs_type)

    # Observe the image
    # tint_sec is the integration time in seconds, and tadv_sec is the advance time between scans
    # tstart_hr is the GMST time of the start of the observation and tstop_hr is the GMST time of the end
    # bw_hz is the  bandwidth in Hz
    # sgrscat=True blurs the visibilities with the Sgr A* scattering kernel for the appropriate image frequency
    # ampcal and phasecal determine if gain variations and phase errors are included
    if obs_type=='dense':
        tadv_sec = 600
    elif obs_type=='sparse':
        tadv_sec = 6000
    else: # default use EHT
        tadv_sec = 600
    tstart_hr = 0
    tstop_hr = 24
    tint_sec = 12
    bw_hz = 4.096e9
    eht_obs = eht_im.observe(eht_meta, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                             sgrscat=False, add_th_noise=False, ampcal=True, phasecal=True, ttype='direct')
    
    # FOV used in CHIRP (approx angular size of M87 SMBH) [200x200]
    eht_fov = np.radians(.000291/3600)  

    # Resolution
    eht_res = eht_obs.res() # nominal array resolution, 1/longest baseline
    print("Nominal Resolution: " , eht_res)
    print("FoV: " , eht_fov)
    
    return eht_obs, eht_im, eht_res, eht_fov, eht_npix

def make_im_torch(
    uv_arr, vis_arr, npix, fov, pulse=ehc.PULSE_DEFAULT, weighting='uniform', norm_fact=None, return_im=False, seperable_FFT=True, rescaled_pix=True):
    """Make the observation image using direct Fourier transform. 
    Assume the visibilities are on regulars grid in the continuous domain

        Args:
            uv_arr- U x 2 (U==V)
            vis_arr- B x U x V
            npix (int): The pixel size of the square output image.
            fov (float): The field of view of the square output image in radians.
            pulse (function): The function convolved with the pixel values for continuous image.
            weighting (str): 'uniform' or 'natural'
        Returns:
            (Image): an Image object with dirty image.
    """
    import math

    if rescaled_pix:
        pdim = 1. #scaled input
    else:
        pdim = fov / npix

    u = uv_arr[:,0]
    v = uv_arr[:,1]

    B, U, V= vis_arr.shape[0], vis_arr.shape[1], vis_arr.shape[2]
    assert U==V

    #TODO: xlist as input to speed up
    #DONE: calculate the scale of u*x and v*x directly
    #DONE: scaled by normfac
    xlist = torch.arange(0, -npix, -1, device=uv_arr.device) * pdim + (pdim * npix) / 2.0 - pdim / 2.0


    # #--Sequence 1D Inverse DFT--#
    if seperable_FFT:
        X_coord= xlist.reshape(1, npix, 1, 1, 1)
        Y_coord= xlist.reshape(1, 1, npix, 1, 1)
        U_coord= u.reshape(1,1,1, U,1)
        V_coord= v.reshape(1,1,1, 1,V)
        Vis= vis_arr.reshape(B, 1, 1, U, V)
        #the inner integration (over u) 
        U_X= U_coord*X_coord
        
        # temp_a = Vis * torch.exp(-2.j* math.pi* U_X)
        # inner_integral= torch.sum(temp_a , dim=-2,keepdim=True)/temp_a.size(-2) #B X 1 1 V
        

        inner_integral= torch.mean(Vis * torch.exp(-2.j* math.pi* U_X) , dim=-2,keepdim=True) #B X 1 1 V
        #the outer integration (over v) 
        V_Y= V_coord*Y_coord

        # temp_b=inner_integral * torch.exp(-2.j*math.pi* V_Y)
        # outer_integral= torch.sum(temp_b, dim=-1, keepdim=True )/temp_b.size(-1) # B X Y 1 1
        outer_integral= torch.mean(inner_integral * torch.exp(-2.j*math.pi* V_Y), dim=-1, keepdim=True ) # B X Y 1 1
        image_complex= outer_integral.squeeze(-1).squeeze(-1) # B X Y
    else:
        #--2D raw version IDFT--#
        X_coord= xlist.reshape(1, npix, 1, 1, 1).expand(B,npix,npix, U,V)
        Y_coord= xlist.reshape(1, 1, npix, 1, 1).expand_as(X_coord)
        U_coord= u.reshape(1,1,1, U,1).expand_as(X_coord)
        V_coord= v.reshape(1,1,1, 1,V).expand_as(X_coord)
        U_X= U_coord*X_coord
        V_Y= V_coord*Y_coord
        Vis= vis_arr.reshape(B, 1, 1, U, V).expand_as(X_coord)
        temp_c = Vis * torch.exp(-2.j*math.pi*(U_X + V_Y))
        image_complex= torch.mean(temp_c, dim=-1).mean(dim=-1)
        # temp_d = torch.sum(temp_c, dim=-1)/temp_c.size(-1)
        # image_complex = torch.sum(temp_d)/temp_d.size(-1)


    if norm_fact is not None:
        image_complex= image_complex* norm_fact

    
    # import pdb; pdb.set_trace()
    return image_complex


def make_dirtyim(uv_arr, vis_arr, npix, fov, pulse=ehc.PULSE_DEFAULT, weighting='uniform', return_im=False, cutoff_freq=0.03, sigma=1.0):
    """Make the observation dirty image (direct Fourier transform).

        Args:
            
            npix (int): The pixel size of the square output image.
            fov (float): The field of view of the square output image in radians.
            pulse (function): The function convolved with the pixel values for continuous image.
            weighting (str): 'uniform' or 'natural'
        Returns:
            (Image): an Image object with dirty image.
    """

    pdim = fov / npix
    u = uv_arr[:,0]
    v = uv_arr[:,1]

    xlist = np.arange(0, -npix, -1) * pdim + (pdim * npix) / 2.0 - pdim / 2.0
    if weighting == 'natural':
        sigma = np.atleast_2d(sigma)
        print(u.shape, sigma.shape); input()
        weights = 1. / (sigma*sigma)
    else:
        weights = np.ones(u.shape)

    dim= np.array([[np.mean(weights * np.cos(-2 * np.pi * (i * u + j * v)))
                     for i in xlist]
                     for j in xlist])
    normfac= 1. / np.sum(dim)

    vis = vis_arr

    # TODO -- use NFFT
    # TODO -- different beam weightings
    im = np.array([[np.mean(weights *  (np.real(vis) * np.cos(-2 * np.pi * (i * u + j * v)) -
                                        np.imag(vis) * np.sin(-2 * np.pi * (i * u + j * v))))
                    for i in xlist]
                    for j in xlist])

    # Final normalization
    im = im * normfac
    im = im[0:npix, 0:npix]

    do_sinc = False
    if do_sinc:

        fc = cutoff_freq  # Cutoff frequency as a fraction of the sampling rate (in (0, 0.5)).
        b = 2.0*fc/3.0  # Transition band, as a fraction of the sampling rate (in (0, 0.5)).
        N = int(np.ceil((4 / b)))
        if not N % 2: N += 1  # Make sure that N is odd.
        crop = int(N / 2)
        n = np.arange(N)

        # Compute sinc filter.
        h = np.sinc(2 * fc * (n - (N - 1) / 2))

        # Compute Blackman window.
        #w = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + \
        #    0.08 * np.cos(4 * np.pi * n / (N - 1))
        w = np.blackman(N)

        # Multiply sinc filter by window.
        h_windowed = h * w

        # Normalize to get unity gain.
        h_windowed = h_windowed / np.sum(h_windowed)

        do_plot = False
        if do_plot:
            import pylab as plt
            fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=True)

            axs[0, 0].plot(h)
            axs[0, 0].set_title("sinc filter")
            axs[0, 1].plot(w)
            axs[0, 1].set_title("blackman window")
            axs[0, 2].plot(h_windowed)
            axs[0, 2].set_title("windowed sinc")
            axs[1, 0].plot(np.fft.fftshift(np.fft.fft(h)), 'o')
            axs[1, 0].set_title("sinc filter")
            axs[1, 1].plot(np.fft.fftshift(np.fft.fft(w)), 'o')
            axs[1, 1].set_title("blackman window")
            axs[1, 2].plot(np.fft.fftshift(np.fft.fft(h_windowed)), 'o')
            axs[1, 2].set_title("windowed sinc")
            plt.show()

        im_shape = im.shape
        im_x = np.stack([np.convolve(im[i,:], h) for i in range(im.shape[0])])

        im_xy = np.stack([np.convolve(im_x[:,i], h) for i in range(im_x.shape[1])])

        im = im_xy[crop:im_shape[0]+crop, crop:im_shape[1]+crop].T

        print(im.shape, N, crop, im_xy.shape, im_x.shape, w.shape, h.shape);

    out = eh.image.Image(im, pdim, 10, 20, pulse=pulse) # filler RA/Dec values
    #out = ehtim.image.Image(im, pdim, self.ra, self.dec, polrep=self.polrep,
    #                        rf=self.rf, source=self.source, mjd=self.mjd, pulse=pulse)
    if not return_im:
        return out

    else:
        return out, im, normfac


def get_uvvis_data(img_path, obs_type='eht', eht_npix=200, num_fourier_coeff=64):
    """ obs an image with ehtim, return {u,v,vis} for grid dense, continuous sparse, grid sparse data
    """
    # data dicts
    grid_dense = {}
    cont_sparse = {}
    grid_sparse = {}
    obs_meta = {}

    # eht-im observation (continuous sparse)
    eht_obs, eht_im, eht_res, eht_fov, eht_npix = obs_with_eht(img_path, obs_type=obs_type, eht_npix=eht_npix)
    u_eht = np.array(eht_obs.unpack(['u'], conj=True)).astype(np.float)
    v_eht = np.array(eht_obs.unpack(['v'], conj=True)).astype(np.float)
    vis_eht = np.array(eht_obs.unpack(['vis'], conj=True)).astype(np.complex)
    uv_dist_eht = np.array(eht_obs.unpack(['uvdist'], conj=True)).astype(np.float)

    # dataset: ground truth (scaled to eht_npix)
    obs_meta['gt_img'] = eht_im.imarr()
    obs_meta['res'] = eht_res
    obs_meta['fov'] = eht_fov
    obs_meta['npix'] = eht_npix
    obs_meta['n_FC'] = num_fourier_coeff
    obs_meta['sigma'] = eht_obs.unpack(['sigma'])

    # dataset: continuous sparse
    cont_sparse['uv'] = np.stack((u_eht, v_eht), axis=1)
    cont_sparse['vis'] = vis_eht
    #cont_sparse['dim'] = make_dirtyim(cont_sparse['uv'], cont_sparse['vis'], eht_npix, eht_fov)

    # dataset: grid dense
    '''max_base = np.max(uv_dist_eht)
    x = np.linspace(-max_base, max_base, num_fourier_coeff)
    y = np.linspace(-max_base, max_base, num_fourier_coeff)
    xv, yv = np.meshgrid(x, y)
    grid_dense['uv'] = np.stack((xv.ravel(), yv.ravel()), axis=1)
    grid_dense['vis'] = eht_im.sample_uv(grid_dense['uv'])[0]  # ignore polarizations
    #grid_dense['dim'] = make_dirtyim(grid_dense['uv'], grid_dense['vis'], eht_npix, eht_fov)

    # dataset: grid sparse
    x_centers = (x[1:]+x[:-1])/2
    y_centers = (y[1:]+y[:-1])/2
    u_dig = np.digitize(u_eht, x_centers)
    v_dig = np.digitize(v_eht, y_centers)
    uv_dig = np.stack((x[u_dig], y[v_dig]), axis=1)
    grid_sparse['uv'] = np.unique(uv_dig , axis=0) # remove duplicates
    grid_sparse['vis'] = eht_im.sample_uv(grid_sparse['uv'])[0]'''
    #grid_sparse['dim'] = make_dirtyim(grid_sparse['uv'], grid_sparse['vis'], eht_npix, eht_fov)

    #return grid_dense, cont_sparse, grid_sparse, obs_meta
    return None, cont_sparse, None, obs_meta


def plot_eht_compare(grid_dense, cont_sparse, grid_sparse, obs_meta, savefig=False, cutoff_freq=0.03):
    """ 3-row plot for eht_npix resolution obs """

    # make dirty images:
    eht_npix, eht_fov, num_fourier_coeff, sigma = obs_meta['npix'], obs_meta['fov'], obs_meta['n_FC'], obs_meta['sigma']
    dim_grid_dense, im1, norm_grid_dense = make_dirtyim(grid_dense['uv'], grid_dense['vis'], eht_npix, eht_fov, sigma=sigma, cutoff_freq=cutoff_freq, return_im=True)
    dim_cont_sparse, im2, norm_cont_sparse = make_dirtyim(cont_sparse['uv'], cont_sparse['vis'], eht_npix, eht_fov, sigma=sigma, cutoff_freq=cutoff_freq, return_im=True)
    dim_grid_sparse, im3, norm_grid_sparse = make_dirtyim(grid_sparse['uv'], grid_sparse['vis'], eht_npix, eht_fov, sigma=sigma, cutoff_freq=cutoff_freq, return_im=True)

    dirty_beam, im4, norm_dirty_beam = make_dirtyim(cont_sparse['uv'], np.ones_like(cont_sparse['vis']), eht_npix, eht_fov, sigma=sigma, cutoff_freq=cutoff_freq, return_im=True)

    dim_grid_dense = dim_grid_dense.imarr()
    dim_cont_sparse = dim_cont_sparse.imarr()
    dim_grid_sparse = dim_grid_sparse.imarr()
    dirty_beam = dirty_beam.imarr()

    '''import pylab as plt
    fig, ax = plt.subplots(nrows=4, ncols=2)
    ax[0,0].imshow(dim_grid_dense)
    ax[0,1].imshow(im1)
    ax[1,0].imshow(dim_cont_sparse)
    ax[1,1].imshow(im2)
    ax[2,0].imshow(dim_grid_sparse)
    ax[2,1].imshow(im3)
    ax[3,0].imshow(dirty_beam)
    ax[3,1].imshow(im4)
    print(norm1, norm2, norm3, norm4)
    print(dim_grid_dense.max(), im1.max())
    print(dim_cont_sparse.max(), im2.max())
    print(dim_grid_sparse.max(), im3.max())
    print(dirty_beam.max(), im4.max())
    print('---')
    print(dim_grid_dense.min(), im1.min())
    print(dim_cont_sparse.min(), im2.min())
    print(dim_grid_sparse.min(), im3.min())
    print(dirty_beam.min(), im4.min())
    plt.show()'''

    '''import pylab as plt
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(dim_cont_sparse)
    ax[1].imshow(dirty_beam)
    plt.show()'''

    # import gICLEAN
    # gICLEAN.clean_cuda(dirty_im=dim_cont_sparse/norm_cont_sparse, dirty_psf=dirty_beam/norm_dirty_beam, thresh=0.001, gain=1.0, clean_beam_size=4.0,
    #                    maxIter=1e6,
    #                    prefix='test4',
    #                    im_gt=dim_grid_dense/norm_grid_dense,
    #                    polarity=False)
    # input("done!")

    # plot properties
    vmin, vmax = 1e-4, 1e2  # 1e-2, 1e3  # fft color range
    #vmin_img, vmax_img = 1.5*np.min(img), 1.5*np.max(img) 
    uv_dist_eht = np.linalg.norm(cont_sparse['uv'], axis=1)
    max_base = np.max(uv_dist_eht)

    # make figure
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = gridspec.GridSpec(3,4, hspace=0.3, wspace=0.25)

    # grid_dense
    ax = plt.subplot(gs[0,0])
    ax.set_title("(%s x %s Dense grid) Dirty Image\n$I^{D}_{grid}(l,m) \equiv \mathscr{F}^{-1}_{NU}[\hat{\mathcal{V}}_{EHT}(u,v)]$" % (num_fourier_coeff, num_fourier_coeff), fontsize=10)
    ax.imshow(dim_grid_dense) #, vmin=vmin_img, vmax=vmax_img)

    ax = plt.subplot(gs[0,1])
    ax.set_title("Visibity Phase\n$ \\angle{\mathcal{V}(u,v)}$", fontsize=10)
    ax.scatter(grid_dense['uv'][:,0], grid_dense['uv'][:,1], c=np.angle(grid_dense['vis']), 
               s=1, cmap='twilight', vmin=-np.pi, vmax=np.pi, rasterized=True)
    ax.set_xlim([-1.1*max_base, 1.1*max_base])
    ax.set_ylim([-1.1*max_base, 1.1*max_base])

    ax = plt.subplot(gs[0,2])
    ax.set_title("Visibility Amplitude\n$|\mathcal{V}(u,v)|$", fontsize=10)
    ax.scatter(grid_dense['uv'][:,0], grid_dense['uv'][:,1], c=np.abs(grid_dense['vis']), 
               s=1, cmap='viridis', vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlim([-1.1*max_base, 1.1*max_base])
    ax.set_ylim([-1.1*max_base, 1.1*max_base])

    ax = plt.subplot(gs[0,3])
    ax.set_title("Visibility Amplitude \n vs. UV distance", fontsize=10)
    ax.scatter(np.linalg.norm(grid_dense['uv'], axis=1), np.abs(grid_dense['vis']), c=np.abs(grid_dense['vis']), 
               s=1, marker='.', vmin=vmin, vmax=vmax, rasterized=True)
    ax.text(0.03, 0.97, f"n={len(grid_dense['uv'])}", fontsize=8, ha='left', va='top', transform=ax.transAxes)
    ax.set_yscale('log')
    ax.set_xlim([0,1.25e10])
    ax.set_ylim([1e-1,3000])

    # cont_sparse
    ax = plt.subplot(gs[1,0])
    ax.set_title("(EHT) Dirty Image\n$I^{D}_{EHT}(l,m) \equiv \mathscr{F}^{-1}_{NU}[\hat{\mathcal{V}}_{EHT}(u,v)]$", fontsize=10)
    ax.imshow(dim_cont_sparse) #, vmin=vmin_img, vmax=vmax_img)

    ax = plt.subplot(gs[1,1])
    ax.set_title("(EHT) Observed Visib. Phase\n$ \\angle{\hat{\mathcal{V}}_{EHT}(u,v)}$", fontsize=10)
    ax.scatter(grid_dense['uv'][:,0], grid_dense['uv'][:,1], c='0.5', alpha=0.7, s=0.1, marker='.', rasterized=True)
    ax.scatter(cont_sparse['uv'][:,0], cont_sparse['uv'][:,1], c=np.angle(cont_sparse['vis']), 
               s=1, marker='.', cmap='twilight', vmin=-np.pi, vmax=np.pi, rasterized=True)
    ax.set_xlim([-1.1*max_base, 1.1*max_base])
    ax.set_ylim([-1.1*max_base, 1.1*max_base])

    ax = plt.subplot(gs[1,2])
    ax.set_title("(EHT) Observed Visib. Amp\n$|\hat{\mathcal{V}}_{EHT}(u,v)|$", fontsize=10)
    ax.scatter(grid_dense['uv'][:,0], grid_dense['uv'][:,1], c='0.5', alpha=0.7, s=0.1, marker='.', rasterized=True)
    ax.scatter(cont_sparse['uv'][:,0], cont_sparse['uv'][:,1], c=np.abs(cont_sparse['vis']), 
               s=1, marker='.', cmap='viridis', vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlim([-1.1*max_base, 1.1*max_base])
    ax.set_ylim([-1.1*max_base, 1.1*max_base])

    ax = plt.subplot(gs[1,3])
    ax.set_title("(EHT) Visib. Amp. \n vs. UV distance", fontsize=10)
    ax.scatter(np.linalg.norm(grid_dense['uv'], axis=1), np.abs(grid_dense['vis']), c='0.5', alpha=0.7, s=0.1, marker='.', rasterized=True)
    ax.scatter(np.linalg.norm(cont_sparse['uv'], axis=1), np.abs(cont_sparse['vis']), c=np.abs(cont_sparse['vis']), 
               s=1, vmin=vmin, vmax=vmax, rasterized=True)
    ax.text(0.03, 0.97, f"n={len(cont_sparse['uv'])}", fontsize=8, ha='left', va='top', transform=ax.transAxes)
    ax.set_yscale('log')
    ax.set_xlim([0,1.25e10])
    ax.set_ylim([1e-1,3000])

    # grid_sparse
    ax = plt.subplot(gs[2,0])
    ax.set_title("(EHT grid) Dirty Image\n$I^{D}_{EHT,grid}(l,m) \equiv \mathscr{F}^{-1}_{NU}[\hat{\mathcal{V}}_{EHT,grid}(u,v)]$", fontsize=10)
    ax.imshow(dim_grid_sparse) #, vmin=vmin_img, vmax=vmax_img)

    ax = plt.subplot(gs[2,1])
    ax.set_title("(EHT,grid) Visib. Phase\n$ \\angle{\hat{\mathcal{V}}_{EHT,grid}(u,v)}$", fontsize=10)
    ax.scatter(grid_dense['uv'][:,0], grid_dense['uv'][:,1], alpha=0.7, s=0.1, c='0.5', marker='.', rasterized=True)
    ax.scatter(grid_sparse['uv'][:,0], grid_sparse['uv'][:,1], c=np.angle(grid_sparse['vis']), 
               s=1, marker='.', cmap='twilight', vmin=-np.pi, vmax=np.pi, rasterized=True)
    ax.set_xlim([-1.1*max_base, 1.1*max_base])
    ax.set_ylim([-1.1*max_base, 1.1*max_base])

    ax = plt.subplot(gs[2,2])
    ax.set_title("(EHT,grid) Visib. Amp\n$|\hat{\mathcal{V}}_{EHT,grid}(u,v)|$", fontsize=10)
    ax.scatter(grid_dense['uv'][:,0], grid_dense['uv'][:,1], alpha=0.7, s=0.1, c='0.5', marker='.', rasterized=True)
    ax.scatter(grid_sparse['uv'][:,0], grid_sparse['uv'][:,1], c=np.abs(grid_sparse['vis']), 
               s=1, marker='.', cmap='viridis', vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_xlim([-1.1*max_base, 1.1*max_base])
    ax.set_ylim([-1.1*max_base, 1.1*max_base])

    ax = plt.subplot(gs[2,3])
    ax.set_title("(EHT grid) Visib. Amp. \n vs. UV distance", fontsize=10)
    ax.scatter(np.linalg.norm(grid_dense['uv'], axis=1), np.abs(grid_dense['vis']), c='0.5', alpha=0.7, s=0.1, marker='.', rasterized=True)
    ax.scatter(np.linalg.norm(grid_sparse['uv'], axis=1), np.abs(grid_sparse['vis']), c=np.abs(grid_sparse['vis']), 
               s=1, vmin=vmin, vmax=vmax, rasterized=True)
    ax.text(0.03, 0.97, f"n={len(grid_sparse['uv'])}", fontsize=8, ha='left', va='top', transform=ax.transAxes)
    ax.set_yscale('log')
    ax.set_xlim([0,1.25e10])
    ax.set_ylim([1e-1,3000])

    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
        plt.close()


def load_h5(fpath):
    print('--loading h5 file for Galaxy10 dataset...')
    with h5py.File(fpath, 'r') as F:
        x = np.array(F['images'])
        y = np.array(F['ans'])
    print('Done--')

    return x, y


class Galaxy10_Dataset(Dataset):
    '''
    loader for Galaxy10 version_1, lower resolution
    ''' 
    def __init__(self,  h5_path ='./dataset_ssd/astroImg/Galaxy10.h5', transform_in = None):
        if transform_in is None:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            transform = transform_in

        imgs, labels= load_h5(h5_path)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        scale = 1/255.
        img_Lab = color.rgb2lab(self.imgs[idx])
        img = self.transform(img_Lab[...,0] * scale)
        #tf2 = transforms.Compose([transforms.ToPILImage()])
        #img_Lab = tf2(color.rgb2lab(self.imgs[idx]))
        #img = self.transform(img_Lab[...,0])
        #img *= scale
        label = self.labels[idx]
        return img, label

    def __len__(self):
        #return len(img)
        return len(self.imgs)

class Galaxy10_DECals_Dataset(Dataset):
    '''
    loader for Galaxy10 DECals (version 2), 256x256 resolution
    '''
    def __init__(self,  h5_path ='/astroim/data/Galaxy10_DECals.h5', transform_in = None):
        if transform_in is None:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        else:
            transform = transform_in

        imgs, labels= load_h5(h5_path)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        scale = 1/255.
        img_Lab = color.rgb2lab(self.imgs[idx])
        img = self.transform(img_Lab[...,0] * scale)
        #tf2 = transforms.Compose([transforms.ToPILImage()])
        #img_Lab = tf2(color.rgb2lab(self.imgs[idx]))
        #img = self.transform(img_Lab[...,0])
        #img *= scale
        label = self.labels[idx]
        return img, label

    def __len__(self):
        #return len(img)
        return len(self.imgs)

class EHT_Continuous_Dataset(Dataset):
    '''
    dataset for EHT imaging of MNIST or Galaxy10: 
    returns {u,v,vis} for dense grid, sparse continuous, sparse grid

    dset_name = ['MNIST', 'Galaxy10']
    obs_type = ['eht', 'sparse', 'dense'] # note: sparse/dense replace EHT array with artificial telescope array
    ''' 

    def __init__(self, 
            eht_npix = 200,
            num_FC = 64, 
            dset_name = 'Galaxy10',
            h5_path_img = '../data/Galaxy10.h5', 
            transform_in = None,
            obs_type='eht'):

        if dset_name == 'MNIST':
            from torchvision.datasets import MNIST
            from torchvision import transforms

            transform = transforms.Compose([transforms.Resize((200, 200)),
                                            transforms.ToTensor(), 
                                            transforms.Normalize((0.1307,), (0.3081,)),
                                            ])
            self.dataset = MNIST('', train=True, download=True, transform=transform)

        elif dset_name == 'Galaxy10':
            h5_path_img = '../data/Galaxy10.h5'
            self.dataset = Galaxy10_Dataset(h5_path_img, transform_in)

        elif dset_name == 'Galaxy10_DECals':
            h5_path_img = '../data/Galaxy10_DECals.h5'
            self.dataset = Galaxy10_DECals_Dataset(h5_path_img, transform_in)

        else:
            print("choose dset_name from ['MNIST', 'Galaxy10', 'Galaxy10_DECals']")
            raise NotImplementedError
        

        self.eht_npix = eht_npix
        self.num_FC = num_FC
        self.obs_type = obs_type

    def __getitem__(self, idx):
        
        # rescale to 200x200 for eht-im setup
        img_res_initial = int(torch.numel(self.dataset[idx][0])**(0.5))
        img = self.dataset[idx][0].reshape((img_res_initial,img_res_initial))

        if img_res_initial != 200:
            #print('scaling input to match requested size:', img_res_initial, 200)
            img = upscale_tensor(img, final_res=200, method='cubic')
            img = torch.from_numpy(img)

        grid_dense, cont_sparse, grid_sparse, obs_meta = get_uvvis_data(img, obs_type=self.obs_type, eht_npix=self.eht_npix, num_fourier_coeff=self.num_FC)

        #--DEBUG replace the vis member of grid_dense with the one genrated by DPI helper
        # if hostname=='NV':
        #     import dpi_helper
        #     vis_grid_dense = dpi_helper.get_uvvis_data_dpi(
        #         img.reshape(1, img.shape[-2], img.shape[-1]).repeat(2, 1, 1),
        #         uvfit_filepath='../data/gt.fits',
        #         obs_path='../data/obs.uvfits',
        #         fov=  obs_meta['fov'],
        #         pdim= obs_meta['fov']/ img.shape[-1],
        #         npix= img.shape[-1],
        #         num_fourier_coeff=self.num_FC,
        #         uv_input= grid_dense['uv'])
        #     vis_grid_dense= torch.view_as_complex(vis_grid_dense[0].T.contiguous()).cpu().numpy()
        #     grid_dense['vis']= vis_grid_dense
        #---- END OF DEBUG ---#


        return grid_dense, cont_sparse, grid_sparse, obs_meta

    def __len__(self):
        return len(self.dataset)


def plot_compare_dirtyim_ehtobs(grid_dense, cont_sparse, grid_sparse, obs_meta, gt_image):
    # make dirty images:
    cutoff_freq = 0.0
    weighting = 'uniform'
    eht_npix, eht_fov, num_fourier_coeff, sigma = obs_meta['npix'], obs_meta['fov'], obs_meta['n_FC'], obs_meta['sigma']
    dim_grid_dense, im1, norm_grid_dense = make_dirtyim(grid_dense['uv'], grid_dense['vis'], eht_npix, eht_fov, sigma=sigma,
                                                        cutoff_freq=cutoff_freq, return_im=True,
                                                        weighting=weighting)
    dim_cont_sparse, im2, norm_cont_sparse = make_dirtyim(cont_sparse['uv'], cont_sparse['vis'], eht_npix, eht_fov, sigma=sigma,
                                                          cutoff_freq=cutoff_freq, return_im=True,
                                                          weighting=weighting)
    dim_grid_sparse, im3, norm_grid_sparse = make_dirtyim(grid_sparse['uv'], grid_sparse['vis'], eht_npix, eht_fov, sigma=sigma,
                                                          cutoff_freq=cutoff_freq, return_im=True,
                                                          weighting=weighting)

    dirty_beam, im4, norm_dirty_beam = make_dirtyim(cont_sparse['uv'], 10.0*np.ones_like(cont_sparse['vis']), eht_npix, eht_fov, sigma=sigma,
                                                    cutoff_freq=cutoff_freq, return_im=True,
                                                    weighting=weighting)

    dim_grid_dense = dim_grid_dense.imarr()
    dim_cont_sparse = dim_cont_sparse.imarr()
    dim_grid_sparse = dim_grid_sparse.imarr()
    dirty_beam = dirty_beam.imarr()


    fov=1.4108078120287498e-09
    npix=len(gt_image)
    pdim = fov/npix
    im = eh.image.Image(gt_image, pdim, 0, 0,)
    # fov2 = im.xdim * im.psize  # same as fov
    #im.display()

    # observe the image the same way as data generator
    meta_file ='./code/EHT2017.txt'
    eht_meta = eh.array.load_txt(meta_file)

    tadv_sec = 600
    tstart_hr = 0
    tstop_hr = 24
    tint_sec = 12
    bw_hz = 4.096e9
    obs = im.observe(eht_meta, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                                sgrscat=False, add_th_noise=False, ampcal=True, phasecal=True, ttype='direct')

    #fov_expanded = fov * 1.1

    # Resolution
    beamparams = obs.fit_beam()  # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
    res = obs.res()  # nominal array resolution, 1/longest baseline
    print("Clean beam parameters: ", beamparams)
    print("Nominal Resolution: ", res)

    #obs.save_uvfits('galaxy10_decals_obs.fits') # exports a UVFITS file modeled on template.UVP
    #obs.save_fits('galaxy10_decals_obs.fits')
    #print('saved file!')

    dim = obs.dirtyimage(npix, fov).imarr()
    dbeam = obs.dirtybeam(npix, fov).imarr()
    cbeam = obs.cleanbeam(npix, fov).imarr()

    clean_beam_size = 4.0
    imsize = np.int32(dirty_beam.shape[0])
    dirty_psf_max = np.float32(dirty_beam.max())
    dirty_psf = dirty_beam / dirty_psf_max
    # clean_psf = gICLEAN.serial_clean_beam(dirty_beam, imsize / clean_beam_size)*dirty_psf_max

    cmap = 'afmhot'
    prefix = 'compare_beams'

    fig, axs = plt.subplots(5, 2, sharex='all', sharey='all', figsize=(7, 15))
    plt.subplots_adjust(wspace=0)

    vra = [np.percentile(dirty_beam, 1), np.percentile(dirty_beam, 99)]
    axs[0,0].imshow(dirty_beam,vmin=vra[0],vmax=vra[1],cmap=cmap, origin='upper')
    axs[0,0].set_title('Dirty beam (dirtyim)')
    axs[0,1].imshow(dbeam,vmin=vra[0],vmax=vra[1],cmap=cmap, origin='upper')
    axs[0,1].set_title('Dirty beam (EHT)')

    vra = [np.percentile(dim_cont_sparse, 1), np.percentile(dim_cont_sparse, 99)]
    axs[1,0].imshow(dim_cont_sparse,vmin=vra[0],vmax=vra[1],cmap=cmap,origin='upper')
    axs[1,0].set_title('Dirty image (dirtyim)')
    axs[1,1].imshow(dim,vmin=vra[0],vmax=vra[1],cmap=cmap,origin='upper')
    axs[1,1].set_title('Dirty image (EHT)')

    vra = [np.percentile(clean_psf, 1), np.percentile(clean_psf, 99)]
    axs[2,0].imshow(clean_psf,vmin=vra[0],vmax=vra[1],cmap=cmap,origin='upper')
    axs[2,0].set_title('Clean beam (clean-cuda)')
    axs[2,1].imshow(cbeam,vmin=vra[0],vmax=vra[1],cmap=cmap,origin='upper')
    axs[2,1].set_title('Clean beam (EHT)')

    vra = [np.percentile(gt_image, 1), np.percentile(gt_image, 99)]
    axs[3,0].imshow(dim_grid_dense, vmin=vra[0], vmax=vra[1], cmap=cmap, origin='upper')
    axs[3,0].set_title('Dense IFFT')
    axs[3,1].imshow(gt_image,vmin=vra[0],vmax=vra[1],cmap=cmap,origin='upper')
    axs[3,1].set_title('GT Image (original)')


    vra = [np.percentile(gt_image, 1), np.percentile(gt_image, 99)]
    dirty_convolve_mine = convolve2d(gt_image, dirty_beam, mode='same')
    dirty_convolve_eht = convolve2d(gt_image, dbeam, mode='same')
    axs[4,0].imshow(dirty_convolve_mine,vmin=vra[0],vmax=vra[1],cmap=cmap,origin='upper')
    axs[4, 0].set_title('Convolved (dirtyim)')
    axs[4,1].imshow(dirty_convolve_eht, vmin=vra[0], vmax=vra[1], cmap=cmap, origin='upper')
    axs[4, 1].set_title('Convolved (EHT)')

    plt.savefig(prefix+'_clean_final.png')
    plt.close()
    #dim.display()
    #dbeam.display()
    #cbeam.display()

    # gICLEAN.clean_cuda(dirty_im=dim, dirty_psf=dbeam, thresh=1e-10, gain=1e-1, clean_beam_size=4.0,
    #                    maxIter=1e6,
    #                    prefix='Galaxy10_decals_EHT_lessnoise_dirty10.0',
    #                    im_gt=gt_image,
    #                    clean_psf=cbeam,
    #                    polarity=False)

    # gICLEAN.clean_cuda(dirty_im=dim_cont_sparse, dirty_psf=dirty_beam, thresh=1e-10, gain=1e-1, clean_beam_size=4.0,
    #                    maxIter=1e6,
    #                    prefix='Galaxy10_decals_Mine_lessnoise_dirty10.0',
    #                    im_gt=gt_image,
    #                    clean_psf=None,
    #                    polarity=False)

    prior = eh.image.make_square(obs, npix, im.fovx())
    outvis = dd_clean_vis(obs, prior, niter=500, loop_gain=0.1,
                          method='max_delta', weighting='natural',
                          show_updates=True)

    beamparams = obs.fit_beam()
    dirty_im_pred_CLEAN = outvis.blur_gauss(beamparams, 0.5).imarr()




def dd_CLEAN(gt_image, niter, loop_gain):

    fov=1.4108078120287498e-09
    npix=len(gt_image)
    pdim = fov/npix
    im = eh.image.Image(gt_image, pdim, 0, 0,)
    #im.display()

    # observe the image the same way as data generator
    meta_file ='./code/EHT2017.txt'
    eht_meta = eh.array.load_txt(meta_file)

    tadv_sec = 600
    tstart_hr = 0
    tstop_hr = 24
    tint_sec = 12
    bw_hz = 4.096e9
    obs = im.observe(eht_meta, tint_sec, tadv_sec, tstart_hr, tstop_hr, bw_hz,
                                sgrscat=False, add_th_noise=False, ampcal=True, phasecal=True, ttype='direct')

    #npix = 32
    fov2 = im.xdim * im.psize  # same as fov

    # Resolution
    beamparams = obs.fit_beam()  # fitted beam parameters (fwhm_maj, fwhm_min, theta) in radians
    res = obs.res()  # nominal array resolution, 1/longest baseline
    print("Clean beam parameters: ", beamparams)
    print("Nominal Resolution: ", res)

    #prior = eh.image.make_square(obs, 128, 1.5*im.fovx())
    #prior = eh.image.make_square(obs, 64, im.fovx())
    prior = eh.image.make_square(obs, npix, im.fovx())

    # data domain clean with visibilities
    #outvis = dd_clean_vis(obs, prior, niter=100, loop_gain=0.1, method='min_chisq', weighting='uniform', show_updates=True) # to see iterations
    #outvis = dd_clean_vis(obs, prior, niter=niter, loop_gain=loop_gain, method='min_chisq', weighting='uniform')
    #outvis = dd_clean_vis(obs, prior, niter=niter, loop_gain=loop_gain, method='min_chisq', weighting='natural')
    #outvis = dd_clean_vis(obs, prior, niter=niter, loop_gain=loop_gain, method='max_delta', weighting='uniform')
    outvis = dd_clean_vis(obs, prior, niter=niter, loop_gain=loop_gain,
                          method='max_delta', weighting='natural',
                          show_updates=False)

    beamparams = obs.fit_beam()
    dirty_im_pred_CLEAN = outvis.blur_gauss(beamparams, 0.5).imarr()

    return dirty_im_pred_CLEAN


def do_test(compare_sparse_dense=False, do_clean=False, compare_dirty=False, do_clean_cuda=True):
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, random_split

    pl.seed_everything(42)
    numVal = 32 * 16

    num_fourier_coeff = 200
    eht_npix = 200
    dset_name = 'Galaxy10_DECals' #'Galaxy10' #

    eht_cont_dset = EHT_Continuous_Dataset(eht_npix=eht_npix,
                                           num_FC=num_fourier_coeff,
                                           dset_name=dset_name,
                                           obs_type='eht')

    split_train, split_val = random_split(eht_cont_dset, [len(eht_cont_dset) - numVal, numVal])
    split_val, _ = random_split(split_val, [1, len(split_val)-1])

    # CLEAN figs
    cleaned_lst = []
    for idx in range(len(split_val)):
        print(idx)
        print('-------')
        grid_dense, cont_sparse, grid_sparse, obs_meta = split_val[idx]
        # dim_grid_dense = make_dirtyim(grid_dense['uv'], grid_dense['vis'], eht_npix, fov).imarr()

        if compare_dirty:
            plot_compare_dirtyim_ehtobs(grid_dense, cont_sparse, grid_sparse, obs_meta, obs_meta['gt_img'])

        if do_clean:
            dirty_im_pred_CLEAN = dd_CLEAN(obs_meta['gt_img'], niter=500, loop_gain=0.05)
            plt.imshow(dirty_im_pred_CLEAN, cmap='afmhot')

        if compare_sparse_dense:
            savefig = f'ehtim_grid_{num_fourier_coeff}FC_{eht_npix}im_{dset_name}_{idx:05d}_{cutoff_freq}.png'
            plot_eht_compare(grid_dense, cont_sparse, grid_sparse, obs_meta, savefig=savefig, cutoff_freq=cutoff_freq)

        if do_clean_cuda:
            cutoff_freq = 0.0
            weighting = 'uniform'
            eht_npix, eht_fov, num_fourier_coeff, sigma = obs_meta['npix'], obs_meta['fov'], \
                                                          obs_meta['n_FC'], obs_meta['sigma']

            dim_cont_sparse = make_dirtyim(cont_sparse['uv'], cont_sparse['vis'], eht_npix,
                                                                  eht_fov, sigma=sigma,
                                                                  cutoff_freq=cutoff_freq, return_im=False,
                                                                  weighting=weighting).imarr()
            
            plt.imsave('reconstructed_dirty.png', arr = abs(dim_cont_sparse), cmap='hot')
            dirty_beam = make_dirtyim(cont_sparse['uv'], np.ones_like(cont_sparse['vis']),
                                                            eht_npix, eht_fov, sigma=sigma,
                                                            cutoff_freq=cutoff_freq, return_im=False,
                                                            weighting=weighting).imarr()
            np.save("beam.npy", dirty_beam)
            plt.imsave('reconstructed_beam.png', arr = dirty_beam, cmap='gray')
            # dirty_im = np.load('dirty.npy')
            print(dim_cont_sparse.shape)
            # cleaned = gICLEAN.clean_cuda(dirty_im=abs(dim_cont_sparse), dirty_psf=dirty_beam, thresh=1e-10, gain=0.1,
            #                                clean_beam_size=4,
            #                                maxIter=1e5,
            #                                prefix='../clean-cuda_val/Galaxy10_decals_clean-cuda_idx%05d' % idx,
            #                                im_gt=obs_meta['gt_img'],
            #                                clean_psf=None,
            #                                polarity=False)
            print(cleaned.shape)
            plt.imsave('reconstructed_image.png', arr = cleaned, cmap='hot')
            cleaned_lst.append(cleaned)
            cleaned_npy = np.stack(cleaned_lst)
            np.save('val_cleaned_idx%05d.npy' % idx, cleaned_npy)

if __name__ == "__main__":
    do_test(compare_sparse_dense=False, do_clean=False, compare_dirty=False, do_clean_cuda=True)
