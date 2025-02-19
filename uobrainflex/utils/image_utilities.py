# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 11:30:51 2021

@author: admin
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.measure import block_reduce
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.filters import gaussian
from scipy import ndimage
import scipy.ndimage.morphology as ni

# def import_multi_tif(tif_folder, n_channels=2, down_sample_factor= 2,dtype=np.float16):
#     tif_filepaths = glob.glob(os.path.join(tif_folder + '\*.tif'))
#     print('\n' + str(len(tif_filepaths)) + ' file(s) to process')
#     for i, image_path in enumerate(tif_filepaths):
#         im = io.imread(image_path)
#         im = np.array(im,dtype=dtype)
#         if n_channels==2:
#             if i == 0:
#                 wf_blue = np.ndarray([0, int(im.shape[1]/down_sample_factor), int(im.shape[2]/down_sample_factor)])
#                 wf_green = np.ndarray([0, int(im.shape[1]/down_sample_factor), int(im.shape[2]/down_sample_factor)])
                
#             first = np.atleast_3d(block_reduce(im[np.arange(0,len(im),2)],(1,down_sample_factor,down_sample_factor),func=np.mean))
#             second = np.atleast_3d(block_reduce(im[np.arange(1,len(im),2)],(1,down_sample_factor,down_sample_factor),func=np.mean))
#             if wf_blue.shape[0] == wf_green.shape[0]:
#                 wf_blue = np.append(wf_blue, first, axis=0)
#                 wf_green = np.append(wf_green, second, axis=0)
#             else:
#                 wf_blue = np.append(wf_blue, second, axis=0)
#                 wf_green = np.append(wf_green, first, axis=0)                  
#         print('\nFile ' + str(i+1) + ' of ' + str(len(tif_filepaths)) + ' complete')
#     return wf_blue, wf_green

#####################
# functions for handling widefield one photon calcium imaging data

def import_multi_tif(tif_folder, n_channels=2, down_sample_factor= 2,dtype=np.float16):
    tif_filepaths = glob.glob(os.path.join(tif_folder + '\*.tif'))
    print('\n' + str(len(tif_filepaths)) + ' file(s) to process')
    blue_frame = 0 
    green_frame = 0 
    for i, image_path in enumerate(tif_filepaths):
        im = io.imread(image_path)
        im = np.array(im,dtype=dtype)
        file_frames = round(im.shape[0]/down_sample_factor)
        
        if n_channels==2:
            if i == 0:
                wf_blue = np.ndarray([len(tif_filepaths)*file_frames, int(im.shape[1]/down_sample_factor), int(im.shape[2]/down_sample_factor)],dtype=dtype)
                wf_green = np.ndarray([len(tif_filepaths)*file_frames, int(im.shape[1]/down_sample_factor), int(im.shape[2]/down_sample_factor)],dtype=dtype)
                
            if blue_frame == green_frame:
                green_data = np.atleast_3d(block_reduce(im[np.arange(1,len(im),2)],(1,down_sample_factor,down_sample_factor),func=np.mean))
                blue_data = np.atleast_3d(block_reduce(im[np.arange(0,len(im),2)],(1,down_sample_factor,down_sample_factor),func=np.mean))
            elif blue_frame!=green_frame:
                green_data = np.atleast_3d(block_reduce(im[np.arange(0,len(im),2)],(1,down_sample_factor,down_sample_factor),func=np.mean))
                blue_data = np.atleast_3d(block_reduce(im[np.arange(1,len(im),2)],(1,down_sample_factor,down_sample_factor),func=np.mean))           
            
            wf_green[green_frame:green_frame+green_data.shape[0]]= green_data
            wf_blue[blue_frame:blue_frame+blue_data.shape[0]]= blue_data
        elif n_channels==1:
            if i == 0:
                wf_blue = np.ndarray([len(tif_filepaths)*im.shape[0], int(im.shape[1]/down_sample_factor), int(im.shape[2]/down_sample_factor)],dtype=dtype)
                wf_green = np.full([10000,100,100],np.nan) 
                green_data = np.full([10000,100,100],np.nan) 
            blue_data = np.atleast_3d(block_reduce(im,(1,down_sample_factor,down_sample_factor),func=np.mean))
            wf_blue[blue_frame:blue_frame+blue_data.shape[0]]= blue_data


        blue_frame = blue_frame+blue_data.shape[0]
        green_frame = green_frame+green_data.shape[0]
        print('\nFile ' + str(i+1) + ' of ' + str(len(tif_filepaths)) + ' complete')
    return wf_blue[:blue_frame,:,:], wf_green[:green_frame,:,:]

def dff_mean_calculation(image_stack):
    pixel_mean = np.mean(image_stack,axis=0)
    dff = (image_stack-pixel_mean)/pixel_mean
    return dff

def dff_median_calculation(image_stack):
    pixel_median = np.median(image_stack,axis=0)
    dff = (image_stack-pixel_median)/pixel_median
    return dff

def hemo_correction(signal_dff,hemo_dff):
    if signal_dff.shape[0] != hemo_dff.shape[0]:
        print('WARNING! image stack size mismatch. truncating to hemo signal size')
        corrected_dff = signal_dff[:hemo_dff.shape[0],:,:]-hemo_dff
    else:
        corrected_dff = signal_dff-hemo_dff
    return corrected_dff

def image_contrast_adjust(im_array,quantile = .99):
    im = im_array-np.quantile(im_array[im_array==im_array],.2)
    im[im!=im]=0
    im[im<0]=0
    
    im = im/np.quantile(im,quantile)
    im[im>1]=1
    return im*255

def create_vessel_and_skull_image(tif_folder, n_channels=2, down_sample_factor= 2):
    tif_filepaths = list(tif_folder.glob('*.tif'))
    im = io.imread(str(tif_filepaths[0]))    
    v_substack = block_reduce(im[np.arange(0,2000,n_channels)],(1,down_sample_factor,down_sample_factor),func=np.mean)
    vessel_im = v_substack.std(axis=0)
    vessel_im = image_contrast_adjust(vessel_im,quantile = .99)
    
    s_substack = block_reduce(im[np.arange(1,2000,n_channels)],(1,down_sample_factor,down_sample_factor),func=np.mean)
    skull_im = s_substack[100,:,:]
    skull_im = image_contrast_adjust(skull_im,quantile = .95)   
    return vessel_im.astype(np.uint8), skull_im.astype(np.uint8)
    
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def dff_to_alpha(dff,fs=30):
    alpha=dff.copy()
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [3,6], btype='pass', analog=False, fs=fs)
    for i in range(alpha.shape[1]):
        print(i)
        for j in range(alpha.shape[2]):
            apower = filtfilt(b, a, alpha[:,i,j])
            apower = moving_average(abs(apower), round(fs/3*2))
            alpha[:,i,j]=apower.astype(np.float16)
    return alpha

def dff_to_sigma(dff,fs=30):
    sigma=dff.copy()
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [9,16], btype='pass', analog=False, fs=fs)
    for i in range(sigma.shape[1]):
        print(i)
        for j in range(sigma.shape[2]):
            spower = filtfilt(b, a, sigma[:,i,j])
            spower = moving_average(abs(spower), round(fs/3))
            sigma[:,i,j]=spower.astype(np.float16)
    return sigma

def dff_to_delta(dff,fs=30):
    delta=dff.copy()
    from scipy.signal import butter, filtfilt
    b, a = butter(4, [1,4], btype='pass', analog=False, fs=fs)
    for i in range(delta.shape[1]):
        print(i)
        for j in range(delta.shape[2]):
            dpower = filtfilt(b, a, delta[:,i,j])
            dpower = moving_average(abs(dpower), round(fs/3*2))
            delta[:,i,j]=dpower.astype(np.float16)
    return delta
#####################
# functinos for Allen CCF alignment

def set_transform_anchors(stationary_image, warp_image):
    figure = plt.figure()
    ax = plt.subplot(1,2,1)
    ax.imshow(stationary_image)
    ax2 = plt.subplot(1,2,2)
    ax2.imshow(warp_image)
    
    plt.suptitle('Set points to anchor transformation\n'+
                 'Select point on left then right image for each anchor\n'+
                 'left click to select | right click to cancel last select | middle click to complete')
    points = np.array(plt.ginput(100,timeout=300))

    stationary_points = points[np.array(range(0,len(points),2),dtype=int)]
    warp_points = points[np.array(range(1,len(points),2),dtype=int)]
    ax.scatter(stationary_points[:,0],stationary_points[:,1],color='b')
    for idx in range(len(stationary_points)):
        ax.text(stationary_points[idx][0],stationary_points[idx][1],str(idx),color='r')
    ax2.scatter(warp_points[:,0],warp_points[:,1],color='b')
    for idx in range(len(warp_points)):
        ax2.text(warp_points[idx][0],warp_points[idx][1],str(idx),color='r')
    plt.pause(0.1)
    return stationary_points, warp_points, figure

def transform_image(stationary_image,warp_image,stationary_points, warp_points):
    tform = PiecewiseAffineTransform()
    tform.estimate(stationary_points, warp_points)
    out_rows = stationary_image.shape[0]
    out_cols = stationary_image.shape[1]
    warped_image = warp(warp_image, tform, output_shape=(out_rows, out_cols))
    return warped_image

def smooth_masks(masks):
    for i in range(masks.shape[0]):
        img = masks[i,:,:]
        dst = gaussian(1-img,sigma=(1))
        dst[dst<=.01]=1
        dst[dst<.1]=0
        masks[i,:,:]=dst
    return masks

def reduce_mask_overlap(masks):
    for i in range(masks.shape[0]):
        this_mask = masks[i,:,:]
        other_masks = masks[np.where([np.arange(masks.shape[0])!=i])[1],:,:]
        other_masks_merge = np.max(other_masks,axis=0)
        overlap = np.where((this_mask+other_masks_merge)>1)
        this_mask[overlap[0],overlap[1]]=0
        masks[i,:,:]=this_mask
    return masks

def transform_masks(static_image,masks,static_points,warp_points):
    t_mask = np.full([masks.shape[0],static_image.shape[0],static_image.shape[1]],0)
    for i in range(masks.shape[0]):
        this_mask = masks[i,:,:]
        this_mask[this_mask!=0]=.5
        new_mask = transform_image(static_image,this_mask,static_points, warp_points)
        new_mask[new_mask>0]=1
        t_mask[i,:,:] =new_mask
    return t_mask

def masks_to_outlines(masks):
    outlines = np.zeros(masks.shape,dtype=np.int8)
    for i in range(outlines.shape[0]):
        mask = masks[i,:,:]
        outlines[i,:,:] = mask - ni.binary_erosion(mask)
    return outlines

def rotate_3d_matrix(array_3d,angle = 22.5):   
    this_slice = array_3d[0,:,:]
    this_rotated = ndimage.rotate(this_slice, angle, reshape=True)
    rotated_matrix = np.ndarray([array_3d.shape[0],this_rotated.shape[0],this_rotated.shape[1]],dtype=np.int8)
    for idx in range(array_3d.shape[0]):
        this_slice = array_3d[idx,:,:]
        if len(np.where(this_slice==1)[0])>0:
            this_rotated = ndimage.rotate(this_slice, angle, reshape=True)
            this_rotated[this_rotated>=.2]=1
            this_rotated[this_rotated<.2]=0
            rotated_matrix[idx,:,:] = this_rotated
    return rotated_matrix

def surface_projection(volume):
    shell_volume = np.zeros(volume.shape,np.int8)
    for ii in range(volume.shape[0]):
        for jj in range(volume.shape[2]):
            zplane = np.where(volume[ii,:,jj]>0)[0]
            if len(zplane)>0:
                shell_volume[ii,zplane[0],jj] = 1
    return shell_volume