# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:09:27 2021

@author: Daniel Hulsey

this script aligns a ccf to an individual session, and extracts dff data for each indicated region. 
Running it fully requires:
    1. saved dff trace for the session being analyzed
    2. saved points to align CCF to multimodalmap session
    3. blood vessel images from the CCF and session for analysis for alignment
Save location for the region dff traces needs to be better defined    

"""

from skimage import io
import uobrainflex.utils.image_utilities as im_util
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import glob 

def area_mean_dff(dff,pixel_list):
    xs = pixel_list[0,:]
    ys = pixel_list[1,:]
    return np.mean(dff[:,xs,ys],axis=1)

def area_max_dff(dff,pixel_list):
    xs = pixel_list[0,:]
    ys = pixel_list[1,:]
    return np.max(dff[:,xs,ys],axis=1)

# select folders and align session and mmm vessel images
newfolders=''
while newfolders != 'n': # repeat until user indicates that new folders should not be selected
    
    subject_title = 'select subject mmm folder'
    session_title = 'select session folder'
    root = tk.Tk()
    root.withdraw()
    subject_folder = filedialog.askdirectory(title = subject_title)
    session_folder = filedialog.askdirectory(title = session_title)
    save_folder = session_folder
    

    mmm_image_path = glob.glob(subject_folder + '/*MMM.png')[0]
    
    # session_vessels_path = glob.glob(subject_folder + '/*vessel*')[0] ## this is the future one. below for testing
    session_vessels_path = glob.glob(session_folder + '/*MMM.png')[0]
        
    mmm_image = io.imread(mmm_image_path)
    vessel_image = io.imread(session_vessels_path)
    
    repeat_align=''
    while repeat_align != 'n': # repeat until user replys to not
        session_vessel_points, session_mmm_points, figure = im_util.set_transform_anchors(vessel_image, mmm_image)
        plt.savefig(save_folder +'\mmm_to_session_align.png')
        repeat_align = input("repeat alignment with these images? Y/n\n")
    newfolders = input("repeat alignment with different folder selections? Y/n\n")
    

plt.savefig(save_folder +'\mmm_to_session_align.png')
np.save(save_folder + '\session_mmm_points',session_mmm_points)
np.save(save_folder + '\session_vessel_points',session_vessel_points)

# Now grab MMM & CCF alignments & DFF, create session masks, and grab mean region dffs
print('loading dff data')
# dff_filepath =  glob.glob(session_folder + '/*dff.npy')[0] # this is real one to use. hardcoded to demo with no real data
dff_filepath =  'T:/BW048_211013_115911/WF_raw/WF_raw_dff.npy'
dff = np.load(dff_filepath)

mmm_points_filepath = glob.glob(subject_folder + '/*mmm_points.npy')[0]
ccf_points_filepath = glob.glob(subject_folder + '/*ccf_points.npy')[0]
masks_filepath = 'C:\\Users\\admin\\Desktop\\figure dump\\CCF\\22deg\\masks.npy'
areas_filepath = 'C:\\Users\\admin\\Desktop\\figure dump\\CCF\\22deg\\areas.npy'

mmm_points = np.load(mmm_points_filepath)
ccf_points = np.load(ccf_points_filepath)
masks = np.load(masks_filepath)
areas = np.load(areas_filepath)


# st2, wp2 = im_util.set_transform_anchors(session_vessels, mmm_vessels)

print('aligning ccf regions')
session_masks=[]
# mmm_masks=[]
for i, mask in enumerate(masks):
    mmm_mask = im_util.transform_image(mmm_image, mask, mmm_points, ccf_points)
    mmm_mask[mmm_mask<.01]=0
    mmm_mask[mmm_mask>0]=1
    # mmm_masks.append(mmm_mask)
    
    session_mask = im_util.transform_image(vessel_image, mmm_mask, session_vessel_points, session_mmm_points)
    session_mask[session_mask<.01]=0
    session_mask[session_mask>0]=1
    
    session_masks.append(session_mask)
session_masks = np.array(session_masks)

session_masks = im_util.reduce_mask_overlap(session_masks)

print('extracting are dff values')
area_list = ['r_VISp','r_AUDp','r_MOs']
for area in area_list:
    # plt.figure()
    mask_ind = np.where(areas==area)[0]
    this_mask = session_masks[mask_ind,:,:][0].T
    this_dff = area_mean_dff(dff,np.array(np.where(this_mask)))
    np.save(save_folder + '/' + area + '_dff',this_dff)
    