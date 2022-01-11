# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 16:43:40 2021

@author: Daniel Hulsey

This script creates and saves a 3d frame x pixel x pixel array of dff values from a user identified folder containing a series of multitiff files.
"""

import uobrainflex.utils.image_utilities as im_util
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def dff_calculation(image_stack):
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

folder_title = 'select folder with session WF data'
root = tk.Tk()
root.withdraw()
tif_folder = filedialog.askdirectory(title = folder_title)

session_id = os.path.basename(tif_folder)
while session_id.find('/') != -1:
    session_id=session_id[session_id.find('/')+1:]

wf_blue, wf_green = im_util.import_multi_tif(tif_folder, n_channels=2, down_sample_factor= 2)
print('calculating dff')
dff_blue = dff_calculation(wf_blue)
dff_green = dff_calculation(wf_green)
print('applying hemodynamic correction')
dff = hemo_correction(dff_blue, dff_green)

print('saving dff file')
np.save(tif_folder + '\\'+ session_id + '_dff',np.array(dff,dtype=np.float16))