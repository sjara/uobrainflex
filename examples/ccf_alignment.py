# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:09:27 2021

@author: Daniel Hulsey

this script is used to align a CCF to a multimodalmapping session. Points selected to align the ccf will be saved for future use.
"""

import os
from skimage import io
import uobrainflex.utils.image_utilities as im_util
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

mmm_title = 'Select multi modal map image'
mmm_dir = 'C:\\Users\\admin\\Desktop\\figure dump\\CCF'
mmm_image_path = filedialog.askopenfilename(title = mmm_title, initialdir = mmm_dir)

##hardcode this? or put it in conf file?
ccf_title = 'Select CCF image'
ccf_dir = 'C:\\Users\\admin\\Desktop\\figure dump\\CCF'
ccf_file_path = filedialog.askopenfilename(title = ccf_title, initialdir = ccf_dir)


save_folder = os.path.dirname(mmm_image_path)
subject = os.path.basename(mmm_image_path)
subject = subject[:subject.find("_")]


mmm_image = io.imread(mmm_image_path)
ccf_image = io.imread(ccf_file_path)
mmm_points, ccf_points, figure = im_util.set_transform_anchors(mmm_image, ccf_image)

plt.savefig(save_folder + '/' + subject +'_mmm_ccf_fit.png')
np.save(save_folder + '/' + subject + '_mmm_points',mmm_points)
np.save(save_folder + '/' + subject + '_ccf_points',ccf_points)