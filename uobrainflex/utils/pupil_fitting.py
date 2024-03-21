# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:03:00 2022

@author: admin
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

def set_eye_crop(avi_files):
    cap = cv2.VideoCapture(str(avi_files[0]))
    ret, frame = cap.read()
    plt.figure()
    # for i in range (1000):
    ret, frame = cap.read()
    plt.imshow(frame)
    plt.show()
    crop_points = plt.ginput(2)
    plt.close()
    cap.release()
    cv2.destroyAllWindows()
    
    x=[]
    y=[]
    for point in crop_points:
        x.append(int(point[1]))
        y.append(int(point[0]))
    x.sort()
    y.sort()

    return x, y

def largest_contour(crop_frame):
    contours,_ = cv2.findContours(crop_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    long_axis = [None]*len(contours)
    for i, c in enumerate(contours):
        if len(c) > 10:
            ellipse = cv2.fitEllipse(c)
            long_axis[i] = max(ellipse[1])
        else:
            long_axis[i]=0

    if len(long_axis) > 0:
        if count>100:
            last_pupil = np.median(pupil_diameter[count-100:count-1])
            if max(long_axis)>last_pupil*2:
                ind = np.where(long_axis>last_pupil*2)[0]
                long_axis[int(ind)]=0
                
    return max(long_axis)

def combined_contours(crop_frame):
    ind = 0 
    contours,_ = cv2.findContours(crop_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    conts = np.array(contours)
    while ind<len(conts):
        if len(conts)>1:
            c= np.array(conts[ind])
        else:
            c= np.array(conts[0])
        all_vals = np.concatenate(np.concatenate(c))
        mx = max(all_vals)
        mn = min(all_vals)
        if any([mn <2, mx>max(crop_frame.shape)-2, len(conts[ind])<10]):
            if all([ind == 0,len(conts)==1]):
                conts=[]
            else:
                conts = np.concatenate((conts[:ind], conts[ind+1:]))
        else:
            ind=ind+1
                        
    if len(conts)>0:
        c = np.concatenate(conts)
        ellipse = cv2.fitEllipse(c)
        long_axis=max(ellipse[1])
        short_axis = min(ellipse[1])
    else:
        long_axis=np.nan
        short_axis=np.nan
        
    return long_axis, short_axis


def set_threshold(avi_files,x,y):
    frames = np.full([len(avi_files),np.diff(x)[0],np.diff(y)[0]],np.nan)
    for i, file in enumerate(avi_files):
        cap = cv2.VideoCapture(str(file))
        ret, frame = cap.read()
        frames[i,:,:] = np.array(frame[x[0]:x[1],y[0]:y[1],0])
        cap.release()

    n_subs = round(len(avi_files)/5)+3
    upper_t = 100
    lower_t = 10
    plt.figure(figsize=(12, 20))
    for i in range(10):
        if upper_t != lower_t:
            for i,threshold in enumerate(range(lower_t,upper_t,int(upper_t/10-lower_t/10))):
                if i<9:
                    plt.subplot(n_subs,5,i+1)
                    frame_extra = np.array(frames[-1,:,:]).astype(np.uint8)
                    crop_frame = frame_extra      
                    crop_frame[crop_frame>threshold]=255
                    crop_frame[crop_frame<=threshold]=0
                    contours,_ = cv2.findContours(crop_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    long_axis = combined_contours(crop_frame)
                                            
                    img_contours = np.zeros([crop_frame.shape[0],crop_frame.shape[1],3])
                    # draw the contours on the empty image
                    img = cv2.drawContours(img_contours, contours, -1, (0,255,0))
                    plt.imshow(img.astype(int))
                    plt.title('threshold = ' +str(threshold) + ' \nlong axis = ' + str(np.round(max(long_axis),decimals=2)))
            plt.draw()
            
            
            for j,frame in enumerate(frames):
                plt.subplot(n_subs,5,j+11)
                frame_extra = np.array(frame).astype(np.uint8)
                crop_frame = frame_extra           
                crop_frame[crop_frame>lower_t]=255
                crop_frame[crop_frame<=lower_t]=0
                contours,_ = cv2.findContours(crop_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                long_axis = combined_contours(crop_frame)
                                        
                img_contours = np.zeros([crop_frame.shape[0],crop_frame.shape[1],3])
                # draw the contours on the empty image
                img = cv2.drawContours(img_contours, contours, -1, (0,255,0))
                plt.imshow(img.astype(int))
                plt.title('file ' + str(j) +'\nlong axis = ' + str(round(max(long_axis),2)))
            plt.draw()
            plt.ginput(1)
            upper_t =  int(input('enter threshold upper limit\n'))
            lower_t = int(input('enter threshold lower limit\n'))
            plt.clf()
            
    plt.close()
    return lower_t

def extract_motion_energy(avi_files,x,y):
    total_frames = 10000*len(avi_files)
    me = np.full(total_frames,np.nan)
    true_frame_count = 0 
    start_t = time.time()
    last_frame = np.zeros([np.diff(x)[0],np.diff(y)[0]])
    for i, filename in enumerate(avi_files):
        count = i*10000
        cap = cv2.VideoCapture(str(filename))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if count % 10000 == 0:
                    fps = round(count/(time.time()-start_t))
                    if fps==0:
                        mins_left = '????'
                    else:
                        mins_left = round((total_frames-count)/fps/60)
                    print(str(count) + ' of ' + str(total_frames) + ' frames complete at ' + str(fps) + 'fps. ' + str(mins_left) + ' minutes reamining.')
                crop_frame = frame[x[0]:x[1],y[0]:y[1],0].astype(int)
                if crop_frame.shape[:2] == last_frame.shape[:2]:
                    if true_frame_count==0:
                        last_frame = crop_frame
                    me[count] = np.mean(abs(last_frame-crop_frame))/255
                    count = count+1
                    last_frame = crop_frame
                    true_frame_count = true_frame_count+1
                else:
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()   
    return me, true_frame_count

def fit_pupils(avi_files,x,y,threshold,method = 1):
    total_frames = 10000*len(avi_files)
    pupil_long_axis = np.full(total_frames,np.nan)
    pupil_short_axis = np.full(total_frames,np.nan)
    true_frame_count = 0 
    start_t = time.time()
    for i, filename in enumerate(avi_files):
        count = i*10000
        cap = cv2.VideoCapture(str(filename))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if count % 10000 == 0:
                    fps = round(true_frame_count/(time.time()-start_t))
                    if fps==0:
                        mins_left = '????'
                    else:
                        mins_left = round((total_frames-count)/fps/60)
                    print(str(count) + ' of ' + str(total_frames) + ' frames complete at ' + str(fps) + 'fps. ' + str(mins_left) + ' minutes reamining.')
                    
                crop_frame = frame[x[0]:x[1],y[0]:y[1],0]
                crop_frame[crop_frame>threshold]=255
                crop_frame[crop_frame<=threshold]=0
                
                if method == 0:
                    long_axis = largest_contour(crop_frame)
                elif method ==1:
                    long_axis,short_axis = combined_contours(crop_frame)
                
                pupil_long_axis[count] = long_axis
                pupil_short_axis[count]= short_axis
                count = count+1
                true_frame_count = true_frame_count+1
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        
    pupil_long_axis = pupil_long_axis[:count]
    pupil_short_axis = pupil_short_axis[:count]
    
    return pupil_long_axis, pupil_short_axis

def pupil_cleanup(pupil_diameter,lookback=30,upper_threshold=1.5,lower_threshold=.75):
    clean_pupil_diameter = np.full(pupil_diameter.shape,np.nan)
    
    # make values with large jumps up or down nan
    for i,dia in enumerate(pupil_diameter):
        if i>100:
            last_pupil = np.nanmedian(clean_pupil_diameter[i-lookback:i-1])
            if last_pupil!=last_pupil:
                last_pupil = np.nanmedian(clean_pupil_diameter[i-100:i-1]) 
            if any([dia>last_pupil*upper_threshold, dia<last_pupil*lower_threshold]):
                clean_pupil_diameter[i] = np.nan
            else:
                clean_pupil_diameter[i] = dia
        elif i<=100:
            clean_pupil_diameter[i] = dia
    
    
    # make values around a nan also nan
    ind = np.where(clean_pupil_diameter != clean_pupil_diameter)[0]
    ind = np.unique(np.concatenate([ind,ind+1,ind+2,ind-1]))
    ind = ind[1:-2]
    clean_pupil_diameter[ind]=np.nan
    
    
    ## interpolate across nan values
    # if interp:
    #     vals = np.where(pupil_diameter==pupil_diameter)
    #     nans = np.where(pupil_diameter!=pupil_diameter)
    #     clean_pupil_diameter = np
        
        
    #     i=0
    #     while i<len(clean_pupil_diameter):
    #         i_val =  clean_pupil_diameter[i]
            
    #         if i <len(pupil_diameter)-1:
    #             j=i+1
    #             j_val = clean_pupil_diameter[j]
            
    #         if j_val != j_val:
    #             while all([j_val!=j_val,j<len(clean_pupil_diameter)]):
    #                 j=j+1
    #                 j_val=clean_pupil_diameter[j]     
    #             clean_pupil_diameter[range(i,j)]=np.interp(range(i,j),[i,j],[i_val,j_val])
        
    #         if i==0:
    #             i=j
    #         elif all([i_val==i_val,j_val==j_val]):
    #             while j_val==j_val:
    #                 j=j+1
    #                 j_val=clean_pupil_diameter[j]
    #             i=j-1    
    #         elif j==len(clean_pupil_diameter)-1:
                # break
    return clean_pupil_diameter

########## example ########
# session_folder = '\\\\mammatus2.uoregon.edu\\home\\brainflex\\temp data\\BW039_210826_143635\\'
# avi_folder = session_folder + 'AVI_Files\\'

# avi_files = glob.glob(avi_folder +'*.avi')[1:]
# x,y = set_eye_crop(avi_files)
# threshold = set_threshold(avi_files,x,y)


# pupil_long_axis, pupil_short_axis = fit_pupils(avi_files,x,y,threshold,method = 1)
# clean_long_axis = pupil_cleanup(pupil_long_axis)
# clean_short_axis = pupil_cleanup(pupil_short_axis)
        

# np.save(session_folder + 'post_hoc_pupil_diameter.npy',clean_long_axis)
# np.save(session_folder + 'post_hoc_pupil_short_axis.npy',clean_short_axis)

# plt.figure()
# plt.plot(pupil_diameter)
# plt.plot(clean_pupil)
