# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:32:16 2019

@author: alien
"""

import numpy as np


from skimage.measure import label
from skimage.transform import resize
import skimage.morphology as skmorpho


from scipy import ndimage as ndi
import scipy.ndimage.morphology as morpho



# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

import time

from alienlab import *

import segmentation_func
from segmentation_func import *

from progressbar import ProgressBar


#%%
if __name__ == '__main__':

    # Load point cloud
    # ****************
 

    # Path of the file
    #file_path = 'Cassette_idclass/Cassette_GT.ply'
    #file_path = '../Z5-9/Z5.ply'
    file_path =  '../rueMadame_database/GT_Madame1_2.ply'
    
    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    #  Removal of exterme points
    pts_ind = np.sqrt((points[:,0]-np.mean(points[:,0]))**2)<3*np.std(points[:,0])
    points = points[pts_ind]
    
        

    if True:
        '''Elevation projection'''

        print('Computing elevation images')
        
        
        kx = 0.1
        ky = 0.1
        
        t0 = time.time()
        #Get the elevation image as well as the tables allowing to reverse the projection
        #For future reconstruction
        elevation_image, reverse_projection = get_elevation(points, kx, ky) 
        t1 = time.time()    
        
        print('Elevation images computed in {:.3f} seconds'.format(t1 - t0))
        
        
        write_ply('elevation_image.ply', [elevation_image],
                  ['x', 'y', 'max_elevation', 'min_elevation', 'relative_elevation', 'accumulation'])
#%%        
    if True:
        '''Observation of the elevation  images'''
        
        #Turn the elevation arrays into images
        im_max, elevation_mask = make_image(elevation_image, elevation_image, im_type = 0)
        im_min, msk = make_image(elevation_image, elevation_image, im_type = 1)
        im_range, msk = make_image(elevation_image, elevation_image, im_type = 2)
        im_accum, msk = make_image(elevation_image, elevation_image, im_type = 3)
        
        #Fill holes of the max elevation image
        im = np.copy(im_max)
        im[1:-1, 1:-1] = im_max.max()
        mask = im_max
        filled_max = skmorpho.reconstruction(im, mask, method='erosion')
        
        #Fill holes of the min elevation image
        im = np.copy(im_min)
        im[1:-1, 1:-1] = im_min.max()
        mask = im_min
        filled_min = skmorpho.reconstruction(im, mask, method='erosion')
        
        g = showclass() #Class equivalent to matplotlib.pyplot (code in alienlab)
        g.save_name = 'Elevation_images'
        g.col_num = 4
        g.title_list = ['Filled maximal elevation', 'Filled minimal elevation', 
                        'Elevation range','Accumulation' ]
        g.showing([filled_max.T, filled_min.T, im_range.T,im_accum.T])
#%%        
    if True: 
        '''Ground detection, method based on region growth in the image'''
        print('Segmentation of the ground')
        #Region growth radius of exploration
        r = 40
        #Region growth selection parameter
        C1 = 0.5
        #New seeds are chosen as the new elements 
        #added to the region at the edge of the radius of exploration
        
        g = showclass()
        
        #compute ground from maximal elevation image
        ground_max = lambda_flat2(filled_max, r, C1)
        no_ground_max = elevation_mask * ~ground_max
        
        g.cmap = 'inferno'
        g.col_num = 1
        g.save_name = 'Ground_max_images'
        g.title_list = (['original', 'ground', 'not ground'])
        g.showing([filled_max, ground_max, no_ground_max])
        
        #compute ground from minimal elevation image
        ground_min = lambda_flat2(filled_min, r, C1)
        no_ground_min = elevation_mask * ~ground_min
        g.save_name = 'Ground_min_images'
        g.showing([filled_min, ground_min, no_ground_min])
        
#%%
    if True:
        '''Selection of the object candidates'''
        
        #OBJECTS ON THE SEGMENTED GROUND  
        
        ground = (ground_min + ground_max).astype(int)        
        im_ground = ground*filled_max #To find the objects that lay on the segmented ground
        
        #To perform the search for peaks we select the surroudings of the actual image for the 
        #initiation of the search by applying a binary erosion on the ground 
        #and keep only the part that has been eroded
        surround = morpho.binary_erosion(ground, iterations = 1) - ground
        surround = surround.astype(bool)
        seed = np.copy(im_ground)
        seed[~surround] = np.min(im_ground) #All the rest of the image is set to the image minimum
        mask = im_ground
        cut_ground = skmorpho.reconstruction(seed, mask, method='dilation')
        #Collect the peaks that have been cut-off
        obj2 = im_ground-cut_ground
        obj2= make_binary(obj2, 0.1) #Locate the peaks
        
        #g.col_num = 1
        #g.title_list = ['Seed', 'Detected peaks']
        #g.showing([seed,obj2])

        #OBJECTS NOT NECESSARILY ON THE SEGMENTED GROUND
        obj1 = no_ground_max*im_max
        obj1 = make_binary(obj1, 1)       

        #ALL OBJECT CANDIDATES
        obj = obj1.astype(bool) + obj2.astype(bool)
        
        #Remove the artifacts by opening
        obj_clean = morpho.binary_opening(obj) 
        residue = obj & ~obj_clean 
        residue = residue
        
        #Reinject residues with a high accumulation value: they are not artifacts
        #But their small diameter seen from the top had them removed by the opening
        #High accumulation value show they are not artifacts but rather thin vertical objects
        #Like street posts.
        im_accum_bin = make_binary(im_accum, 6, dtyp = 'bool')
        clean = im_accum_bin & residue.astype(bool)
        obj_restored = obj_clean + clean
        
        g.col_num = 2
        g.title_list = ['Ground', 'No ground (object candidates)', 
                        'Fill ground holes (object candidates)',
                        'All object candidates (union)', 'opening result', 
                        'opening corrected with accumulation']
        g.save_name = 'Object_detection'
        plot = g.showing([im_ground, obj1, obj2, obj, residue, obj_restored])    
        
#%%
    if True:
        '''Watershed segmentation of the objects'''
        
        #Selection of local maxima
        obj_height = obj_restored * im_max
        marks = skmorpho.h_maxima(obj_height,1, selem=None)
        #Labelling as markers
        marks = ndi.label(marks)[0]
        #watershed segmentation
        label_obj = skmorpho.watershed(-obj_height, marks)
        
        #Coarser segmentation
        #reference elevation image
        ref = im_accum * im_min * obj_restored
        #removal of small objects already segmented
        ref = morpho.grey_opening(ref, size = 1)
        #Downsizing factor
        DS_factor = 10
        #Resize image     
        image_resized = resize(ref, (ref.shape[0] // DS_factor, ref.shape[1] // DS_factor))
        #New local maxima
        marks2 = skmorpho.h_maxima(image_resized,10, selem=None)
        marks2 = ndi.label(marks2)[0]# + np.max(obj_labels) to avoid giving the same labels, 
        #but removed here for visualisation
        
        #Locate the local maxima
        (indx, indy) = np.where(marks2 != 0)
        marks_resized = obj_height * 0
        #Assign local maxima in an image with original size
        marks_resized[DS_factor*indx, DS_factor*indy] = marks2[indx, indy]
        #Dilate the maxima to ease the watershed
        marks_resized = morpho.grey_dilation(marks_resized, size = 2)
        #Watershed segmentation on the new maxima
        label_obj_mark = skmorpho.watershed(-obj_height, marks_resized, mask = ref,connectivity=1)
        #selection of only the new labels
        ind = label_obj_mark != 0
        #update the labels in the original labelled image
        label_obj[ind] = label_obj_mark[ind]
        label_obj = label_obj*obj_restored
        g.cmap = 'flag'    
        g.save_name = 'Object_segmentation'
        g.title_list = ['Refined segmentation']

        g.showing(label_obj)
#%%
    if True:
        '''Projection back to 3D'''
        mount_cloud_labels = image_to_2Dcloud(label_obj, elevation_mask)
        im_ground = make_binary(im_ground, 0, 'int')

        ground = image_to_2Dcloud(im_ground, elevation_mask)
        
        obj_labels = conv_2D_3D(mount_cloud_labels, ground, reverse_projection, points)
        write_ply('object_labels.ply',
          [points, obj_labels],
          ['x', 'y', 'z', 'segm'])
#%%
    if True:
        '''Evaluation'''
        bar = ProgressBar()

        #Class-wise segmentation evaluation
        pts = np.vstack((data['x'], data['y'], data['z'], data['id'], data['class'])).T
        pts = pts[pts_ind]

        b = np.unique(pts[:,4])
        correct_tot = 0

        pred = []
        for j in bar(range(len(b))):
            ind_class =  pts[:,4]==b[j]
            pts2 = pts[ind_class]
            obj_labels2 = obj_labels[ind_class]
            a = np.unique(pts2[:,3])
            correct = 0
            for i in range(len(a)):
                L = a[i]
                ind_L = pts2[:,3] == L
                segment_L = obj_labels2[ind_L]
                set_L, count_L = np.unique(segment_L, return_counts = True)
                ID = set_L[np.argmax(count_L)]
                correct += np.count_nonzero(segment_L==ID)
                correct_tot += np.count_nonzero(segment_L==ID)
                
            pred.append(correct/pts2.shape[0]*100)
            
        p1 = correct_tot/pts.shape[0]*100        
            
        #Undersegmentation evaluation
        a = np.unique(obj_labels)
        correct = 0
        for i in range(len(a)):
            L = a[i]
            ind_L = obj_labels == L
            segment_L = pts[ind_L]
            set_L, count_L = np.unique(segment_L, return_counts = True)
            ID = set_L[np.argmax(count_L)]
            correct += np.count_nonzero(segment_L==ID)
            
        p2 = correct/pts.shape[0]*100
            
        print('Segmentation results per class (p1)',pred)
        print('Oversegmentation index (p1)', p1)
        print('Undersegmentation index (p2)',p2)
               

