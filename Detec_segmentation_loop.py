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


#%%
if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = 'Cassette_idclass/Cassette_GT.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'], data['id'], data['class'])).T
    points = points[np.sqrt((points[:,0]-np.mean(points[:,0]))**2)<3*np.std(points[:,0])]
    
        

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
                  ['x', 'y', 'max_elevation', 'min_elevation', 
                   'relative_elevation', 'accumulation', 'min_ground', 'max_ground'])
#%%        
    if True:
        '''Observation of the elevation  images'''
        
        #Turn the elevation arrays into images
        im_max, elevation_mask = make_image(elevation_image, elevation_image, im_type = 0)
        im_min, msk = make_image(elevation_image, elevation_image, im_type = 1)
        im_range, msk = make_image(elevation_image, elevation_image, im_type = 2)
        im_accum, msk = make_image(elevation_image, elevation_image, im_type = 3)
        im_class, msk = make_image(elevation_image, elevation_image, im_type = 5)
        
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
        g.col_num = 4
        g.title_list = ['Filled maximal elevation', 'Filled minimal elevation', 
                        'Elevation range','Accumulation' ]
        g.save_name = 'Elevation_images'
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
        ground_max = lambda_flat2(filled_max, r, C1)

        no_ground_max = elevation_mask * ~ground_max
        g.cmap = 'inferno'
        g.col_num = 1
        g.title_list = (['original', 'ground', 'not ground'])
        g.save_name = 'Ground_max_images'
        g.showing([filled_max, ground_max, no_ground_max])
        
        ground_min = lambda_flat2(filled_min, r, C1)

        no_ground_min = elevation_mask * ~ground_min
        g.save_name = 'Ground_min_images'
        g.showing([filled_min, ground_min, no_ground_min])
        
        
        
#%%
    if True:
        '''Selection of the object candidates'''
        
        #OBJECTS ON THE SEGMENTED GROUND  
        print('Object Detection')
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
        
        g.col_num = 1
        g.title_list = ['Seed', 'Detected peaks']
        g.showing([seed,obj2])

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
        g.save_name = 'Object_Detection'
        plot = g.showing([im_ground, obj1, obj2, obj, residue, obj_restored])    
        
#%%
    if True:
        print('Object segmentation')
        obj_height = obj_restored * im_max
        marks = skmorpho.h_maxima(obj_height, 2, selem=None)
        marks = ndi.label(marks)[0]
        label_obj = skmorpho.watershed(-obj_height, marks, connectivity=1,
                           offset=None, mask=obj_restored.astype(bool), compactness=0, watershed_line=False)
        g.cmap = 'flag'  
        g.save_name = 'Object_Segmentation'
        g.title_list = ['Object segmentation']
        g.showing(label_obj)
                

#%%
    if True:
        print('Save cloud')
        mount_cloud_labels = image_to_2Dcloud(label_obj, elevation_mask)
        im_ground = make_binary(im_ground, 0, 'int')

        ground = image_to_2Dcloud(im_ground, elevation_mask)
        
        obj_labels = conv_2D_3D(mount_cloud_labels, ground, reverse_projection, points)
        write_ply('object_labels.ply',
          [points, obj_labels],
          ['x', 'y', 'z', 'id', 'class','segm'])
        
#%% 

    if True:
        print('Coarser elevation image')
        from imp import reload
        reload(segmentation_func)

        labs = np.unique(points[:, 4])
        tree = labs[20]
        name = 'object_labels_tree.ply'
        ind = points[:, 4] == tree
        write_ply(name,[points[ind], obj_labels[ind]],['x', 'y', 'z', 'id', 'class', 'segments'])
        tree_cloud = np.copy(points)
        tree_cloud[~ind, 2] = points[:,2].min()

        tree_segments = obj_labels[ind]
        kx = 1
        ky = 1
        
        elevation_tree, reverse_projection_tree= get_elevation(tree_cloud, kx, ky) 
#%%   
    if True:
        larg = 1000
        im_max_L, elevation_mask_L = make_image(elevation_tree, elevation_tree, im_type = 0)
        im_min_L, msk = make_image(elevation_tree, elevation_tree, im_type = 1)
        im_range_L, msk = make_image(elevation_tree, elevation_tree, im_type = 2)
        im_accum_L, msk = make_image(elevation_tree, elevation_tree, im_type = 3)
        im_id_L, msk = make_image(elevation_tree, elevation_tree, im_type = 4)
        im_class_L, msk = make_image(elevation_tree, elevation_tree, im_type = 5)

        
        im_max_L = im_max_L[:, :larg]
        im_min_L = im_min_L[:, :larg]
        im_range_L = im_range_L[:, :larg]
        im_accum_L = im_accum_L[:, :larg]
        
        g = showclass()
        g.col_num = 1
        g.cmap = 'flag'
        g.title_list = ['Maximal_elevation', 'Image ID', 'Image class' ]
        g.save_name = 'Coarser_Elevation'
        g.showing([im_max_L, im_id_L,im_class_L])
 
#%%
    if True:
        marks = skmorpho.h_maxima(im_accum_L*im_min_L, 1, selem=skmorpho.selem.diamond(5))
        marks = ndi.label(marks)[0]
        im_class_L[im_class_L != tree] = 0

        label_tree = skmorpho.watershed(im_class_L, marks)
        
        im_class, msk = make_image(elevation_image, elevation_image, im_type = 5)

        marks_resized = resize(marks*0, im_class.shape)

        im_class[im_class != tree] = 0
        im_class = morpho.binary_dilation(im_class, iterations = 3)
        (indx, indy) = np.where(marks != 0)
        marks_resized[10*indx, 10*indy] = marks[indx, indy]
        marks_resized = morpho.grey_dilation(marks_resized, size = 5)

        distance_tree = ndi.distance_transform_edt(im_class)
        
        label_tree =skmorpho.watershed(-obj_height, marks_resized, mask = im_class)
        g.title_list = ['Markers', 'Segmented trees']

        g.save_name = 'Coarser_markers'
        g.showing([marks_resized, label_tree])#, label_obj * im_class])
#%%  
        
    if True:
        label_obj[label_tree != 0] = label_tree[label_tree != 0]*10# + np.max(label_obj)
        
        g.cmap = 'flag' 
        g.title_list = ['Refined Segmentation']
        g.save_name = 'Looped_segmentation'
        g.showing(label_obj)
                

#%%
    if True:
        print('Save refined segmentation')
        mount_cloud_labels = image_to_2Dcloud(label_obj, elevation_mask)
        im_ground = make_binary(im_ground, 0, 'int')

        ground = image_to_2Dcloud(im_ground, elevation_mask)
        
        obj_labels = conv_2D_3D(mount_cloud_labels, ground, reverse_projection, points)
        write_ply('object_labels_loop.ply',
          [points, obj_labels],
          ['x', 'y', 'z', 'id', 'class','segm'])
        
        labs = np.unique(points[:, 4])
        tree = labs[20]
        name = 'object_labels_tree_loop.ply'
        ind = points[:, 4] == tree
        write_ply(name,[points[ind], obj_labels[ind]],['x', 'y', 'z', 'id', 'class', 'segments'])
        tree_cloud = np.copy(points)
        tree_cloud[~ind, 2] = points[:,2].min()