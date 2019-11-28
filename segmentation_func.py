# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:30:53 2019

@author: alien
"""

import numpy as np

from sklearn.neighbors import KDTree

from skimage.measure import label
from skimage.transform import resize

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

import time

import matplotlib.pyplot as plt

from alienlab import *

import scipy.ndimage.morphology as morpho
import skimage.morphology as skmorpho

from progressbar import ProgressBar
#%%
# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/

bar = ProgressBar()

    
def get_elevation(cloud, kx, ky):
    '''
    Generate the elevation images from the point cloud
    Inputs: 3D cloud [stack of arrays]: list of positions x,y,z, and other attributes are acceptable
            ks, ky [float] parameter for height to voxel conversion
            
    Outputs: elevation images: array of at least 4 images:
                                minimal elevation, maximal elevation, range, accumulation
                                and if provided class and labels of the projected points
            reverse projection: list of tuples containing 
                                -the set of points projected onto each voxel,
                                -among those, points whose elevation is lower then 5mm above minimal elevation
                                -maximal elevation below 5mm of maximal elevation
                                
    '''

    points = np.copy(cloud)
    #point cloud limits
    point_cloud_max = np.max(points, 0)
    point_cloud_min = np.min(points, 0)
    
    #converion into voxel limits
    x_max = (point_cloud_max[0]/kx).astype(int)
    x_min = (point_cloud_min[0]/kx).astype(int)
    
    y_max = (point_cloud_max[1]/ky).astype(int)
    y_min = (point_cloud_min[1]/ky).astype(int)
    
    #outputs to fill
    elevation_image = []
    reverse_projection = []
    
    points_vox = np.copy(points)
    #Assignment to voxel values
    points_vox[:,0] = (points[:,0]/kx).astype(int)
    points_vox[:,1] = (points[:,1]/ky).astype(int)
    
    #minimal elevation of the point cloud
    zmin = point_cloud_min[2]
    
    #Scan the xy grid to update the elevation images
    #Could be probably faster with np.unique which I discovered later on...
    for x in bar(range(x_min, x_max)):
        x_ind = points_vox[:,0] == x
        points_x = points[x_ind]
        P_locx = points_vox[x_ind]
        val_ind = np.where(x_ind)[0]
        
        for y in range(y_min, y_max):
            y_ind = P_locx[:,1] == y
            points_y = points_x[y_ind]      
            
            if len(points_y) != 0:
                val_ind2 = val_ind[y_ind]
                #maximal elevation                    
                maxi = np.max(points_y[:,2])
                #minimal elevation
                mini = np.min(points_y[:,2])
                
                #Points with close minimal or maximal elevation (especially minimal for ground assignment)
                eps = 0.05
                ind_max = val_ind2[np.abs(points_y[:, 2]-maxi) < eps]
                ind_min = val_ind2[np.abs(points_y[:, 2] - mini) < eps]
                
                if cloud.shape[1] == 3: #point clous x, y, z
                    
                    elevation_image.append([x, y, maxi-zmin, mini-zmin, 
                                            maxi - mini, len(points_y)])
                else: #point clous x, y, z, label, class
                    elevation_image.append([x, y, maxi-zmin, mini-zmin, maxi - mini, 
                                        len(points_y), points[ind_max[0], -2], points[ind_max[0], -1]])
                    
                reverse_projection.append((val_ind2, ind_max, ind_min))
    return np.array(elevation_image), reverse_projection

 
def region_criterion(P1, P2, lbd):  
    '''Selection criterion for ground region growth
    inputs: P1 [array]: image patch neighbour of the seed
            P2 [float]: elevation of the seed
            lbd [float]: difference of elevatioon constraint
            '''    
    crit = (np.abs(P1-P2) < lbd) * (P1 != 0)   
    return crit


def lambda_flat2(im, r = 25, C1 = 2):
    '''Region growth to detect the ground'''
    Lx, Ly = im.shape
  
    aux = np.copy(im)
    #Set the background to maximal value to avoid including it in the region growth
    aux[im == 0] = np.max(im)
    #Initial seed should be in the ground, highly probable if taken among the 
    #points with an elevation around 0% of the elevation histogram
    #This parameter works fairly well on different point clouds
    v = np.percentile(aux, 0.2, interpolation = 'lower')
    #seed selection
    seed = np.where(aux == v)
    i = int(seed[0][0])
    j = int(seed[1][0])   
    
    #Region to fill with ground
    region = np.zeros((Lx, Ly), dtype=bool)
    
    #Seed lists: actual and memorial
    Q = []
    Q_mem = []
    #memorize the seeds that have already been used
    #to avoid using them again

    Q.append(j*Lx + i)
    Q_mem.append(seed)


        
    while len(Q) > 0:
        #exxtract the seed
        p_ind = Q.pop()
        
        i = p_ind % Lx
        j = p_ind // Lx
        
        P2 = im[i, j]
        region[i, j] += 1
        #Characteristics of the seed
        #Deal with limit condition problems when extracting neighborhood
        #Source: Gabriel Peyre NL-means patch-wise denoising - Numerical tours for Python
        [X,Y] = np.meshgrid(np.arange(i-r,i + r+1),np.arange(j-r,j + r +1))
        X[X < 0] = 1-X[X < 0] 
        Y[Y < 0] = 1-Y[Y < 0]
        X[X >= Lx] = 2*Lx-X[X >= Lx]-1
        Y[Y >= Ly] = 2*Ly-Y[Y >= Ly]-1
        
        
        P1 = im[X, Y]
        #Neighborhood selection
        
        crit1 = region_criterion(P1, P2, C1)
        #Elevation difference criterion
        
        region[X*crit1, Y*crit1] = region[i, j]
        #update the region
        
        #potential new seeds  
        crit2 = np.zeros((2*r + 1, 2*r + 1), dtype = bool)
        crit2[r, 0] = True
        crit2[0, r] = True
        crit2[2*r, r]= True
        crit2[r, 2*r] = True

        #new seeds selection
        new_seeds = X[crit2*crit1] + Y[crit2*crit1]*Lx
        new_seeds = new_seeds.tolist()
        seeds = [x for x in new_seeds if x not in Q_mem]
        #check they have not been used yet
        Q_mem+=seeds
        Q+=seeds
        #print(len(Q))
        #add relevant new seeds to the seed bank
    return region
    
    
def make_image(elevation_image, original_shape, im_type = 0):
    '''Build elevation images from elevation projection of the point cloud.
    input: elevation_image [array of images] output of get_elevation, or subpart of it
            original_shape [array of images] full output of get_elevation
            im_type: 
                -0: maximal elevation image
                -1: minimal elevation image
                -2: elevation range image
                -3: accumulation image
                -4, 5, 6... image made of the supplementary information of the point cloud (labels, classes)
    output: image from the output of get_elevation function
            mask marking the non-zero pixels in this image, for reverse projection
    '''
    #original shape is needed because the image needs to respect the dimension of the whole point
    #cloud projection to be transferable to other operations with other images
    
    #image basis
    MINS = np.min(original_shape[:, 0:2], axis = 0)
    L_X, L_Y = np.max(original_shape[:, 0:2], axis = 0) - MINS    
    im = np.zeros((int(L_X+1.), int(L_Y+1.)))
    #mask to locate the points that have been updated
    #important for reverse projection
    mask = np.zeros(im.shape).astype(bool)
    im_type = int(im_type + 2) #which elevation image to build

    
    for i in range(elevation_image.shape[0]):
        x, y = int(elevation_image[i, 0]-MINS[0]), int(elevation_image[i, 1]-MINS[1])

        im[x, y] = elevation_image[i, im_type]
        mask[x, y] = True
        
    return im, mask

def make_binary(im_in, thresh, dtyp = 'int'):
    #Binarize an image given a threshold thresh. Output type can be specified
    im = np.copy(im_in)
    im[im <= thresh] = False
    im[im > thresh] = True
    if dtyp == 'int':
        return im.astype(int)
    else: 
        return im.astype('bool')
    
def image_to_2Dcloud(im, elevation_mask):
    #reverse projection: from image to list of index. Backward operation of "make image"
    im = im.reshape(-1, order = 'C')
    elevation_mask = elevation_mask.reshape(-1, order = 'C')
    
    mount_cloud = im[elevation_mask]
    
            
    return mount_cloud
        
            
def conv_2D_3D(mount_cloud, ground, reverse_proj, points):
    #backward operation of "get elevation"
    
    a, b, c = zip(*reverse_proj)
    N = np.max(mount_cloud)
    N = int(N)
    u = np.linspace(0, N, N + 1)
    np.random.shuffle(u)
    is_obj = np.zeros(points.shape[0])
    for i in range(len(mount_cloud)):
        ind = a[i]
        is_obj[ind] = u[mount_cloud[i]]
        if ground[i] == 1 and mount_cloud[i] != 0:
            potential_ground = c[i]
            #print(np.min(np.abs(points[potential_ground, 2]-zmin-ground_dilate[i])))
            #print(ground_dilate[i])
            #potential_ground = potential_ground[np.abs(points[potential_ground, 2]-zmin-ground_dilate[i]) > 0.4]
            #if potential_ground.shape[0]>0:
            #    print(potential_ground)
            is_obj[potential_ground] = u[0]
            #print(u[0])

    return is_obj