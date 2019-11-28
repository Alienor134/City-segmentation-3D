# -*- coding: utf-8 -*-


"""
Created on Thu Feb 14 22:29:53 2019

@author: Alienor Lahlou
"""


"""OPEN FILE"""

import os
cwd = os.getcwd()
def catch_file(direc = cwd ):
    """Opens a dialog window to select a file. Default directory: current directory
    direc [str]: directory to open (default: current directory)
    """
    
    from PyQt5 import QtWidgets
    fname = QtWidgets.QaFileDialog.getOpenFileName(None, directory=direc, caption = "Select a video file...",
                                                  filter="All files (*)")
    return fname[0]

"""PLOTS"""

import matplotlib.pyplot as plt
import datetime

class plotclass(object):
    """This class contains functions that can plot graphs and several curves on a graph (and save the plot)
    Input: x [array or list/tuple of arrays]: x axis values, either one array or multiple arrays
    Input: y [array or list/tuple of arrays]: x axis values, either one array or multiple arrays
    Output: plot f(x) = y, or overlayed curves f(xi) = yi"""
        
    def __init__(self, xval=None, yval=None):
        #plot parameters
        self.figsize = (9, 6)
        self.fontsize = 13
        self.color = 'steelblue'
        self.marker = 'o-'
        self.linewidth = 2
        self.title = 'My Title'
        self.xlabel = 'x label (unit)'
        self.ylabel = 'y label (unit)'
        
        #saving parameters
        self.date = True
        self.save_name = 'Figure'
        self.extension = '.tiff'
        
        #multiplot parameters
        self.label_item = ['MyLabel']
        self.label_list = self.label_item * 100
        self.color_list = [self.color] + ['indianred', 'seagreen', 'mediumslateblue', 'maroon', 'palevioletred'
                          'orange', 'lightseagreen', 'dimgrey', 'slateblue']


        self.xval = xval
        self.yval = yval
    
    def plotting(self):
        if type(self.xval) != tuple and type(self.xval) != list: #converts to a list if there is only one element
            self.xval = [self.xval]
        
        if type(self.yval) != tuple and type(self.yval) != list: #converts to a list if there is only one element
            self.yval = [self.yval]
            
        NX = len(self.xval)
        NY = len(self.yval)
        if NX != NY:
            if NX != 1:
                print('OooOouups! X should be a list or tuple containing either 1 array or the same number of arrays as Y')
                return False
            else: 
                self.xval = self.xval * NY #extends the X list to match the size of the Y list
        
        f = plt.figure(figsize = self.figsize)
   
        for i in range(NY):
            plt.title(self.title, fontsize = self.fontsize + 2)
            plt.xlabel(self.xlabel, fontsize = self.fontsize)
            plt.ylabel(self.ylabel, fontsize = self.fontsize)
            plt.plot(self.xval[i], self.yval[i], self.marker, color = self.color_list[i],
                    linewidth = self.linewidth, label = self.label_list[i]) #overlays new curve on the plot
        if NY > 1:
            plt.legend()
        if self.date == True:            
            plt.savefig(str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')) + self.save_name + self.extension)
        else: 
            plt.savefig(self.save_name + self.extension)
        return f
    


class showclass(object):
    """This class contains functions that can show images and subplot several images (and save the plot)
    Input: x [array or list/tuple of arrays]: images to plot
    Output: plot of the image x or subplots of images xi"""

        
    def __init__(self):
        #imshow parameters
        self.figsize = (9, 6) #figure size
        self.fontsize = 13  #font size
        self.title = 'My Title' #figure title
        self.cmap = 'inferno'

        #multiple image imshow        
        self.title_item = ['MyLabel']
        self.title_list = self.title_item * 100
        self.col_num = 3
       
        #figure save parameters
        self.date = True #write date and time before figure name
        self.save_name = 'Figure' #setting the format!
        self.extension = '.tiff'

    def showing(self, x=None, y=None):
        if type(x) != tuple and type(x) != list: #When there is only one image, convert it in a list element
            x = [x]
            
        N = len(x)
        
        COLS = self.col_num
        if N == 1: #when there is only one image
            ROWS, COLS = 1, 1
        elif N%COLS == 0: #when its a multiple of the number of columns expected, no extra row should be added
            ROWS = N//COLS
        else: 
            ROWS = N//COLS + 1 #extra row for  remaining figures otherwise    
            
        f = plt.figure(figsize = self.figsize)
        
        for i in range(N):
            plt.subplot(ROWS, COLS, i+1)
            plt.imshow(x[i], cmap = self.cmap)
            plt.axis('off')
            plt.grid(False)
            plt.title(self.title_list[i], fontsize = self.fontsize) #update subfigure title
        
        if self.date == True:            
            plt.savefig(str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_')) + self.save_name + self.extension,
                        bbox_inches='tight', frameon = False) #save with the date and time befor the figure name
        else: 
            plt.savefig(self.save_name + self.extension, bbox_inches='tight', frameon = False)
        return f