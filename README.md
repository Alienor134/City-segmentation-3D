# City-segmentation-3D

Implementation: Python 3.7

'packages'
segmentation_func: useful functions for the projection, detection, segmentation and reverse projection pipeline
alienlab: useful functions for image display and saving

'content'
Detec_Segm_0: Pipeline described in the reference article (with slight modifications)
Detec_Segm_Loop: Supervised segmentation of trees using class ground truth

Change the file_path in the programs for other point clouds 
(here it works automatically for file_path = 'Cassette_idclass/Cassette_GT.ply'
(Rue Soufflot point cloud))

Remark: 
-the function to get the elevation is quite slow and could be speed up using numpy built in functions like np.unique

-If it doesn't work properly on other point clouds:
Parameters that could be tunable:
-kx and ky voxel parameter size
If ground search fails:
-Initial seed selection for the ground search (in segmentation_func, lambdaflat2)
-Elevation criterion ground search
-Neighbourhood size ground search
If the computation of the performances fails: 
-check how are named the ground truth fields in the point cloud