import cv2 
import copy
import numpy as np 
from matplotlib import pyplot as plt 
import skimage.draw as skDraw 
import skimage.measure as skmeasure

import json
import os, time, sys
from pathlib import Path

import egg
import VoronoiBG


######## ========= general purpose functions ==========
dist_calc = lambda x1, y1, x2, y2: np.sqrt(np.square(x1-x2) + np.square(y1-y2))

def extend_line(p1, p2, distance=20):
    # Extend line given 2 points to a certain length, useful for filling a line to its border 
    # p1, p2: (x,y) coordinates 
    diff = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    midX, midY = 0.5*(p1[0] + p2[0]), 0.5*(p1[1] + p2[1])
    p3_x = int(midX + distance*np.cos(diff))
    p3_y = int(midY + distance*np.sin(diff))
    p4_x = int(midX - distance*np.cos(diff))
    p4_y = int(midY - distance*np.sin(diff))
    return np.array([p3_x, p3_y]), np.array([p4_x, p4_y])



def get_terminalPt(initialPt, direction, dist_len):
	# as a slingshot, sending initialPt in defined direction dist_len away
	# given a initialPt (x, y) and direction (deg), create a straight line 
	# and rotate in direction to create a terminal pt with defined len

	horizonPt = (initialPt[0] + dist_len, initialPt[1]) # make a horizontal line, with the other endpoint being the initialPt 
	# rotate edge by initialPt so that initialPt is not moved 
	rot_mat = cv2.getRotationMatrix2D((int(initialPt[0]),int(initialPt[1])), direction, scale=1.0)
	terminalPt = np.dot(rot_mat, np.array([horizonPt[0],horizonPt[1],1]).reshape(3,1)).reshape(-1,)

	return terminalPt


def fillArea_manualRaster(input_img, img_size, visualize=False):
	# https://stackoverflow.com/questions/41925853/fill-shapes-contours-using-numpy

	## adaptation
	if (input_img.ndim == 3): # transform into 2d 
		input_img_gray = np.average(input_img, axis=2)
		new_input_img = np.zeros(img_size, np.uint8)
		new_input_img[np.flatnonzero(input_img_gray)] = 255
		input_img = new_input_img # .astype(int)

	# if it is enclosed all around 
	area_filled_mat = np.maximum.accumulate(input_img, 1) &\
           np.maximum.accumulate(input_img[:, ::-1], 1)[:, ::-1] &\
           np.maximum.accumulate(input_img[::-1, :], 0)[::-1, :] &\
           np.maximum.accumulate(input_img, 0)

	return area_filled_mat


def viz_mat(edges_mat, canvas_size, connect=False, viz=False):
	vis_mat = np.zeros(canvas_size, np.uint8) 

	prev_x, prev_y = None, None 
	for x1,y1, x2,y2 in edges_mat:
		if (prev_x is not None):
			cv2.line(vis_mat, (int(prev_x),int(prev_y)), (int(x1),int(y1)), 255, 1)
		cv2.line(vis_mat, (int(x1),int(y1)), (int(x2),int(y2)), 255, 1)

		if (connect):
			prev_x, prev_y = x2, y2
	if (connect): # close off front back
		cv2.line(vis_mat, (int(prev_x),int(prev_y)), (int(edges_mat[0,0]),int(edges_mat[0,1])), 255, 1)
	
	if (viz):
		plt.imshow(vis_mat, cmap=plt.cm.gray)
		plt.title("viz_mat()")
		plt.show()

	return vis_mat


def serialize_np_for_json(in_dict):
	# Note: np array cannot be json serialized 
	for key, val in in_dict.items():
		if isinstance(val, np.ndarray ):
			in_dict[key] = val.tolist()
		if isinstance(val, (np.int64, np.int32)):
			in_dict[key] = int(val)

	return in_dict
######## ========= general purpose functions ==========

# --- NEW: utilities for turning a binary mask into line segments --- #
import cv2
import numpy as np
import skimage.measure as skmeasure

def _resample_polyline(points, step_px=4):
    """Resample a contour polyline to roughly uniform spacing (step_px)."""
    if len(points) < 2: 
        return points
    pts = np.array(points, dtype=float)
    d = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] == 0: 
        return pts
    new_s = np.arange(0, s[-1], step_px)
    new_pts = np.empty((len(new_s), 2), dtype=float)
    for i, t in enumerate(new_s):
        j = np.searchsorted(s, t) - 1
        j = np.clip(j, 0, len(pts)-2)
        r = (t - s[j]) / max(1e-6, (s[j+1] - s[j]))
        new_pts[i] = (1-r)*pts[j] + r*pts[j+1]
    return new_pts

def mask_to_contour_segments(mask, edge_len, gap_px_rng=(0,0), jitter_deg=0, line_thickness=1, change_len=None):
    """
    Convert a binary mask (H×W, 0/255 or bool) into a set of short line segments
    that follow its boundary, with optional random gaps/jitter — same spirit as addEggEdges2Canvas_fromEq.
    Returns: segments np.ndarray [N,4] of (x1,y1,x2,y2)
    """
    # ensure binary
    mask_u8 = (mask > 0).astype(np.uint8)
    # get outermost contour
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0,4))
    # pick the largest contour
    cont = max(contours, key=cv2.contourArea).reshape(-1,2)  # (N,2): (x,y)

    # resample to ~uniform spacing so “edge_len” is meaningful
    res = _resample_polyline(cont, step_px=2)
    if len(res) < 2:
        return np.zeros((0,4))

    segs = []
    i = 0
    L = len(res)
    rng = np.random.default_rng()
    while i+1 < L:
        # maybe skip a random gap
        if gap_px_rng and (gap_px_rng[1] > 0):
            gap = int(rng.integers(gap_px_rng[0], gap_px_rng[1]+1))
            i = min(i + gap, L-2)

        # pick span to approximate desired edge_len
        j = i+1
        run = 0.0
        while j < L and run < edge_len:
            run += np.linalg.norm(res[j] - res[j-1])
            j += 1
        if j >= L: 
            break

        x1,y1 = res[i]
        x2,y2 = res[j-1]

        # optional jitter (like StimParam.sample_jitter)
        if jitter_deg:
            mid = 0.5*(np.array([x1,y1])+np.array([x2,y2]))
            v = np.array([x2-x1, y2-y1])
            ang = np.deg2rad(np.random.randint(jitter_deg-5, jitter_deg+6)) * (1 if np.random.rand()<0.5 else -1)
            rot = np.array([[np.cos(ang), -np.sin(ang)],[np.sin(ang), np.cos(ang)]])
            v2 = rot @ v
            x1,y1 = (mid - 0.5*v2)
            x2,y2 = (mid + 0.5*v2)

        seg = np.array([x1,y1,x2,y2])

        # honor change_len (edge_len_factor) via existing helper
        seg = StimParam.clip2edgelen(seg.reshape(1,4), change_len if change_len is not None else int(edge_len))[0]

        segs.append(seg)
        i = j  # advance

    return np.array(segs, dtype=float)

# --- END NEW: utilities for turning a binary mask into line segments --- #




class EggParam:
	def __init__(self, distorting_factor=0.12, direction=1, egg_size=1/3, jitter=0):
		self.ellipse_axisY, self.ellipse_axisX = 5, 4 # constant 
		self.distorting_factor, self.direction, self.egg_size, self.jitter = distorting_factor, direction, egg_size, jitter

class CanvasParam:
	def __init__(self, canvas_size=(225,225), egg_ecc=0, egg_center=None, egg_theta_StartEnd=(90,450), 
		line_thickness=1, edge_len_factor=1, grid_size=(15,15), tot_gridNum=15, noise_offGrid=1.5):
		self.canvas_size, self.egg_ecc, self.egg_center, self.egg_theta_StartEnd = canvas_size, egg_ecc, egg_center, egg_theta_StartEnd
		self.line_thickness, self.edge_len_factor, self.grid_size, self.tot_gridNum, self.noise_offGrid = line_thickness, edge_len_factor, grid_size, tot_gridNum, noise_offGrid

		# if not taking up 100% of grid space, calc desired edge_len
		self.change_len = None if self.edge_len_factor==1 else self.edge_len_factor*np.min(self.grid_size)
		self.grid_margin = 0 if self.edge_len_factor==1 else int(np.ceil((1-self.edge_len_factor)*np.min(self.grid_size)/2))
		self.update_canvasSize() # make sure canvas size is following grid_size 

		# update based on eggEcc
		self.egg_center = egg.eggEcc_2_eggCent(self.egg_ecc, self.canvas_size) if self.egg_ecc != 0 else egg_center
		self.canvas = np.zeros(self.canvas_size, np.uint8)


	def update_canvasSize(self):
		# update canvas size based on tot_gridNum
		self.canvas_size = np.array(self.grid_size) * self.tot_gridNum


class ImageParam:
	def __init__(self, output_directory="", file_id=0, set_dire="", vorOnly=False, placeholder=False):
		self.output_directory, self.file_id = output_directory if output_directory != "" else os.getcwd(), file_id 
		self.compiled_directory = "%s%s%s"%(self.output_directory, os.sep, set_dire)
		self.vorOnly = vorOnly

		if (not self.vorOnly):
			self.stim_dire = "%s%s%s%s%s%s"%(self.output_directory, os.sep, set_dire, os.sep, "Stim", os.sep)
			self.stim_eggOnly_dire = "%s%s%s%s%s%s"%(self.output_directory, os.sep, set_dire, os.sep, "Stim_eggOnly", os.sep)
			self.stim_mat_dire = "%s%s%s%s%s%s"%(self.output_directory, os.sep, set_dire, os.sep, "Mat", os.sep)

			if (not placeholder):
				for dire in [self.stim_dire, self.stim_eggOnly_dire, self.stim_mat_dire]:
					if not os.path.exists(dire):
					    os.makedirs(dire)
         
		else:
			self.vorImg_dire = "%s%s%s%s"%(self.output_directory, os.sep, "Vor", os.sep)
			if (not placeholder):
				if not os.path.exists(self.vorImg_dire):
				    os.makedirs(self.vorImg_dire)
		
		if (not placeholder):
			self.change_fileID(self.file_id)


	def change_fileID(self, file_id):
		self.file_id = file_id

		if (not self.vorOnly):
			self.stim_path = "%s%s.png"%(self.stim_dire, self.file_id)
			self.stim_eggOnly_path = "%s%s.png"%(self.stim_eggOnly_dire, self.file_id)
			self.stim_egg_mat_path = "%s%segg.npy"%(self.stim_mat_dire, self.file_id)
			self.stim_noiseIn_mat_path, self.stim_noiseOut_mat_path = "%s%snoiseIn.npy"%(self.stim_mat_dire, self.file_id), "%s%snoiseOut.npy"%(self.stim_mat_dire, self.file_id)
			self.edge2Color_dict_path = "%s%sedge2Color_dict.txt"%(self.stim_mat_dire, self.file_id)
		else:
			self.stim_vor_path = "%s%s.npy"%(self.vorImg_dire, self.file_id)




class VorParam:
	def __init__(self, section_num=3, vor_imgsize=[200,200], vor_randomseed=None, vor_figname=None, vor_dict=None, color_arr=None):
		'''
			section_num: num of seed points to generate Vor
		'''
		self.section_num, self.vor_imgsize, self.vor_randomseed, self.vor_figname, self.color_arr = section_num, vor_imgsize, vor_randomseed, vor_figname, color_arr

		if (vor_dict is None):
			self.createVorImg()
		else:
			for key in vor_dict:
				setattr(self, key, vor_dict[key])

			self.vor_img = self.read_vorPath(self.vor_path, self.vor_imgsize[::-1])


		
	def read_vorPath(self, vor_path, img_size):
		vor_img = cv2.cvtColor(cv2.imread(vor_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
		# resize takes width first 
		return cv2.resize(vor_img, img_size[::-1], interpolation = cv2.INTER_AREA)

	def createVorImg(self):
		self.vor_img = 255*np.ones((*self.vor_imgsize[:2],3))

		vor_counter = 0
		while np.argwhere(np.all(self.vor_img == (255, 255, 255), axis=2)).size > 20:# vor_img.size*0.1: # if white is too much, redo
			if (vor_counter >=1 ): # if this is not the first loop 
				os.remove(self.vor_path)
    
			if self.vor_figname is None:
				self.vor_figname = "tmp_vor.png"

			self.vor_path, self.color_map, self.color_choices = VoronoiBG.genVoronoiBG(section_num=self.section_num, img_size=self.vor_imgsize, 
				random_seed=self.vor_randomseed, color_arr=self.color_arr, savefig_name=self.vor_figname, return_colorUsed=True)

			self.color_choices = self.color_choices[:, :3] # there were alpha values 
			self.vor_img = self.read_vorPath(self.vor_path, self.vor_imgsize[::-1])
			cv2.cvtColor(cv2.imread(self.vor_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
			# resize takes width first 
			self.vor_img = cv2.resize(self.vor_img, self.vor_imgsize[:2][::-1], interpolation = cv2.INTER_AREA)

			assert np.all(self.vor_img.shape[:2] == self.vor_imgsize[:2])

			vor_counter += 1 
			self.section_num += 1 

	def masking_Vor(self, mask_img, region_restricted):
		'''
		Given a mask_img and region_restricted, create a vor mask 
			mask_img: binary mask to restrict noise (always color white for foreground)
			region_restricted: "target" / "background"
		'''
		self.mask_img, self.region_restricted = mask_img, region_restricted
		vor_mask, _ = self.restrictNoisePattern(self.mask_img, self.vor_img, region_restricted)

		return vor_mask

	@staticmethod
	def donut_masking(mask_img, donut_factor):
		# make a donut shape for mask (the center color may not be important if color fill-in is working)
		if (donut_factor == 1): # no changes needed
			return mask_img

		# this create a smaller img_size
		mask_img_centerDonut = cv2.resize(mask_img.astype(np.uint8), (int(mask_img.shape[1]/donut_factor), int(mask_img.shape[0]/donut_factor))) # width first then height 
		resize_maskDonut = np.zeros(mask_img.shape, np.uint8)
		# paste the donut in, turning off the donut 
		egg_mask_y, egg_mask_x = np.nonzero(mask_img)
		top_left2paste = (np.array([np.average(egg_mask_y), np.average(egg_mask_x)]) - np.array(mask_img_centerDonut.shape[:2])//2).astype(int) # egg_cent - donut_cent
		resize_maskDonut[top_left2paste[0]: top_left2paste[0]+mask_img_centerDonut.shape[0], 
				 top_left2paste[1]: top_left2paste[1]+mask_img_centerDonut.shape[1]] = mask_img_centerDonut

		mask_img[np.nonzero(resize_maskDonut)] = 0
		return mask_img


	def change_color_involved(self, new_color_arr):
		ori_colors = np.unique(self.vor_img.reshape(-1, self.vor_img.shape[2]), axis=0)
		self.vor_path2, self.color_map2, self.color_choices2 = VoronoiBG.genVoronoiBG(section_num=self.section_num, img_size=self.vor_imgsize, 
			random_seed=self.vor_randomseed, color_arr=new_color_arr, savefig_name=self.vor_figname, return_colorUsed=True)

		self.color_choices2 = self.color_choices2[:, :3] # there were alpha values 
		self.vor_img2 = cv2.cvtColor(cv2.imread(self.vor_path2, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

		if hasattr(self, "mask_img"):
			self.vor_mask2 = self.masking_Vor(self.mask_img, self.region_restricted)

	def get_color_arr(self):
		return np.array(self.color_choices) # because self.color_used is 0-255, and after multiplying float with int, many values taht are close ot each other are generated 


	@staticmethod
	def restrictNoisePattern(restricted_img, mask_img, restricted):
		# mask_img is the noise pattern for hte current implementation
		# (optional, use mask_img if n/a): mask_img_preserve is the raw (undilated, thickness=1) version which generated the curr mask_img
		# restricted_img: img with target region filled 

		if (mask_img.ndim == 3) and (restricted_img.ndim != 3):
			restricted_img = np.dstack([restricted_img, restricted_img, restricted_img])
		if (restricted_img.ndim == 3) and (mask_img.ndim != 3):
			mask_img = np.dstack([mask_img, mask_img, mask_img])


		if (mask_img.shape != restricted_img.shape):
			mask_img = cv2.resize(mask_img, restricted_img.shape[:2][::-1], interpolation = cv2.INTER_AREA)

		# find the overlapping region between restricted_img and mask 
		restricted_indices = None
		if (restricted== "target"):
			restricted_indices = np.nonzero(np.bitwise_and(mask_img, restricted_img))
		elif(restricted == "background"):
			restricted_indices = np.nonzero(np.bitwise_and(mask_img, np.bitwise_not(restricted_img)))

		if (restricted_indices is not None):
			# update mask to restricted area
			new_mask_img = np.zeros(mask_img.shape, np.uint8)
			new_mask_img[restricted_indices[0], restricted_indices[1], :] = mask_img[restricted_indices[0], restricted_indices[1], :]

		# get colors in the restricted areas (flatten out row and col, retaining color channels)
		colors_involved = np.unique(mask_img[restricted_indices[0], restricted_indices[1], :].reshape(-1, mask_img.shape[2]), axis=0)

		return new_mask_img, colors_involved



class NoiseCanvas:
	def __init__(self, canvas_param, jitter=0):
		'''	
			gabor: instance of class Gabor holding parmeters for gabor 
			canvas_size: size of canvas
			grid_size: grid segmenting img into equal portions 
			addNoise: if uniform background noise should be added 	
		'''
		self.canvas_param, self.jitter = canvas_param, jitter
		self.img_height, self.img_width = self.canvas_param.canvas_size

		self.establishGrids()
		self.bin_gabor_dict = {} # store all gabors by their binned_grid_num and gabor midpoint 


	def establishGrids(self):
		self.grid_sizeY = self.canvas_param.grid_size[1] if self.canvas_param.grid_size[1] is not None else self.img_height//10 # default is dividing into 10 equal pieces
		self.grid_sizeX = self.canvas_param.grid_size[0] if self.canvas_param.grid_size[0] is not None else self.img_width//10
		# our limit: smallest gabor is size 11x11 pixels squared
		self.grid_sizeX, self.grid_sizeY = np.clip(self.grid_sizeX, a_min=11, a_max=None), np.clip(self.grid_sizeY, a_min=11, a_max=None)

		# ====== blind bin_edges, based on direct segmentation ========
		# bin edges, add in last bin right edge to conclude 
		self.bin_edgesY, self.bin_edgesX = np.arange(0, self.img_height+(self.img_height%self.grid_sizeY+1), self.grid_sizeY
			), np.arange(0, self.img_width+(self.img_width%self.grid_sizeX+1), self.grid_sizeX) 
		self.supportRatio_thres = np.min([self.grid_sizeY, self.grid_sizeX])//4 # min length for edge to be added 
		# ====== blind bin_edges, based on direct segmentation ========


	@staticmethod
	def gridInd2Endpts(x_ind, y_ind, grid_sizeX, grid_sizeY):
		# takes in indices of the bin of the grid, and their respectice sizes 
		# returns the (start_x, start_y, end_x, end_y)
		return x_ind*grid_sizeX, y_ind*grid_sizeY, (x_ind+1)*grid_sizeX, (y_ind+1)*grid_sizeY

	@staticmethod
	def gridInd2MidPix(x_ind, y_ind, grid_sizeX, grid_sizeY):
		# takes in indices of the bin of the grid, and their respectice sizes 
		# returns the mid point of the grid 
		return int(0.5*(x_ind*grid_sizeX + (x_ind+1)*grid_sizeX)), int(0.5*(y_ind*grid_sizeY + (y_ind+1)*grid_sizeY))

	@staticmethod
	def get_endpoints_from_mid(midY, midX, grid_sizeX, grid_sizeY, img_height, img_width):
		# calculate the endpoints given mid, forming grid_size 
		# use method for consistent treatment: startpoint is floor and endpoint is ceil
		start_end_y = (np.clip(midY-np.floor(grid_sizeY/2), 0, img_height).astype(int),
					   np.clip(midY + np.floor(grid_sizeY/2), 0, img_height).astype(int) )
		start_end_x = (np.clip(midX-np.floor(grid_sizeX/2), 0, img_width).astype(int),
					   np.clip(midX + np.floor(grid_sizeX/2), 0, img_width).astype(int) )

		return start_end_y, start_end_x


	def get_curr_patch(self, endpointsY, endpointsX, line_thickness=1):
		endpointsY, endpointsX = np.sort(endpointsY), np.sort(endpointsX)
		# curr grid size away (using grid margin if using gridsize % to control gap) [kwon dup description]
		curr_patch = self.canvas[endpointsY[0]-self.canvas_param.grid_margin//2:endpointsY[1]+self.canvas_param.grid_margin//2+1, 
		endpointsX[0]-self.canvas_param.grid_margin//2:endpointsX[1]+self.canvas_param.grid_margin//2+1]

		return curr_patch


	# ### Kwon 2016 duplicate
	def addNoise2Canvas(self, canvas2avoid=None, mask_img=None, randomSeedNum=None):
		# canvas2avoid: usually egg_edge_canvas. Edge-based canvas where edges in there should be avoided 
		if (randomSeedNum is not None):
			np.random.seed(randomSeedNum)
		self.canvas = canvas2avoid if canvas2avoid is not None else self.canvas # add in egg canvas so can avoid 


		noiseMat_InsideEgg, noiseMat_OutsideEgg = np.zeros((0,4)), np.zeros((0,4))
		for x_ind, gridX in enumerate(self.bin_edgesX[:-1]):
			for y_ind, gridY in enumerate(self.bin_edgesY[:-1]):
				# midX, midY from center of usual grid 
				midX, midY = self.gridInd2MidPix(x_ind, y_ind, self.grid_sizeX, self.grid_sizeY)

				# move center of noise edge by self.grid margin (20% if length is 60%)
				midX = int(midX + np.random.uniform(0, int(self.canvas_param.grid_margin))) if np.random.rand() < 0.5 else  int(midX - np.random.uniform(0, int(self.canvas_param.grid_margin))) 
				midY = int(midY + np.random.uniform(0, int(self.canvas_param.grid_margin))) if np.random.rand() < 0.5 else  int(midY - np.random.uniform(0, int(self.canvas_param.grid_margin)))

				# grid size should be respected 
				curr_orient = np.random.randint(360)
				edge_len = np.min(self.canvas_param.grid_size) # *self.canvas_param.edge_len_factor

				x1,y1, x2,y2 = midX-1, midY, midX+1, midY # create a horizontal edge 
				start_end_y, start_end_x = self.get_endpoints_from_mid(midY, midX, self.grid_sizeX, self.grid_sizeY, self.img_height, self.img_width)		
				curr_edgeImg, curr_edge = StimParam.create_edge_fromMid(midX, midY, start_xVal=start_end_x[0], start_yVal=start_end_y[0],
					edge_orient=curr_orient, edge_len=edge_len, curr_edge_mat=[x1,y1,x2,y2], img_size=self.canvas_param.grid_size, line_thickness=self.canvas_param.line_thickness)

				curr_edge, bool_inside = StimParam.create_edge_helper(
					curr_edge=curr_edge, jitter=0,  # since curr_orient would have handled it 
					line_thickness=self.canvas_param.line_thickness, change_len=self.canvas_param.change_len,
					ref_img=mask_img) 

				if isinstance(bool_inside, str): # it is halfway in halfway out 
					pass # dont add 
				else:
					curr_patch = self.get_curr_patch((curr_edge[-1,1], curr_edge[-1,3]), (curr_edge[-1,0], curr_edge[-1,2]), line_thickness=self.canvas_param.line_thickness)
					noise_criterion = np.all(curr_patch == 0) 

					if (noise_criterion):
						cv2.line(self.canvas, curr_edge[-1, :2], curr_edge[-1, 2:], 255, self.canvas_param.line_thickness)

						if (bool_inside):
							noiseMat_InsideEgg = np.concatenate((noiseMat_InsideEgg, curr_edge), axis=0)
						else:
							noiseMat_OutsideEgg = np.concatenate((noiseMat_OutsideEgg, curr_edge), axis=0)


		return noiseMat_InsideEgg, noiseMat_OutsideEgg




class StimParam:
	'''
		hold all parameters to generate a stimulus
		move_cent: 0 do not move; 1 move out, -1 move in 

		creates: self.egg_img, self.stim_img, self.stim_wGrid, self.stim_eggOnly
	'''

	# egg_majorA_size
	def __init__(self, egg_param=None, canvas_param=None, image_param=None, 
		vorEgg_param=None, vorBG_param=None, vorCode=None, move_cent=None,
		donut_factor=1):
		# use default if not given 
		self.egg_param = EggParam() if egg_param is None else egg_param
		self.canvas_param = CanvasParam() if canvas_param is None else canvas_param
		self.image_param = ImageParam() if image_param is None else image_param

		self.vorEgg_param, self.vorBG_param = vorEgg_param, vorBG_param
		self.vorCode = {"In": "CC", "Out": "W"} if vorCode is None else vorCode
		self.move_cent = move_cent
		self.donut_factor = donut_factor


	@staticmethod
	def clip2edgelen(edge_mat, edge_len, line_thickness=1, verbose=False):
		# clip from 2 sides to middle 
		new_edge_mat = np.zeros((0,4))
		for x1, y1, x2, y2 in edge_mat.astype(int):
			x_coord, y_coord = skDraw.line(x1,y1,x2,y2)
			_, x_idx = np.unique(x_coord.astype(int), return_index=True)
			_, y_idx = np.unique(y_coord.astype(int), return_index=True)
			idx2use = x_idx if x_idx.size >= y_idx.size else y_idx
			x_coord, y_coord = x_coord[idx2use], y_coord[idx2use]
			diff_len = x_coord.size - edge_len + (2*line_thickness)
						
			if (diff_len > 1):
				clip_front = int(diff_len//2 )
				clip_back = int(diff_len - clip_front)
				if (verbose):
					print("in clip2edgelen, diff_len > 1, edge_len, clip_front, clip_back", edge_len, clip_front, clip_back)
					print("before: ", [x_coord[0], y_coord[0], x_coord[-1], y_coord[-1]])
				x_coord, y_coord = x_coord[clip_front:x_coord.size-clip_back], y_coord[clip_front:y_coord.size-clip_back]

			if (verbose):
				print("after: ", [x_coord[0], y_coord[0], x_coord[-1], y_coord[-1]])
			new_edge_mat = np.concatenate((new_edge_mat, np.array([x_coord[0], y_coord[0], x_coord[-1], y_coord[-1]]).reshape(1,4)), axis=0)
		return new_edge_mat


	@staticmethod
	def create_edge_fromMid(midX, midY, start_xVal, start_yVal, edge_orient, edge_len, curr_edge_mat, img_size, line_thickness):
		midY_zeroed, midX_zeroed = midY - start_yVal, midX - start_xVal # zero it 
		rot_mat = cv2.getRotationMatrix2D((int(midX_zeroed), int(midY_zeroed)), edge_orient, scale=1.0)
		start_xy_rot, end_xy_rot = np.dot(rot_mat, np.array(
			[curr_edge_mat[0]-start_xVal, curr_edge_mat[1]-start_yVal, 1]).reshape(3,1)).reshape(-1,
		), np.dot(rot_mat, np.array(
			[curr_edge_mat[2]-start_xVal,curr_edge_mat[3]-start_yVal,1]).reshape(3,1)).reshape(-1,)

		(x1,y1), (x2,y2) = extend_line(start_xy_rot, end_xy_rot, distance=2*np.max(img_size))
		x1,y1, x2,y2 = StimParam.clip2edgelen(np.array([x1,y1,x2,y2]).reshape(1,4), edge_len)[0,:]

		# # Draw an image of the current edge (extended)
		edgeImg = np.zeros(img_size, np.uint8) # gabor_canvas.grid_size
		cv2.line(edgeImg, (int(x1), int(y1)), (int(x2),int(y2)), 255, line_thickness)
		x1,y1,x2,y2 = x1+start_xVal,y1+start_yVal,x2+start_xVal,y2+start_yVal

		return edgeImg, np.array([x1,y1,x2,y2]).reshape(-1,)


	@staticmethod
	def addColor(edges_mat, canvas, line_thickness=1, color_choices=None, ref_img=None):
		edge2Color_dict = {}

		for x1,y1,x2,y2 in edges_mat.astype(int):
			curr_color = None 
			if (ref_img is not None):
				mul_factor = 1 if np.ptp(ref_img.flatten()) > 1 else 255
				# resize ref_img to the same size of canvas 
				if (ref_img.shape != canvas.shape):
					ref_img = cv2.resize(ref_img, canvas.shape[:2][::-1], interpolation = cv2.INTER_AREA)

				x_coord, y_coord = skDraw.line(x1, y1, x2, y2) 
				x_coord, y_coord = np.clip(x_coord, a_min=0, a_max=ref_img.shape[1]-1), np.clip(y_coord, a_min=0, a_max=ref_img.shape[0]-1)
				curr_color = [int(mul_factor*x) for x in np.average(ref_img[y_coord, x_coord, :].reshape(-1, 3), axis=0)]
				
			elif (color_choices is not None):
				curr_color = [int(255*x) for x in color_choices[np.random.randint(0, len(color_choices))]]

			# need to convert each color individually to int for cv2.line to work 
			cv2.line(canvas, (int(x1),int(y1)), (int(x2),int(y2)), curr_color, line_thickness)
			edge2Color_dict[((x1,y1),(x2,y2))] = curr_color
		return canvas, edge2Color_dict



	def get_colorChoices_refImg(self, vorCode, region="In"):
		color_choices, ref_img = None, None 

		if (vorCode[region] == "W"):
			color_choices = np.array([[1.,1.,1.]]) 
		elif vorCode[region] == "R":
			color_choices = self.vorEgg_param.get_color_arr() if region=="In" else self.vorBG_param.get_color_arr()
		else:
			if (region == "In"):
				# assuming the previous base case alerady handled this 
				if not hasattr(self.vorEgg_param, "vor_mask") and hasattr(self, "mask_img"):
					self.vorEgg_param.vor_mask = self.vorEgg_param.masking_Vor(self.mask_img, "target")
				ref_img = self.vorEgg_param.vor_mask

			else: 
				if not hasattr(self.vorBG_param, "vor_mask") and hasattr(self, "mask_img"):
					self.vorBG_param.vor_mask = self.vorBG_param.masking_Vor(self.mask_img, "background")
				if (vorCode["Out"] == "PartialCamo"): # first change the colors involved 
					if not hasattr(self.vorBG_param, "vor_mask2"):
						self.vorBG_param.change_color_involved(self.vorEgg_param.get_color_arr())

					ref_img=self.vorBG_param.vor_mask2
				elif(vorCode["Out"] == "CC"):
					ref_img=self.vorBG_param.vor_mask

				elif(vorCode["Out"] == "CompleteCamo"):
					ref_img=self.vorEgg_param.vor_img


		return color_choices, ref_img


	def draw_colorCanvas(self, randomSeedNum=None, create_img=True):
		'''
			# first draw just egg contours, then based on the color-control, add in color for egg contours only
			# add in noise contours, skipping areas where there are egg contours 
			# then add in color in 2 separate steps using mask, once for inside egg and another time outside 

		'''

		# edges_canvas and edges_mat are generated the time of creating edges, they simplify operations for different addition 
		# - edges_canvas is easy for color-controlled "CC", and for bulk changes such as white, "W"
		# - edges_mat is used when all edges should take random color 

		# edges_canvas: drawable canvas or an array (for noise only)
		# edges_mat: matrix holding endpoints of edges or an array (for noise only)
		# curr_edge: "egg" or "noise" for egg contour or noise edges (both inside and outside egg)
		if (randomSeedNum is not None):
			np.random.seed(randomSeedNum)

		colored_canvas = np.zeros((*self.canvas_param.canvas_size[:2], 3), np.uint8)

		# draw egg 
		self.eggStim_mat = self.addEggEdges2Canvas_fromEq() 
		# create mask_img 
		mask_edgesImg = viz_mat(self.eggStim_mat, self.canvas_param.canvas_size, connect=True, viz=False)
		self.mask_img = fillArea_manualRaster(mask_edgesImg, img_size=mask_edgesImg.shape[:2]) # create Mask Img
		self.mask_img = self.vorEgg_param.donut_masking(self.mask_img, self.donut_factor) 

		if (create_img):
			# draw egg contour first 
			color_arr, ref_img = self.get_colorChoices_refImg(self.vorCode, "In")
			colored_canvas, self.edge2Color_dict = self.addColor(self.eggStim_mat, canvas=colored_canvas, line_thickness=self.canvas_param.line_thickness, 
				color_choices=color_arr, ref_img=ref_img) 
			self.stim_eggOnly = copy.deepcopy(colored_canvas)
		else: # just create enough for masking 
			color_arr, ref_img = self.get_colorChoices_refImg({"In":"W"}, "In")
			colored_canvas, self.edge2Color_dict = self.addColor(self.eggStim_mat, canvas=colored_canvas, line_thickness=self.canvas_param.line_thickness, 
				color_choices=color_arr, ref_img=ref_img) 
			self.stim_eggOnly = copy.deepcopy(colored_canvas)


		### ============ Noise edges ===============
		noise_canvas = NoiseCanvas(self.canvas_param, jitter=self.egg_param.jitter)
		self.noiseMat_InsideEgg, self.noiseMat_OutsideEgg = noise_canvas.addNoise2Canvas(canvas2avoid=np.average(self.stim_eggOnly,axis=2), mask_img=self.mask_img, randomSeedNum=randomSeedNum)

		if (create_img):
			colored_canvas, edge2Color_dict = self.addColor(self.noiseMat_InsideEgg, canvas=colored_canvas, line_thickness=self.canvas_param.line_thickness, 
				color_choices=color_arr, ref_img=ref_img) 
			self.edge2Color_dict.update(edge2Color_dict)

			# handle noise edges outside (handling "R" and "W" simultaneosly)
			color_arr, ref_img = self.get_colorChoices_refImg(self.vorCode, "Out")
			colored_canvas, edge2Color_dict = self.addColor(self.noiseMat_OutsideEgg, canvas=colored_canvas, line_thickness=self.canvas_param.line_thickness, 
				color_choices=color_arr, ref_img=ref_img) 
			self.edge2Color_dict.update(edge2Color_dict)
			self.stim_img = colored_canvas



	def createBy_vorCode(self, range_vorCode):
		stim_img_arr = [np.zeros((*self.canvas_param.canvas_size[:2], 3), np.uint8) for i in range(len(range_vorCode))]
		stim_eggOnly_arr = []
		edge2Color_dict_arr = [{} for i in range(len(range_vorCode))]


		# draw egg contour first 
		for edge in self.eggStim_mat:
			for vorCode_idx, vorCode in enumerate(range_vorCode):
				color_arr, ref_img = self.get_colorChoices_refImg(vorCode, "In")
				stim_img_arr[vorCode_idx], edge2Color_dict = self.addColor(edge.reshape(1,4), canvas=stim_img_arr[vorCode_idx], line_thickness=self.canvas_param.line_thickness, 
					color_choices=color_arr, ref_img=ref_img) 
				edge2Color_dict_arr[vorCode_idx].update(edge2Color_dict)
		for vorCode_idx, vorCode in enumerate(range_vorCode):
			stim_eggOnly_arr.append(copy.deepcopy(stim_img_arr[vorCode_idx]))


		### ============ Noise edges ===============
		for noise_mat, curr_region in zip([self.noiseMat_InsideEgg, self.noiseMat_OutsideEgg], ["In", "Out"]):
			for edge in noise_mat:
				for vorCode_idx, vorCode in enumerate(range_vorCode):
					color_arr, ref_img = self.get_colorChoices_refImg(vorCode, curr_region)
					stim_img_arr[vorCode_idx], edge2Color_dict = self.addColor(edge.reshape(1,4), canvas=stim_img_arr[vorCode_idx], line_thickness=self.canvas_param.line_thickness, 
						color_choices=color_arr, ref_img=ref_img) 
					edge2Color_dict_arr[vorCode_idx].update(edge2Color_dict)
		return stim_img_arr, stim_eggOnly_arr, edge2Color_dict_arr



	def addEggEdges2Canvas_fromEq(self): # creating gaps by restricting edge_len to 0.5*gridSz
		# declare constants 
		edge_len = np.min(self.canvas_param.grid_size) # 60% * grid size handled later  # self.canvas_param.edge2GridSz_proportion * 
		random_gap_range = [0, (1-self.canvas_param.edge_len_factor)/2*np.min(self.canvas_param.grid_size) ]
		# Create Egg 
		egg_center = np.array(self.canvas_param.canvas_size)//2 if self.canvas_param.egg_center is None else self.canvas_param.egg_center
		eggX, eggY = egg.drawEgg_byEquation(egg_size=self.egg_param.egg_size, canvas_size=self.canvas_param.canvas_size, 
				distorting_coeff=self.egg_param.distorting_factor, direction=self.egg_param.direction,
				egg_cent=egg_center, theta_startEnd=self.canvas_param.egg_theta_StartEnd, slant=0)


		egg_len_perStep = np.sqrt(np.diff(eggX)**2+np.diff(eggY)**2) # calculate length of segment of egg per increment in index for eggX, eggY
		egg_diff_cumSum = np.cumsum(egg_len_perStep) # length of contour if taken up to this position (since diff already +1 so can use ind directly)

		eggStim_mat = np.zeros((0,4))
		prev_ind, last_cumSum, counter = 0, 0, 0

		while True:
			curr_cumSum = egg_diff_cumSum[prev_ind:] - last_cumSum # update values (re-zero) based on what was used before 

			curr_gapLen = int(np.random.uniform(*random_gap_range)) if eggStim_mat.size != 0 else 0
			if (curr_gapLen != 0):
				# clip off gap 
				ind_gap = np.argmax(curr_cumSum >= curr_gapLen)
				prev_ind += ind_gap
				curr_cumSum = egg_diff_cumSum[prev_ind:] - egg_diff_cumSum[prev_ind] # update values (re-zero) based on gap

			# sample curr edge
			ind_end = np.argmax(curr_cumSum >= edge_len) 

			if (ind_end == 0):	
				break

			x1,y1,x2,y2 = eggX[prev_ind], eggY[prev_ind], eggX[prev_ind+ind_end], eggY[prev_ind+ind_end]
			eggStim_mat = self.create_edge_helper(
					curr_edge=np.array([x1,y1,x2,y2]), jitter=self.egg_param.jitter, 
					line_thickness=self.canvas_param.line_thickness, change_len=self.canvas_param.change_len, # (1-0.6)*min(self.canvas_param.grid_size), #
					ref_img=None, stim_mat=eggStim_mat) 

			prev_ind += ind_end
			last_cumSum = egg_diff_cumSum[prev_ind] # next slice start from here 
			counter += 1 
		return eggStim_mat


	@staticmethod
	def sample_jitter(jitter=0, jitter_range=5):
		if (jitter==90):
			jitter_amount = np.random.randint(0, 360) # sample a jitter value

		else:
			jitter_amount = np.random.randint(jitter-jitter_range, jitter+jitter_range+1) # sample a jitter value
		jitter_amount = jitter_amount if np.random.random() < 0.5 else -jitter_amount # direction of jitter (clockwise or counter-clockwise)

		return jitter_amount


	@staticmethod
	def create_edge_helper(curr_edge, jitter, line_thickness=1, change_len=None, 
		move_cent=0, move_counter=0, move_dist=0, ref_img=None, stim_mat=None, canvas=None):
		'''
			This function adds the current edge to the given matrix

			curr_edge: [1x4] holding [x1,y1,x2,y2]
			jitter: amount of jitter in deg 
			dist2cut: function of edge_len_factor, dist (px) to reduce edge length by (0 meaning no change)
			move_cent: another way of introducing noise other than jitter, testing the extend where region may be helpful 
				counter: paramter for move_cent, every other element can move_cent
				move_dist: how much to move
			ref_img: mask_img to use when creating noise_edges, returns True if inside egg_mask
			stim_mat: the matrix to append the current edge to 
		'''

		if (change_len is not None):
			curr_edge = StimParam.clip2edgelen(curr_edge.reshape(1,4), change_len)[-1,:]
		x1,y1, x2,y2 = curr_edge.astype(int)

		# move center of Gabor in direction of normal
		if (move_cent != 0): # can encode randomness in the amount if desired
			if (move_counter%2 == 0): # move_cent alternating element
				# find direction of normal
				normal_direction = edge_orient + (move_cent)*90 # moving out is +, moving in is -

				x1, y1 = get_terminalPt(initialPt=(x1, y1), direction=normal_direction, dist_len=move_dist)
				x2, y2 = get_terminalPt(initialPt=(x2, y2), direction=normal_direction, dist_len=move_dist)
				
		if (jitter != 0):
			vis_img = viz_mat(np.array([x1,y1,x2,y2]).reshape(1,4), canvas_size=(1080,1920))
			
			midX, midY = 0.5*(x1+x2), 0.5*(y1+y2)
			sampled_jitter = StimParam.sample_jitter(jitter)
			curr_len = dist_calc(x1,y1, x2,y2)
			
			if (jitter == 90):
				# start afresh
				curr_len = dist_calc(x1,y1, x2,y2)
				len_left = curr_len - curr_len//2
				x1,y1, x2,y2 = int(midX-curr_len//2), int(midY-curr_len//2), int(midX+len_left), int(midY+len_left) 

			# Angle is positive for anti-clockwise and negative for clockwise. 
			rot_mat = cv2.getRotationMatrix2D((int(midX), int(midY)), sampled_jitter, scale=1.0)
			(x1,y1),(x2,y2) = np.dot(rot_mat, np.array(
				[x1,y1,1]).reshape(3,1)).reshape(-1,), np.dot(rot_mat, np.array(
				[x2,y2,1]).reshape(3,1)).reshape(-1,)
			
			x1,y1, x2,y2 = StimParam.clip2edgelen(np.array([x1,y1,x2,y2]).reshape(1,4), curr_len)[0,:]
			x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

		if (stim_mat is not None):
			stim_mat = np.concatenate((stim_mat, np.array([x1,y1, x2,y2]).reshape(1,4)), axis=0)
		else: 
			stim_mat = np.array([x1,y1, x2,y2]).reshape(1,4)
		
		if (ref_img is not None):
			x_coord, y_coord = skDraw.line(x1, y1, x2, y2) 
			x_coord, y_coord = np.clip(x_coord, 0, ref_img.shape[1]-1), np.clip(y_coord, 0, ref_img.shape[0]-1)
				
			if np.all(ref_img[y_coord, x_coord] == 0):
				bool_inside = False 
			else:
				if np.all(ref_img[y_coord, x_coord] != 0):
					bool_inside = True
				else:
					bool_inside = "half"


		if (canvas is not None):
			color= (255,255,255) if canvas.ndim==3 else 255
			cv2.line(canvas, (int(x1),int(y1)), (int(x2),int(y2)), color, line_thickness)
			stim_mat = (stim_mat, canvas)
			
		if (ref_img is not None):
			return stim_mat, bool_inside
		else:
			return stim_mat

	# NEW: create edges from binary mask
	def addObjectEdgesFromMask(self, obj_mask):
		"""
		Build contour-following segments from a binary mask and store in self.eggStim_mat
		so the rest of the pipeline (masking, noise, coloring) stays unchanged.
		"""
		edge_len = np.min(self.canvas_param.grid_size)  # same as egg version
  
		# gap size derived from edge_len_factor for “fragmentation obviousness”
		gap_max = int((1 - self.canvas_param.edge_len_factor) * np.min(self.canvas_param.grid_size) / 2)
		segs = mask_to_contour_segments(
			obj_mask,
			edge_len=edge_len,
			gap_px_rng=(0, max(0, gap_max)),
			jitter_deg=self.egg_param.jitter,
			line_thickness=self.canvas_param.line_thickness,
			change_len=self.canvas_param.change_len,
		)
		if segs.size == 0:
			self.eggStim_mat = np.zeros((0,4))
			return self.eggStim_mat
		self.eggStim_mat = segs
  
		return self.eggStim_mat





