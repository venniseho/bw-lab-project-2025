import os
import numpy as np
# import matplotlib; matplotlib.use('agg') # if error saying AttributeError: 'FigureCanvasMac' object has no attribute 'renderer'

import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scipy.spatial import Voronoi
import cv2


def voronoi_finite_polygons_2d(vor, radius=None):
	"""
	Reconstruct infinite voronoi regions in a 2D diagram to finite
	regions.

	Parameters
	----------
	vor : Voronoi	
	    Input diagram
	radius : float, optional
	    Distance to 'points at infinity'.

	Returns
	-------
	regions : list of tuples
	    Indices of vertices in each revised Voronoi regions.
	vertices : list of tuples
	    Coordinates for revised Voronoi vertices. Same as coordinates
	    of input vertices, with 'points at infinity' appended to the
	    end.

	"""

	if vor.points.shape[1] != 2:
		raise ValueError("Requires 2D input")

	new_regions = []
	new_vertices = vor.vertices.tolist()

	center = vor.points.mean(axis=0)
	if radius is None:
		radius = vor.points.ptp().max()

	# Construct a map containing all ridges for a given point
	all_ridges = {}
	for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
		all_ridges.setdefault(p1, []).append((p2, v1, v2))
		all_ridges.setdefault(p2, []).append((p1, v1, v2))

	# Reconstruct infinite regions
	for p1, region in enumerate(vor.point_region):
		vertices = vor.regions[region]

		if all(v >= 0 for v in vertices):
			# finite region
			new_regions.append(vertices)
			continue

		# reconstruct a non-finite region
		ridges = all_ridges[p1]
		new_region = [v for v in vertices if v >= 0]

		for p2, v1, v2 in ridges:
			if v2 < 0:
				v1, v2 = v2, v1
			if v1 >= 0:
				# finite ridge: already in the region
				continue

			# Compute the missing endpoint of an infinite ridge

			t = vor.points[p2] - vor.points[p1] # tangent
			t /= np.linalg.norm(t)
			n = np.array([-t[1], t[0]])  # normal

			midpoint = vor.points[[p1, p2]].mean(axis=0)
			direction = np.sign(np.dot(midpoint - center, n)) * n
			far_point = vor.vertices[v2] + direction * radius

			new_region.append(len(new_vertices))
			new_vertices.append(far_point.tolist())

		# sort region counterclockwise
		vs = np.asarray([new_vertices[v] for v in new_region])
		c = vs.mean(axis=0)
		angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
		new_region = np.array(new_region)[np.argsort(angles)]

		# finish
		new_regions.append(new_region.tolist())

	return new_regions, np.asarray(new_vertices)

# Generate Voronoi background image
def _solid_color_bg(img_size, color_arr=None, savefig_name=None, cmap_name="tab20b", return_colorUsed=False):
    H, W = int(img_size[0]), int(img_size[1])
    cmap = plt.get_cmap(cmap_name)
    ca = color_arr if color_arr is not None else cmap(range(1))
    color = ca[0]
    arr = (np.array(color[:3]) * np.ones((H, W, 3))).astype(float)

    if savefig_name is not None:
        plt.imsave(savefig_name, arr)

    if not return_colorUsed:
        return savefig_name if savefig_name is not None else None

    return savefig_name if savefig_name is not None else None, cmap_name, np.array([color])


# def genVoronoiBG(section_num=7, img_size=(400,400), random_seed=None, color_arr=None, savefig_name=None, return_colorUsed=False):
# 	# section_num is the min number found to fully divide the area
# 	# img_size = (height, width)

# 	# section_num == 1 => solid color path (quick workaround)
# 	if (section_num <= 1):
# 		return _solid_color_bg(img_size, color_arr=color_arr, savefig_name=savefig_name, return_colorUsed=return_colorUsed)
	
# 	random_seed = 1347 if random_seed is None else random_seed
# 	np.random.seed(random_seed)

# 	if (color_arr is not None): 
# 		# make sure there is alpha values specified 
# 		if (color_arr.ndim != 4):
# 			color_arr = np.concatenate((color_arr, np.ones(color_arr.shape[0]).reshape(-1, 1)), axis=1)

# 	H, W = int(img_size[0]), int(img_size[1])
 
# 	### resorting to some default values 
# 	color_maps = ['tab20b', 'tab20c'] 
# 	selected_cmap = color_maps[random_seed%len(color_maps)] 
# 	color_map = plt.get_cmap(selected_cmap)
		

# 	if (section_num > 1):
# 		# make up data points		
# 		points = np.random.randint(low=0, high=np.max(img_size), size=(section_num, 2)) 
# 		# compute Voronoi tesselation
# 		vor = Voronoi(points)
# 		# plot
# 		regions, vertices = voronoi_finite_polygons_2d(vor)

# 		# colorize
# 		my_norm = colors.Normalize(0, len(regions))
# 		color_arr = color_map(range(len(regions))) if color_arr is None else color_arr

# 		fig = plt.figure(figsize=(img_size[1]/1000, img_size[0]/1000), dpi=1000) # figsize= width first 
# 		plt.axis('off')
# 		for idx, region in enumerate(regions):
# 			polygon = vertices[region]

# 			color = color_arr[idx%(len(color_arr))] # if color_arr is not None else color_map.to_rgba(idx)
# 			plt.fill(*zip(*polygon), color=color) # , alpha=0.4
# 		plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
# 		plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

# 		fig.canvas.draw()
# 		buf = fig.canvas.buffer_rgba()
# 		np_array = np.asarray(buf)[:,:,:3]
# 		# np_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
# 		# 	fig.canvas.get_width_height()[::-1] + (3,)) 
# 		plt.close()



# 		blank_pixels_mask = np.all(np_array == [255, 255, 255], axis=-1)
# 		non_blank_pixels_mask = np.any(np_array != [255, 255, 255], axis=-1)  
# 		new_array = np.zeros_like(np_array)
# 		new_array[blank_pixels_mask] = [0, 0, 0]
# 		new_array[non_blank_pixels_mask] = [255,255,255]
# 		new_array = np.average(new_array, axis=2)

# 		coords = cv2.findNonZero(new_array) # Find all non-zero points (text)
# 		x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
# 		np_array = np_array[y:y+h, x:x+w, :] # Crop the image - note we do this on the original image


# 	else: # just have a canvas with one solid color
# 		regions = [0]
# 		selected_color = color_arr[np.random.randint(color_arr.shape[0])]
# 		np_array = selected_color[:3] * np.ones((img_size[0], img_size[1], 3))
# 		# edit color_arr so that the returned color used later will be correct 
# 		color_arr = np.insert(color_arr, 0, selected_color, axis=0)

# 	if (savefig_name is not None):
# 		plt.imsave(savefig_name, np_array)

# 	if not return_colorUsed:
# 		return savefig_name if savefig_name is not None else None 

# 	color_used = color_arr[:len(regions)] if color_arr is not None else np.array([color_map.to_rgba(idx) for idx in range(len(regions))])
# 	fig_return = savefig_name if savefig_name is not None else None 
# 	return fig_return, selected_cmap, color_used


def genVoronoiBG(section_num=7, img_size=(400,400), random_seed=None, color_arr=None, savefig_name=None, return_colorUsed=False):
	# section_num is the min number found to fully divide the area
	# img_size = (height, width)

	random_seed = 1347 if random_seed is None else random_seed
	np.random.seed(random_seed)

	if (color_arr is not None): 
		# make sure there is alpha values specified 
		if (color_arr.ndim != 4):
			color_arr = np.concatenate((color_arr, np.ones(color_arr.shape[0]).reshape(-1, 1)), axis=1)

	### resorting to some default values 
	color_maps = ['tab20b', 'tab20c'] 
	selected_cmap = color_maps[random_seed%len(color_maps)] 
	color_map = plt.get_cmap(selected_cmap)
		

	if (section_num > 1):
		# make up data points
		# --- TODO patch: try multiple times; fall back to solid color if Voronoi fails ---
		success = False
		for _ in range(15):
			points = np.random.randint(low=0, high=np.max(img_size), size=(section_num, 2))
			# remove accidental duplicates which can break Qhull
			points = np.unique(points, axis=0)
			if points.shape[0] < 4:
				continue
			try:
				vor = Voronoi(points)
				success = True
				break
			except Exception:
				continue
		if not success:
			# fall back to solid color branch below
			regions = [0]
			selected_color = color_arr[np.random.randint(color_arr.shape[0])] if color_arr is not None else np.array(plt.get_cmap(selected_cmap)(0))
			np_array = selected_color[:3] * np.ones((img_size[0], img_size[1], 3))
			if color_arr is not None:
				color_arr = np.insert(color_arr, 0, selected_color, axis=0)
			if (savefig_name is not None):
				plt.imsave(savefig_name, np_array)
			if not return_colorUsed:
				return savefig_name if savefig_name is not None else None 
			color_used = color_arr[:len(regions)] if color_arr is not None else np.array([plt.get_cmap(selected_cmap)(0)])
			fig_return = savefig_name if savefig_name is not None else None 
			return fig_return, selected_cmap, color_used
		# --- patch ---

		# compute Voronoi tesselation
		# vor already computed above
		# plot
		regions, vertices = voronoi_finite_polygons_2d(vor)

		# colorize
		my_norm = colors.Normalize(0, len(regions))
		color_arr = color_map(range(len(regions))) if color_arr is None else color_arr

		fig = plt.figure(figsize=(img_size[1]/1000, img_size[0]/1000), dpi=1000) # figsize= width first 
		plt.axis('off')
		for idx, region in enumerate(regions):
			polygon = vertices[region]

			color = color_arr[idx%(len(color_arr))] # if color_arr is not None else color_map.to_rgba(idx)
			plt.fill(*zip(*polygon), color=color) # , alpha=0.4
		plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
		plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

		fig.canvas.draw()
		buf = fig.canvas.buffer_rgba()
		np_array = np.asarray(buf)[:,:,:3]
		# np_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
		# 	fig.canvas.get_width_height()[::-1] + (3,)) 
		plt.close()



		blank_pixels_mask = np.all(np_array == [255, 255, 255], axis=-1)
		non_blank_pixels_mask = np.any(np_array != [255, 255, 255], axis=-1)  
		new_array = np.zeros_like(np_array)
		new_array[blank_pixels_mask] = [0, 0, 0]
		new_array[non_blank_pixels_mask] = [255,255,255]
		new_array = np.average(new_array, axis=2)

		coords = cv2.findNonZero(new_array) # Find all non-zero points (text)
		x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
		np_array = np_array[y:y+h, x:x+w, :] # Crop the image - note we do this on the original image


	else: # just have a canvas with one solid color
		regions = [0]
		selected_color = color_arr[np.random.randint(color_arr.shape[0])]
		np_array = selected_color[:3] * np.ones((img_size[0], img_size[1], 3))
		# edit color_arr so that the returned color used later will be correct 
		color_arr = np.insert(color_arr, 0, selected_color, axis=0)

	if (savefig_name is not None):
		plt.imsave(savefig_name, np_array)

	if not return_colorUsed:
		return savefig_name if savefig_name is not None else None 

	color_used = color_arr[:len(regions)] if color_arr is not None else np.array([color_map.to_rgba(idx) for idx in range(len(regions))])
	fig_return = savefig_name if savefig_name is not None else None 
	return fig_return, selected_cmap, color_used
