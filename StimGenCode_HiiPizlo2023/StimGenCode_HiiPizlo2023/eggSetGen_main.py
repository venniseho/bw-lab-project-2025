import cv2 
import numpy as np 
from matplotlib import pyplot as plt 

import json
import pandas as pd 
import os, time 
import copy
from pathlib import Path

import egg_dataset_helper as params_classes


''' for each of stim, create set, create 
	- color-control inside only, outside white
		- Baseline: color random inside only, outside white
		- level 2: color-control inside only, outside random sample, using color similar to vor_img
		- ** TO ADD level 3: color-control inside only, outside random sample using different scheme of color (change color_map *can't just use color_map of vor_imgBG since they may be the same)
			- expectation: level 2 and level 3 should be similar, but some may argue that mean value is different for level 2 and 3, may affect performance 
	- color-control outside only, inside white 
		- Baseline: color random outside only, inside white 
		- level 2: color random inside only, outside color controlled using color similar to vor_imgBG
		- ** TO ADD level 3: color random outside only, inside random sample, using different scheme of color (change color_map *can't just use color_map of vor_img since they may be the same)
	- color-control both inside and outside, with different pattern and different color_map 
'''



class SetGenerator:
	def __init__(self, max_stim_size=np.array([1080, 1920], int), num_trials_per_set=1, randomSeedNum=7431, output_dire=None):
		self.randomSeedNum = randomSeedNum
		self.num_trials_per_set = 1 #200 # N pointing left trials + N poitning right trials 

		# set parameters
		self.egg_distortion_factor = 0.04
		self.max_stim_size = max_stim_size ## image size 


		# for all different sets 
		self.range_gridSz = [40] 
		self.range_jitter = [0,20,90] ## 3 levels of jitter 
		self.range_edgeLenFactor = [0.6] 
		self.fixOutside = False ## if the egg should be placed with Fixation outside
		self.fixOutside_code = "all" # "all": every posible locations, "remaining": excluding equidistance, "equidistance": an annular ring 
		self.noise_offGrid = 0

		# within each set 
		self.egg_directions = [1, -1]
		self.range_lineThickness = [1]
		self.range_donutFactor = [1] 
		self.range_vorCode = [{"In":"W", "Out":"W"},{"In":"CC", "Out":"W"},{"In":"CC", "Out":"R"}]

		# setup logfile to save data 
		self.output_directory = f"{os.getcwd()}{os.sep}ClosureStim{os.sep}" if output_dire is None else output_dire
		if not os.path.exists(self.output_directory):
		    os.makedirs(self.output_directory)
	
		self.vor_df = pd.read_json(self.setupVorImg()) ## create color in Voronoi format
		curr_vor_type = "VorColor40-60" ## using glasbey_40_60

		for grid_sz in self.range_gridSz:
			self.curr_grid_sz = np.array([grid_sz, grid_sz], int)
			self.curr_grid_num = (np.array(self.max_stim_size) / grid_sz).astype(int)

			if (self.fixOutside):
				self.fixationCenters = self.fixOutside_cents(self.max_stim_size, self.curr_grid_num, noise_offGrid=self.noise_offGrid,
										returnCode=self.fixOutside_code)

			for jitter in self.range_jitter:
				curr_output_directory =  "%s%s%sGridSz%dJitt%d"%(self.output_directory, curr_vor_type, os.sep, self.curr_grid_sz[0], jitter) 
				self.generate_set(jitter=jitter,
					num_trials_per_set=1, randomSeedNum=self.randomSeedNum, output_dire=curr_output_directory)




	def setupVorImg(self): ## create color images to be used as masks to overlay color 
		vorImg_param = params_classes.ImageParam(output_directory=self.output_directory, 
			file_id=0, set_dire="", vorOnly=True)
		vor_log, log_filename = self.setup_IndividualLogFile(vorImg_param.vorImg_dire, get_fileName=True)

		for vor_iter in range(2*self.num_trials_per_set):	
			color_arr = np.array([[182,   0, 255], [  7, 172, 198], [148, 173, 132], 
				[211,   0, 140], [200, 111, 102], [169, 119, 173], [255,  40, 253], 
				[  1, 159, 125], [255,  98,   0], [255,  52, 101], [141, 155, 178], 
				[130, 145,  38], [188, 146,  88], [107, 133, 104], [146, 110,  86], 
				[173,  87, 125], [255,  74, 177], [194,  87,   3], [ 93, 140, 144], 
				[194,  68, 189]])/255.

			## Generate 2 color images as masks, using Usual voronoi
			section_num, curr_random_seed = np.random.randint(5,9), np.random.randint(0,59999)
			vorEgg_param = params_classes.VorParam(section_num, color_arr=color_arr, vor_imgsize=self.max_stim_size, 
				vor_randomseed=curr_random_seed, vor_figname="%s%s%d_egg.png"%(vorImg_param.vorImg_dire, os.sep, vor_iter))
			vorBG_param = params_classes.VorParam(2*section_num, color_arr=color_arr, vor_imgsize=self.max_stim_size, 
				vor_randomseed=curr_random_seed+1, vor_figname="%s%s%d_BG.png"%(vorImg_param.vorImg_dire, os.sep, vor_iter))
			## Usual voronoi

			vorEgg_dict = vorEgg_param.__dict__
			vorBG_dict = vorBG_param.__dict__
			vorEgg_dict, vorBG_dict = params_classes.serialize_np_for_json(vorEgg_dict), params_classes.serialize_np_for_json(vorBG_dict)
			del vorEgg_dict["vor_img"] # no need to save these in log_file
			del vorBG_dict["vor_img"] # no need to save these in log_file

			all_param_dict = {"Egg":vorEgg_dict, "BG":vorBG_dict}
			vor_log.write("\t{},\n".format(json.dumps(all_param_dict)))
			vor_log.flush()


		vor_log.truncate(vor_log.tell()-2) # windows: (change to) -3
		vor_log.write("\n]")
		vor_log.flush()
		vor_log.close()

		return log_filename

	def createDictStr_forSet(self, elf, vc, df, lt, offG):
		'''
			## code labeling to indicate the conditions generating the dataset
				ELF: edgeLenFactor (how close is each edge to its neighbor) 
				VC: VoronoiColor (color_insideEgg - color_outsideEgg)
				DF: DonutFactor if only the inside ring of egg should be colored. Decided to always use 1.0 to color the egg fully 
				LT: Line thickness
				offG: offGrid (old version of controlling how much the noise edges are shuffled off the grid, used another way to generate noise)
		'''
		return "ELF%dVC%s-%s_DF%d_LT%doffG%d"%(int(elf*10), vc["In"], vc["Out"], int(df*10), lt, int(offG*100)) 



	@staticmethod
	def setup_IndividualLogFile(curr_output_directory, get_fileName=False):
		log_filename = Path("%s%s%s"%(curr_output_directory, os.sep, "datasetLog.json"))

		log_file = open(log_filename, "a+")
		# check if log file is set up properly 
		log_file.seek(0) # go to the top of file 
		first_line = log_file.readline()
		if ("[" not in first_line):
			if (first_line == ""):
				# set up 
				log_file.write("[\n")
			else:
				raise RuntimeError("datasetGenerator.setup_logfile() log_file not in json format")
		else: # remove closed dict bracket at last 
			log_file.seek(0, 2)
			log_file.truncate(log_file.tell()-1)
		# go back to end of file to be appended 
		log_file.seek(0, 2)

		if (get_fileName):
			return log_file, log_filename
		return log_file


	def fixOutside_cents(self, curr_canvas_size, curr_grid_num, noise_offGrid, returnCode="equidistance"):
		### create sampling mask (region fixation outside)
		# create egg in center 
		fixationCenters = []
		vorEgg_param, vorBG_param = params_classes.VorParam(vor_dict=self.vor_df.iloc[0]["Egg"]), None # params_classes.VorParam(vor_dict=self.vor_df.iloc[0]["BG"])
		for egg_direction in [1, -1]: # separate masks for both directions 
			egg_param = params_classes.EggParam(distorting_factor=self.egg_distortion_factor, direction=egg_direction, egg_size=0.3, jitter=0)
			egg_center = np.array(curr_canvas_size)//2 # right in the middle  
			# -- shift horizontal only (Section 5.1.3) ----
			# randomly add shift of 20% in horizontal radius (the position of the egg was shifted either left or right randomly within a range equal to 20% of the major (horizontal) radius)
			# shiftX_dire, shiftX_max = 1 if np.random.rand() < 0.5 else -1,  0.2*np.max(curr_canvas_size)*egg_param.eggSize/2 
			# egg_center[1] = egg_center[1] + shiftX_dire * np.random.uniform() * shiftX_max

			# # # -- shift by a circle R1 & R2 (Section 6.1.3) ----
			# shift_max1 = 0.5*0.8*np.max(curr_canvas_size)*egg_param.egg_size/2 # 25% of minor radius 
			# shift_max = 0.5*np.min(curr_canvas_size)*egg_param.egg_size/2 # 25% of minor radius [matching to Kwon's size]
			# egg_center = params_classes.get_terminalPt(egg_center, direction=np.random.randint(360), dist_len=shift_max)

			# ## ==== visualize the egg centered and the regions R1 R2 
			start_angle = np.random.randint(360)
			canvas_param = params_classes.CanvasParam(canvas_size=curr_canvas_size, egg_ecc=0, egg_center=np.array(curr_canvas_size)//2, # not move egg
				egg_theta_StartEnd=(start_angle, start_angle+360), 
				line_thickness=1, edge_len_factor=0.6, # 1  
				grid_size=np.array([40,40]), tot_gridNum=curr_grid_num, noise_offGrid=noise_offGrid) 

			# a placeholder image_param for now 
			image_param = params_classes.ImageParam(placeholder=False)
			base_vorCode = {"In":"W", "Out":"W"}
			stim_param = params_classes.StimParam(egg_param, canvas_param, image_param, vorEgg_param, vorBG_param, 
				vorCode=base_vorCode, move_cent=0, donut_factor=1) # not move cent for now 
			stim_param.draw_colorCanvas(randomSeedNum=randomSeedNum, create_img=True)
			eggPixels = np.nonzero(stim_param.mask_img)

			region2avoid = 484
			circle1, circle2 = np.zeros(self.max_stim_size), np.zeros(self.max_stim_size)
			cv2.circle(circle1, egg_center[::-1], region2avoid, 255, -1)
			cv2.circle(circle2, egg_center[::-1], region2avoid+24, 255, -1)
			ellipseIdx3, ellipseIdx2 = np.nonzero(circle2), np.nonzero(circle1)

			ringedEllipseGray = np.zeros(self.max_stim_size)
			ringedEllipseGray[ellipseIdx3] = 1
			ringedEllipseGray[ellipseIdx2] = 0

			if (returnCode == "equidistance"):
				ringedEllipse_idx = np.nonzero(ringedEllipseGray)
				step_sizes = np.arange(0, ringedEllipse_idx[0].size, ringedEllipse_idx[0].size//400) # sample stepSize 
				fixationCenter = np.concatenate((ringedEllipse_idx[0][step_sizes].reshape(-1,1), ringedEllipse_idx[1][step_sizes].reshape(-1,1)), axis=1)

				fixationCenters.append(fixationCenter)


			else:

				ellipseMask = np.ones(self.max_stim_size) # everywhere is accessible 		

				if (returnCode == "all"):
					# just remove ellipseIdx2 
					ellipseMask[ellipseIdx2] = 0 # take out the central ellipse 

				if (returnCode == "remaining"):
					# remove ellipseIdx2 and ellipseIdx3 (equidistance)
					ellipseMask[ellipseIdx3] = 0 # ellipseIdx3 is larger than ellipseIdx2


				# Restrict outer, egg fixation center cannot be outside this region to fit the whole egg
				# eggSize y, x (322, 404), after some padding, 1.2* 404 = 484, 1.2*322 = 386
				topLeft, bottRight = (484//2, 386//2), (self.max_stim_size[::-1] - np.array([484//2, 386//2]))		
				max_centRect = np.zeros_like(ellipseMask)
				cv2.rectangle(max_centRect, topLeft, bottRight, 255, -1)

				combined_mask = np.bitwise_and(max_centRect.astype(np.uint8), ellipseMask.astype(np.uint8))
				combinedMask_idx = np.nonzero(combined_mask)

				## visualize allowed eggCent 
				# allowed_eggCent = np.zeros((*self.max_stim_size, 3))
				# allowed_eggCent[combinedMask_idx[0], combinedMask_idx[1], :] = (255, 0, 0)
				# allowed_eggCent[eggPixels[0], eggPixels[1], :] = (255, 255, 255) # add back the egg 
				# plt.imshow(allowed_eggCent) #, cmap=plt.cm.gray)
				# plt.title("Fixation outside allowed egg center locations")
				# plt.show()


				step_sizes = np.arange(0, combinedMask_idx[0].size, combinedMask_idx[0].size//400) # sample stepSize 
				fixationCenter = np.concatenate((combinedMask_idx[0][step_sizes].reshape(-1,1), combinedMask_idx[1][step_sizes].reshape(-1,1)), axis=1)
				fixationCenters.append(fixationCenter)
		return fixationCenters




	def generate_set(self, jitter=0, num_trials_per_set=1, randomSeedNum=7431, output_dire=None):
		np.random.seed(self.randomSeedNum)

		self.start_time = time.time()

		curr_count = -1
		for egg_dire_count, egg_direction in enumerate(self.egg_directions):
			# --- call set data generation 
			for n_iter in range(self.num_trials_per_set):
				curr_count += 1 

				vorEgg_param, vorBG_param = params_classes.VorParam(vor_dict=self.vor_df.iloc[curr_count]["Egg"]
					), params_classes.VorParam(vor_dict=self.vor_df.iloc[curr_count]["BG"])

				curr_random_seed = np.random.randint(0, 59999)
				self.create_one_set(curr_count, vorEgg_param, vorBG_param,
					self.curr_grid_sz, self.curr_grid_num, self.max_stim_size, 
					egg_direction, jitter, self.noise_offGrid, output_dire, randomSeedNum=curr_random_seed)
		print("datasetGenerator done")
		print("--- %s seconds ---" % (time.time() - self.start_time))
		for curr_log_file in self.log_files_arr:
			curr_log_file.truncate(curr_log_file.tell()-3) # windows -3? 
			curr_log_file.write("\n]")
			curr_log_file.flush()
			curr_log_file.close()



	def create_one_set(self, n_iter, vorEgg_param, vorBG_param,
		curr_grid_size, curr_grid_num, curr_canvas_size, egg_direction, jitter, noise_offGrid,
		curr_output_directory, randomSeedNum=1):
		if (n_iter == 0): 
			self.log_files_arr, self.image_param_arr = [], []

		egg_param = params_classes.EggParam(distorting_factor=self.egg_distortion_factor, direction=egg_direction, egg_size=0.3, jitter=jitter)
		egg_center = np.array(curr_canvas_size)//2 # right in the middle  
		# -- shift horizontal only (Section 5.1.3) ----
		# randomly add shift of 20% in horizontal radius (the position of the egg was shifted either left or right randomly within a range equal to 20% of the major (horizontal) radius)
		# shiftX_dire, shiftX_max = 1 if np.random.rand() < 0.5 else -1,  0.2*np.max(curr_canvas_size)*egg_param.eggSize/2 
		# egg_center[1] = egg_center[1] + shiftX_dire * np.random.uniform() * shiftX_max

		# # # -- shift by a circle R1 & R2 (Section 6.1.3) ----
		shift_max = 0.5*np.min(curr_canvas_size)*egg_param.egg_size/2 # 25% of minor radius [matching to Kwon's size]
		egg_center = params_classes.get_terminalPt(egg_center, direction=np.random.randint(360), dist_len=shift_max)

		if (self.fixOutside):
			egg_center_row = 0 if egg_direction == 1 else 1 
			egg_center = self.fixationCenters[egg_center_row][np.random.randint(self.fixationCenters[egg_center_row].shape[0])]

		len_df, len_lt, len_elf, len_vor = len(self.range_donutFactor), len(self.range_lineThickness), len(self.range_edgeLenFactor), len(self.range_vorCode)
		for df_counter, donut_factor in enumerate(self.range_donutFactor):
			for lt_counter, line_thickness in enumerate(self.range_lineThickness):
				np.random.seed(randomSeedNum) # so that all the noise edges are the same 

				for elf_counter, curr_elf in enumerate(self.range_edgeLenFactor):
					start_angle = np.random.randint(360)
					canvas_param = params_classes.CanvasParam(canvas_size=curr_canvas_size, egg_ecc=0, egg_center=egg_center, 
						egg_theta_StartEnd=(start_angle, start_angle+360), 
						line_thickness=line_thickness, edge_len_factor=curr_elf,  
						grid_size=curr_grid_size, tot_gridNum=curr_grid_num, noise_offGrid=noise_offGrid) 

					# a placeholder image_param for now 
					image_param = params_classes.ImageParam(placeholder=False)
					base_vorCode = {"In":"W", "Out":"W"}
					stim_param = params_classes.StimParam(egg_param, canvas_param, image_param, vorEgg_param, vorBG_param, 
						vorCode=base_vorCode, move_cent=0, donut_factor=donut_factor) # not move cent for now 
					stim_param.draw_colorCanvas(randomSeedNum=randomSeedNum, create_img=True)

					stim_img_arr, stim_eggOnly_arr, edge2Color_dict_arr = stim_param.createBy_vorCode(self.range_vorCode)
					if (n_iter == 0): 
						for vorCode_idx, vorCode in enumerate(self.range_vorCode):
							# add log
							set_dict_str = self.createDictStr_forSet(elf=curr_elf, vc=vorCode, df=donut_factor, lt=line_thickness, offG=noise_offGrid)
							self.image_param_arr.append(params_classes.ImageParam(output_directory=curr_output_directory, file_id=n_iter, set_dire=set_dict_str))
								
							log_file = self.setup_IndividualLogFile(self.image_param_arr[-1].compiled_directory)
							self.log_files_arr.append(log_file)


					for log_file_counter, (curr_stim_img, curr_stim_eggOnly,edge2Color_dict ) in enumerate(zip(stim_img_arr, stim_eggOnly_arr,edge2Color_dict_arr)):
						update_idx = (len_lt+len_elf+len_vor-1)*df_counter + (len_elf+len_vor-1)*lt_counter + len_vor*elf_counter + log_file_counter
						stim_param.stim_img, stim_param.stim_eggOnly = curr_stim_img, curr_stim_eggOnly
						stim_param.edge2Color_dict = edge2Color_dict_arr[update_idx]
						stim_param.image_param = self.image_param_arr[update_idx]
						stim_param.image_param.change_fileID(n_iter)
						SetGenerator.addData(stim_param, self.log_files_arr[update_idx])



	@staticmethod 
	def addData(stim_param, opened_logFile):
		''' write to data file '''

		np.save(stim_param.image_param.stim_egg_mat_path, stim_param.eggStim_mat)
		np.save(stim_param.image_param.stim_noiseIn_mat_path, stim_param.noiseMat_InsideEgg)
		np.save(stim_param.image_param.stim_noiseOut_mat_path, stim_param.noiseMat_OutsideEgg)


		with open(stim_param.image_param.edge2Color_dict_path, 'w') as f:
			f.write(str(stim_param.edge2Color_dict))
			f.flush()

		plt.imsave(stim_param.image_param.stim_path, stim_param.stim_img) 
		plt.imsave(stim_param.image_param.stim_eggOnly_path, stim_param.stim_eggOnly)

		egg_dict = stim_param.egg_param.__dict__
		canvas_dict = stim_param.canvas_param.__dict__
		image_dict = stim_param.image_param.__dict__
		vorEgg_dict = stim_param.vorEgg_param.__dict__
		vorBG_dict = stim_param.vorBG_param.__dict__
		vorCode_dict = stim_param.vorCode

		all_param_dict = {**egg_dict, **canvas_dict, **image_dict, 
			**vorEgg_dict, **vorBG_dict, **vorCode_dict}
		all_param_dict["move_cent"] = stim_param.move_cent

		for key in ["vor_img", "vor_mask", "mask_img", "canvas", "vor_img2", "vor_mask2"]:
			if key in all_param_dict:
				del all_param_dict[key] # these are too big, images have been saved as .png anyway

		all_param_dict = params_classes.serialize_np_for_json(all_param_dict)
		opened_logFile.write("\t{},\n".format(json.dumps(all_param_dict)))
		opened_logFile.flush()







if __name__ == "__main__":
	randomSeedNum = 456209845
	output_directory = f"ClosureStim{os.sep}" ## place directory name here
	setgen = SetGenerator(output_dire=output_directory)



