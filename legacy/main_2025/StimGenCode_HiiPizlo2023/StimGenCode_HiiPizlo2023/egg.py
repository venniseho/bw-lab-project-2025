import numpy as np  


def drawEgg_byEquation(egg_size=0.3, canvas_size=(225,225), distorting_coeff=0.16, direction=1, 
	egg_cent=None, theta_startEnd=None, slant=0):
	'''drawn from perspective of sweeping of X 
		-1 is left, 1 is right 
	'''

	rangeY, rangeX = (np.min(canvas_size))*egg_size, 1.25*(np.min(canvas_size))*egg_size

	numer_number = 1/(distorting_coeff**2)
	denom_number = 1/distorting_coeff
	curr_factor = (distorting_coeff/0.04)
	ySquared_coef = 0.0625*curr_factor
	constant = denom_number+curr_factor


	step_size = 0.0005
	if (direction == -1):
		# using solutions are more exact, so that step will always hit y =0, otherwise, can use +0.5 one side for each 0.04 increment of distortion, and make sure to fill in possible blank 
		X = np.arange(-0.5-np.sqrt(101)/2, np.sqrt(101)/2-0.5, step_size) if distorting_coeff==0.04 else np.arange(
			-1-np.sqrt(26), np.sqrt(26)-1, step_size) if distorting_coeff == 0.08 else np.arange(
			0.5*(-3-np.sqrt(109)), 0.5*(np.sqrt(109)-3), step_size) if distorting_coeff == 0.12 else np.arange(-50,50,step_size)
		
		# y_pos_half = 4*np.sqrt(X + 625/(X-25) + 26 ) # 16 from 1/0.0625 (for 0.04)
		y_pos_half = np.sqrt(1/ySquared_coef * (constant + X + numer_number/(X - denom_number)) )
	else:
		X = np.arange(0.5-(np.sqrt(101)/2), (np.sqrt(101)/2)+0.5, step_size) if distorting_coeff==0.04 else np.arange(
			1-np.sqrt(26), np.sqrt(26)+1, step_size) if distorting_coeff == 0.08 else np.arange(
			0.5*(3-np.sqrt(109)), 0.5*(np.sqrt(109)+3), step_size) if distorting_coeff == 0.12 else np.arange(-50,50,step_size)

		# y_pos_half = 4*np.sqrt(-1*X - 625/(X+25) + 26 ) # 16 from 1/0.0625 (for 0.04)
		y_pos_half = np.sqrt(1/ySquared_coef * (constant - X - numer_number/(X + denom_number)) )


	# remove nan values
	idx_notNaN = ~np.isnan(y_pos_half)
	y_pos_half = y_pos_half[idx_notNaN]
	X = X[idx_notNaN]
	
	# scale up egg, we know y = +- 4, when x = 0
	scaleup = np.min(canvas_size)*egg_size/(4*2) # *2 because top and bottom 
	X = X * scaleup
	y_pos_half = y_pos_half * scaleup

	ori_dtype = X.dtype
	top_halfXY = np.concatenate((X.reshape(-1,1), y_pos_half.reshape(-1,1)), axis=1)
	unique_rows = np.unique(top_halfXY, axis=0)
	X_plot, Y_plot = unique_rows[:, 0], unique_rows[:, 1]

	X_plot = np.concatenate((X_plot,X_plot[::-1]), axis=0).astype(ori_dtype)
	Y_plot = np.concatenate((Y_plot, (-1*Y_plot)[::-1]), axis=0)
	
	egg_cent = np.array(canvas_size)//2 if egg_cent is None else egg_cent
	X_plot, Y_plot = np.clip(X_plot+ egg_cent[1], 0, canvas_size[1]-1), np.clip(Y_plot+egg_cent[0],0,canvas_size[0]-1)

	if (theta_startEnd is not None): # loop the coordinates around (coord always starts from x = left)
		# translate theta to ind to loop around 
		ind2loop = int(np.floor(X_plot.size / 360 * theta_startEnd[0]))
		X_plot, Y_plot = np.concatenate((X_plot[ind2loop:], X_plot[:ind2loop])), np.concatenate((Y_plot[ind2loop:], Y_plot[:ind2loop]))

	return X_plot, Y_plot



def eggEcc_2_eggCent(eggEcc, canvas_size):
	''' Convert from eccentricity proportion (float [0,1]) to coordinate (float,float)[0,canvas_size] on canvas '''

	canvas_mid = np.array(canvas_size) // 2 
	orient = np.deg2rad(np.random.randint(0, 361)) # sample a orientation to move 
	distFromCent = eggEcc * np.max(canvas_size) //4 # //4 because half of canvas and half to take care of half of egg

	signY, signX = 1 if np.random.rand() > 0.5 else -1, 1 if np.random.rand() > 0.5 else -1 
	# calculate vector using canvas_mid as origin 
	egg_centY = canvas_mid[0] + (signY * distFromCent * np.sin(orient))
	egg_centX = canvas_mid[1] + (signX * distFromCent * np.cos(orient))

	# return the proportion 
	egg_centY, egg_centX = egg_centY/canvas_size[0], egg_centX/canvas_size[1]

	return (egg_centY, egg_centX)






