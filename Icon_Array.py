from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

size_Icon = 20
size_Space = 5
i_num = 20
j_num = 50

w, h = (j_num*(size_Icon+size_Space))+20, (i_num*(size_Icon+size_Space))+20

K = 1000
N = 100
N_int = 110

Initial = 10;

data = np.zeros((h, w, 3), dtype=np.uint8)
i = 0
j = 0
tick = 0

i_pos = Initial
j_pos = Initial


while i <= i_num-1:
	while j <= j_num-1 :
		if tick <= N-1:
		
			data[i_pos:(i_pos+size_Icon), j_pos:(j_pos+size_Icon)] = [255, 0, 0]
			j_pos = j_pos+size_Icon
			data[i_pos:(i_pos+size_Icon), j_pos:(j_pos+size_Space)] = [0, 0, 0]
			j_pos = j_pos+size_Space
			j+=1
			
			tick+=1
			
		elif tick <= N_int-1:
		
			data[i_pos:(i_pos+size_Icon), j_pos:(j_pos+size_Icon)] = [255, 100, 0]
			j_pos = j_pos+size_Icon
			data[i_pos:(i_pos+size_Icon), j_pos:(j_pos+size_Space)] = [0, 0, 0]
			j_pos = j_pos+size_Space
			j+=1
			tick+=1
			
		else:
			data[i_pos:(i_pos+size_Icon), j_pos:(j_pos+size_Icon)] = [255, 255, 255]
			j_pos = j_pos+size_Icon
			data[i_pos:(i_pos+size_Icon), j_pos:(j_pos+size_Space)] = [0, 0, 0]
			j_pos = j_pos+size_Space
			j+=1
			tick+=1
			
			
	j=0		
	j_pos = Initial
	i_pos = i_pos+size_Icon+size_Space	
	i+=1
		
img = Image.fromarray(data, 'RGB')
img.show()