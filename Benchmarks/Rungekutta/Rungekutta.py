import numpy as np
#import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
#get_ipython().run_line_magic('matplotlib', 'notebook')
#import cv2
import re
import glob
import matplotlib as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import multiprocessing as mp
import psutil
def update_grid_cells_rk(t, cells, pins, D=0.3, T=0.2, decay_rate=0.000005, auxin_source=1000):
    
    """
    pins = [i,j,direction]
    direction index = [up,down,left,right]
    """
    
    d_cells = np.zeros(cells.shape)
    
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            value = cells[i,j]
            # check for spatial limits to evaluate diffusion 
            if i == 0 or i == cells.shape[0]-1:
                # first or last row of the grid
                if j == 0 or j == cells.shape[1]-1:
                    # top-left, top-right, bottom-left, bottom-right corner cells
                    if i == 0 and j == 0:
                        # top-left = D*(down_cell + right_cell - 2*self_cell) + T*(pins_up*down_cell + pins_left*right_cell 
                        #                                                          - pins_down*self_cell -pins_right*self_cell)
                        d_cells[i,j] = D*(cells[i+1,j] + cells[i,j+1] - 3*cells[i,j]) + T*(pins[i+1,j,0]*cells[i+1,j] + pins[i,j+1,2]*cells[i,j+1] -cells[i,j]*(pins[i,j,1]+pins[i,j,3]))
                        flag = "topleft"
                        d_cells[i,j] = d_cells[i,j] - (T*8*cells[i,j] + D*cells[i,j])
                    if i == 0 and j == cells.shape[1]-1:
                        # top-right = D*(left_cell + down_cell - 2*self_cell) + T*(pins_right*left_cell + pins_up*down_cell
                        #                                                          - self_cell*(pins_left + pins_down))
                        d_cells[i,j] = D*(cells[i,j-1] + cells[i+1,j] - 2*cells[i,j]) + T*(pins[i,j-1,3]*cells[i,j-1] + pins[i+1,j,0]*cells[i+1,j]-cells[i,j]*(pins[i,j,2] + pins[i,j,1]))
                        flag = "topright"
                        d_cells[i,j] = d_cells[i,j] - (T*8*cells[i,j] + D*cells[i,j])
                    if i == cells.shape[0]-1 and j == 0:
                        # bottom-left = D*(up_cell + right_cell - 2*self_cell) 
                        #               + T*(pins_left*rigth_cell + pins_down*up_cell
                        #                    - self_cell*(pins_right + pins_up))
                        d_cells[i,j] = D*(cells[i-1,j] + cells[i,j+1] -2*cells[i,j]) + T*(pins[i,j+1,2]*cells[i,j+1] + pins[i-1,j,1]*cells[i-1,j]-cells[i,j]*(pins[i,j,3] + pins[i,j,0]))
                        flag = "bottomleft"
                    if i == cells.shape[0]-1 and j == cells.shape[1]-1:
                        # bottom-right = D*(left_cell + up_cell - 2*self_cell) 
                        #                + T*(auxin_source + pins_right*left_cell + pins_down*up_cell
                        #                     - self_cell*(pins_left + pins_up))
                        d_cells[i,j] = D*(cells[i,j-1] + cells[i-1,j] - 2*cells[i,j]) + T*(pins[i,j-1,3]*cells[i,j-1] + pins[i-1,j,1]*cells[i-1,j]-cells[i,j]*(pins[i,j,2] + pins[i,j,0]))
                        flag = "bottomright"
                elif i == 0:
                    # non-corner cell on top row
                    # top-row-cell = D*(left_cell + down_cell + right cell - 3*self_cell) 
                    #                +T*(pins_right*left_cell + pins_up*down_cell + pins_left*right_cell
                    #                    - self_cell(pins_left + pins_down + pins_right))
                    d_cells[i,j] = D*(cells[i,j-1] + cells[i+1,j] + cells[i,j+1] - 4*cells[i,j]) + T*(pins[i,j-1,3]*cells[i,j-1] + pins[i+1,j,0]*cells[i+1,j] + pins[i,j+1,2]*cells[i,j+1]- cells[i,j]*(pins[i,j,2]+pins[i,j,1]+pins[i,j,3]))
                    flag = "topnoncorner"
                    if 2 <= j <= 7:
                        # We assume they receive auxin from above in qty auxin_source
                        d_cells[i,j] += auxin_source
                    else:
                        # We assume they have another cell on top on which they discharge some auxin
                        # d_cells[i,j] -= D*(cells[i,j])+T*(cells[i,j])
                        d_cells[i,j] = d_cells[i,j] - (T*8*cells[i,j] + D*cells[i,j])
                else:
                    # non-corner cell on bottom row
                    # bottom-row-cell = D*(left_cell + up_cell + right cell - 3*self_cell)
                    #                   + T*(pins_right*left_cell + pins_down*up_cell + pins_left*right_cell
                    #                        - self_cell*(pins_left + pins_up + pins_right))
                    d_cells[i,j] = D*(cells[i,j-1] + cells[i-1,j] + cells[i,j+1] - 3*cells[i,j]) + T*(pins[i,j-1,3]*cells[i,j-1] + pins[i-1,j,1]*cells[i-1,j] + pins[i,j+1,2]*cells[i,j+1]- cells[i,j]*(pins[i,j,2] + pins[i,j,0] + pins[i,j,3]))
                    flag = "bottomnoncorner"
            elif j == 0:
                # non-corner cases on leftmost column
                # left-row-cell = D*(up_cell + right_cell + down_cell - 3*self_cell) 
                #                 + T*(pins_down*up_cell + pins_left*right_cell + pins_up*down_cell
                #                      - self_cell*(pins_up + pins_right + pins_down))
                d_cells[i,j] = D*(cells[i-1,j] + cells[i,j+1] + cells[i+1,j] - 3*cells[i,j])+T*(pins[i-1,j,1]*cells[i-1,j] + pins[i,j+1,2]*cells[i,j+1] + pins[i+1,j,0]*cells[i+1,j] - cells[i,j]*(pins[i,j,0]+pins[i,j,3]+pins[i,j,1]))
                flag = "leftnoncorner"
            elif j == cells.shape[1]-1:
                # non-corner cases on rightmost column
                # right-row-cell = D*(up_cell + left_cell + down_cell - 3*self_cell) 
                #                  + T*(pins_down*up_cell + pins_left*right_cell + pins_up*down_cell
                #                       - self_cell*(pins_up + pins_left + pins_down))
                d_cells[i,j] = D*(cells[i-1,j] + cells[i,j-1] + cells[i+1,j] - 3*cells[i,j])+ T*(pins[i-1,j,1]*cells[i-1,j] + pins[i,j-1,3]*cells[i,j-1] + pins[i+1,j,0]*cells[i+1,j]- cells[i,j]*(pins[i,j,0]+pins[i,j,2]+pins[i,j,1]))
                flag = "rightnoncorner"
            else:
                # central-cell = D*(up_cell + down_cell + left_cell + right_cell - 4*self_cell) 
                #               + T*(pins_down*up_cell + pins_up*down_cell + pins_right*left_cell + pins_left*right_cell
                #                 - self_cell*(pins_up + pins_down + pins_left+ pins_right))
                d_cells[i,j] = D*(cells[i-1,j] + cells[i+1,j] + cells[i,j-1] + cells[i,j+1] - 4*cells[i,j])+T*(pins[i-1,j,1]*cells[i-1,j] + pins[i+1,j,0]*cells[i+1,j] + pins[i,j-1,3]*cells[i,j-1] + pins[i,j+1,2]*cells[i,j+1]- cells[i,j]*(pins[i,j,0]+pins[i,j,1]+pins[i,j,2]+pins[i,j,3]))
                flag = "central"
    
    # basal production
    #d_cells += 8*12*0.0005*4
    
    # basal decay
    #d_cells -= d_cells*decay_rate
    return d_cells

# Actual RK
def RK4(f):
    return lambda t, y, dt: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
            )( dt * f( t + dt  , y + dy3, pins   ) )
        )( dt * f( t + dt/2, y + dy2/2, pins ) )
        )( dt * f( t + dt/2, y + dy1/2, pins ) )
        )( dt * f( t       , y         ,pins) )


# Define root

# In[84]:


# Populate pins matrix with various cell types!

# up down left right
colummella = [8,8,12,12]
# if in future we want to add lateral pin we already have everything set
epidermal_l, epidermal_r = [8,0,0,0], [8,0,0,0] 
# if we want to add the last 5 cells to be longer this needs fix (grieneisen 2007 fig 1)
border_l, border_r = [0,8,0,12], [0,8,12,0]
vascular = [0,8,0,0]

# Set initial auxin concentrations
root = np.ones((20,10))
# root[0,5] = 10
pins = np.zeros((20,10,4))

# Locate the cells
pins[:-4,3:7,:] = vascular
pins[-4:,:,:] = colummella
pins[:-4,2,:] = border_l
pins[:-4,-3,:] = border_r
pins[:-4,:2,:] = epidermal_l
pins[:-4,-2:,:] = epidermal_r


# In[89]:


def alg_execution():
    root = np.ones((20,10))

    t, y, dt = 0.0, root, 0.05
    dy = RK4(update_grid_cells_rk)

    while t <= 120:
        if abs(round(t) - t) < 1:
            t, y = t + dt, y + dy( t, y, dt )
            
def monitor(target):
    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)

    # log cpu usage of `worker_process` every 10 ms
    memory_percents = []
    cpu_percents = []
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent())
        memory_percents.append(p.memory_percent())
        time.sleep(0.01)

    worker_process.join()
    return cpu_percents, memory_percents


start_time = time.time()

monitoring_cpu, monitoring_memory =  monitor(target=alg_execution) 

end_time = time.time()



def method_np_to_list_string(a):
    return "\t".join([item for item in a.astype(str)])

with open('RUNGEKUTTA_time.txt', 'w+') as time_log:
    time_log.write(f"Execution time {end_time-start_time}")



with open('RUNGEKUTTA_CPU_Usage.txt', "w+") as CPU_log:
    CPU_log.write('\t'.join([str(x) for x in monitoring_cpu]))
    CPU_log.write('\n')
    CPU_log.write(method_np_to_list_string(np.arange(0, 0.01*len(monitoring_cpu),0.01)))


with open("RUNGEKUTTA_Memory.txt", "w+") as Memory_logs:
    Memory_logs.write('\t'.join([str(y) for y in monitoring_memory]))
    Memory_logs.write('\n')
    Memory_logs.write(method_np_to_list_string(np.arange(0,0.01*len(monitoring_memory),0.01)))

