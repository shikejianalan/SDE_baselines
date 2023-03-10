import numpy as np
from numpy import genfromtxt
import random
import math
f = open("./in.txt","r") 
file_handle=open('output.txt',mode='w')
lines = f.readlines()
for line in lines:
    file_handle.write(f'~/thea/TheaDepsUnix/Source/TheaPrefix/bin/Thea/Browse3D -p 3 {line}.pts -f {line}.label --bg FFFFFF --fancy-points\n')

    
