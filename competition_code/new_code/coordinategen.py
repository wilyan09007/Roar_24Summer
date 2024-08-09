import numpy as np
from decimal import Decimal, getcontext

getcontext().prec = 28

from mpmath import mp
precision = 100
mp.dps = precision

# Adjustable variables
N = 18
f = 0.01200000000
degree_offset = 0.996802677962
A = degree_offset * -0.07000000000000
x_trans = -305
y_trans = 332

# Function to calculate coordinates
def calculate_coordinates(N, f, A, x_trans, y_trans):
    x3 = []
    x3.append(0)
    for i in range(3, N+1):
        x3.append((1.9 * mp.cos(np.sum([f * j for j in range(1, i-2)])) + x3[i-3])+0.0211316456368*2.5)
        # print(x3[i-3])
    y3 = []
    y3.append(0)
    for i in range(3, N+1):
        y3.append(3  * mp.sin(np.sum([f * j for j in range(1, i-2)])) + y3[i-3])
        # print(y3[i-3])
    
    x2 = [x * mp.cos(A) - y * mp.sin(A) + x_trans for x, y in zip(x3, y3)]
    y2 = [x * mp.sin(A) + y * mp.cos(A) + y_trans for x, y in zip(x3, y3)]
    
    coordinates = list(zip(x2, y2))
    return coordinates

# Calculate and print coordinates
print()
print()
print()
print()
print()
print()
print()
coordinates = calculate_coordinates(N, f, A, x_trans, y_trans)
for coord in coordinates:   
    # print(f"{coord[0]}, {coord[1]}") #desmos
    print("new_x_y("+f"{coord[0]}, {coord[1]}),")        #add to program
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
print()
for coord in reversed(coordinates):
    # print(f"{coord[0]}, {coord[1]}") #desmos
    print("new_x_y("+f"{coord[0]}, {coord[1]}),")        #add to program
