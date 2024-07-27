import numpy as np

# Adjustable variables
N = 37
f = 0.00
A = 0.0
x_trans = 296
y_trans = 913

# Function to calculate coordinates
def calculate_coordinates(N, f, A, x_trans, y_trans):
    x3 = [2 * np.cos(np.sum([f * j for j in range(1, i-2)])) for i in range(2, N+1)]
    y3 = [2 * np.sin(np.sum([f * j for j in range(1, i-2)])) for i in range(2, N+1)]
    
    x2 = [x * np.cos(A) - y * np.sin(A) + x_trans for x, y in zip(x3, y3)]
    y2 = [x * np.sin(A) + y * np.cos(A) + y_trans for x, y in zip(x3, y3)]
    
    coordinates = list(zip(x2, y2))
    return coordinates

# Calculate and print coordinates
coordinates = calculate_coordinates(N, f, A, x_trans, y_trans)
for coord in coordinates:
    print("new_x_y"+f"{coord},")
