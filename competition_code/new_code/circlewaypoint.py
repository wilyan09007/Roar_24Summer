import math

def generate_quarter_circle(turn_angle, distance, start_x, start_y):
    points = [(start_x, start_y)]
    current_angle = turn_angle
    current_x = start_x
    current_y = start_y
    
    while current_angle < turn_angle + 90:
        # Calculate new point
        new_x = current_x + distance * math.cos(math.radians(current_angle))
        new_y = current_y + distance * math.sin(math.radians(current_angle))
        
        points.append((new_x, new_y))
        
        # Update current position and angle
        current_x, current_y = new_x, new_y
        current_angle += distance / (2 * math.pi * (distance / 90))  # Adjust the step to keep distance between points
    
    return points

# Example usage:
turn_angle = 0  # Starting angle
distance = 1  # Distance between points
start_x = 0  # Starting x-coordinate
start_y = 0  # Starting y-coordinate

quarter_circle_points = generate_quarter_circle(turn_angle, distance, start_x, start_y)
for point in quarter_circle_points:
    print(point)
