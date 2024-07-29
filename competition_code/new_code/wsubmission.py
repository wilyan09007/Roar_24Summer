"""
https://www.youtube.com/watch?v=EZyBBGg-VHY
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from collections import deque
from functools import reduce
from typing import List, Tuple, Dict, Optional
import math
import numpy as np
import roar_py_interface


def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def distance_p_to_p(p1: roar_py_interface.RoarPyWaypoint, p2: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(p2.location[:2] - p1.location[:2])

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

def new_x_y(x, y):
        new_location = np.array([x, y, 0])
        return roar_py_interface.RoarPyWaypoint(location=new_location, 
                                                roll_pitch_yaw=np.array([0,0,0]), 
                                                lane_width=5)

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        # self.maneuverable_waypoints = maneuverable_waypoints[:1962] + maneuverable_waypoints[1967:]
        # startInd = 1953
        480, 516
        831, 880

        startInd = [198, 478, 547, 691, 803, 884, 1287, 1508, 1854, 1968, 2264, 2662, 2770]
        endInd =  [198, 520, 547, 691, 803, 884, 1287, 1508, 1854, 1968, 2264, 2662, 2770]
        startInd_2 = 452
        endInd_2 = 565
        startInd_4 = 663
        endInd_4 = 742
        startInd_8 = 1800
        endInd_8 = 2006
        startInd_12 = 2586
        
        # temp[:startInd_2] + SEC_2_WAYPOINTS \
        #     + temp[endInd_2:startInd_4] + SEC_4_WAYPOINTS \
        #     + temp[endInd_4:startInd_8] + SEC_8_WAYPOINTS \
        #     + temp[endInd_8:startInd_12] \
        #     + SEC_12_WAYPOINTS
        temp = maneuverable_waypoints # so indexes don't change
        self.maneuverable_waypoints = \
        temp[:startInd_2] + SEC_2_WAYPOINTS \
            + temp[endInd_2:startInd_4] + SEC_4_WAYPOINTS \
            + temp[endInd_4:startInd_8] + SEC_8_WAYPOINTS \
            + temp[endInd_8:startInd_12] \
            + SEC_12_WAYPOINTS
        
        with open('out.txt', 'w') as f:
            for i in self.maneuverable_waypoints:
                print(f"{i.location[0]}, {i.location[1]}", file=f) # print current waypoints
        
        # self.maneuverable_waypoints = self.modified_points(maneuverable_waypoints)
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.lat_pid_controller = LatPIDController(config=self.get_lateral_pid_config())
        self.throttle_controller = ThrottleController()
        self.section_indeces = []
        self.num_ticks = 0
        self.section_start_ticks = 0
        self.current_section = -1

    async def initialize(self) -> None:
        num_sections = 12
        #indexes_per_section = len(self.maneuverable_waypoints) // num_sections
        #self.section_indeces = [indexes_per_section * i for i in range(0, num_sections)]
        # self.section_indeces = [198, 438, 547, 691, 803, 884, 1287, 1508, 1854, 1968, 2264, 2662, 2770]
        self.section_indeces = [198, 438, 547, 691, 803, 884, 1287, 1508, 1854, 1968, 2264, 2592, 2770]
        print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        print(f"indexes: {self.section_indeces}")

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

    # def modified_points(self, waypoints):
    #     new_points = []
    #     for ind, waypoint in enumerate(waypoints):
    #         if ind == 1964:
    #             new_points.append(self.new_x(waypoint, -151))
    #         elif ind == 1965:
    #             new_points.append(self.new_x(waypoint, -153))
    #         elif ind == 1966:
    #             new_points.append(self.new_x(waypoint, -155))
    #         else:
    #             new_points.append(waypoint)
    #     return new_points
        
    def modified_points_bad(self, waypoints):
        end_ind = 1964
        num_points = 50
        start_ind = end_ind - num_points
        shift_vector = np.array([0.5, 0, 0])
        step_vector = shift_vector / num_points

        s2 = 1965
        num_points2 = 150
        shift_vector2 = np.array([0, 2.0, 0])


        new_points = []
        for ind, waypoint in enumerate(waypoints):
            p = waypoint
            if ind >= start_ind and ind < end_ind:
                p = self.point_plus_vec(p, step_vector * (ind - start_ind))
            if ind >= s2 and ind < s2 + num_points2:
                 p = self.point_plus_vec(p, shift_vector2)
            new_points.append(p)
        return new_points

    def modified_points_good(self, waypoints):
        start_ind = 1920
        num_points = 100
        end_ind = start_ind + num_points
        shift_vector = np.array([2.8, 0, 0])
        step_vector = shift_vector / num_points

        s2 = 1965
        num_points2 = 150
        shift_vector2 = np.array([0, 3.5, 0])

        s3 = 1920
        num_points3 = 195
        shift_vector3 = np.array([0.0, 0, 0])

        new_points = []
        for ind, waypoint in enumerate(waypoints):
            p = waypoint
            if ind >= start_ind and ind < end_ind:
                p = self.point_plus_vec(p, step_vector * (end_ind - ind))
                # p = self.point_plus_vec(p, step_vector * (end_ind - ind))
            if ind >= s2 and ind < s2 + num_points2:
                p = self.point_plus_vec(p, shift_vector2)
            if ind >= s3 and ind < s3 + num_points3:
                p = self.point_plus_vec(p, shift_vector3)
            new_points.append(p)
        return new_points

    def point_plus_vec(self, waypoint, vector):
        new_location = waypoint.location + vector
        # new_location = np.array([waypoint.location[0], new_y, waypoint.location[2]])
        return roar_py_interface.RoarPyWaypoint(location=new_location,
                                                roll_pitch_yaw=waypoint.roll_pitch_yaw,
                                                lane_width=waypoint.lane_width)


    def modified_points_also_bad(self, waypoints):
        new_points = []
        for ind, waypoint in enumerate(waypoints):
            if ind >= 1962 and ind <= 2027:
                new_points.append(self.new_point(waypoint, self.new_y(waypoint.location[0])))
            else:
                new_points.append(waypoint)
        return new_points
    

    def new_x(self, waypoint, new_x):
        new_location = np.array([new_x, waypoint.location[1], waypoint.location[2]])
        return roar_py_interface.RoarPyWaypoint(location=new_location, 
                                                roll_pitch_yaw=waypoint.roll_pitch_yaw, 
                                                lane_width=waypoint.lane_width)
    def new_point(self, waypoint, new_y):
        new_location = np.array([waypoint.location[0], new_y, waypoint.location[2]])
        return roar_py_interface.RoarPyWaypoint(location=new_location, 
                                                roll_pitch_yaw=waypoint.roll_pitch_yaw, 
                                                lane_width=waypoint.lane_width)
    def new_y(self, x):

        y = -math.sqrt(102**2 - (x + 210)**2) - 962
        #print(str(x) + ',' + str(y))
        return y
        
        # a=0.000322627
        # b=2.73377
        # y = a * ( (abs(x + 206))**b ) - 1063.5
        # return y

    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        self.num_ticks += 1

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        current_speed_kmh = vehicle_velocity_norm * 3.6
        
        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

        # compute and print section timing
        for i, section_ind in enumerate(self.section_indeces):
            if section_ind -2 <= self.current_waypoint_idx \
                and self.current_waypoint_idx <= section_ind + 2 \
                    and i != self.current_section:
                elapsed_ticks = self.num_ticks - self.section_start_ticks
                self.section_start_ticks = self.num_ticks
                self.current_section = i
                print(f"Section {i}: {elapsed_ticks}")

        new_waypoint_index = self.get_lookahead_index(current_speed_kmh)
        waypoint_to_follow = self.next_waypoint_smooth(current_speed_kmh)
        #waypoint_to_follow = self.maneuverable_waypoints[new_waypoint_index]

        # Proportional controller to steer the vehicle
        steer_control = self.lat_pid_controller.run(
            vehicle_location, vehicle_rotation, current_speed_kmh, self.current_section, waypoint_to_follow)

        # Proportional controller to control the vehicle's speed
        waypoints_for_throttle = \
            (self.maneuverable_waypoints + self.maneuverable_waypoints)[new_waypoint_index:new_waypoint_index + 300]
        throttle, brake, gear = self.throttle_controller.run(
            self.current_waypoint_idx, waypoints_for_throttle, vehicle_location, current_speed_kmh, self.current_section)

        control = {
            "throttle": np.clip(throttle*1.5, 0.0, 1.0),
            "steer": np.clip(steer_control, -1.0, 1.0),
            "brake": np.clip(brake*1.5, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": gear
        }

        # print("--- " + str(throttle) + " " + str(brake) 
        #             + " steer " + str(steer_control)
        #             + " loc: " + str(vehicle_location)
        #             + " cur_ind: " + str(self.current_waypoint_idx)
        #             + " cur_sec: " + str(self.current_section)
        #             ) 


        await self.vehicle.apply_action(control)
        return control

    def get_lookahead_value(self, speed):
        speed_to_lookahead_dict = {
            70: 12,
            90: 14,
            110: 16,
            130: 18,
            160: 20,
            180: 22,
            200: 24,
            300: 24
        }
        num_waypoints = 3
        for speed_upper_bound, num_points in speed_to_lookahead_dict.items():
            if speed < speed_upper_bound:
              num_waypoints = num_points
              break
        # if self.current_section in [12]:
        #     num_waypoints = 8
            # num_waypoints = num_waypoints // 2
        return num_waypoints

    def get_lookahead_index(self, speed):
        num_waypoints = self.get_lookahead_value(speed)
        # print("speed " + str(speed) 
        #       + " cur_ind " + str(self.current_waypoint_idx) 
        #       + " num_points " + str(num_waypoints) 
        #       + " index " + str((self.current_waypoint_idx + num_waypoints) % len(self.maneuverable_waypoints)) )
        return (self.current_waypoint_idx + num_waypoints) % len(self.maneuverable_waypoints)
    
    def get_lateral_pid_config(self):
        conf = {
        "60": {
                "Kp": 0.8,
                "Kd": 0.05,
                "Ki": 0.05
        },
        "70": {
                "Kp": 0.7,
                "Kd": 0.07,
                "Ki": 0.07
        },
        "80": {
                "Kp": 0.66,
                "Kd": 0.08,
                "Ki": 0.08
        },
        "90": {
                "Kp": 0.63,
                "Kd": 0.09,
                "Ki": 0.09
        },
        "100": {
                "Kp": 0.6,
                "Kd": 0.1,
                "Ki": 0.1
        },
        "120": {
                "Kp": 0.52,
                "Kd": 0.1,
                "Ki": 0.1
        },
        "130": {
                "Kp": 0.51,
                "Kd": 0.1,
                "Ki": 0.09
        },
        "140": {
                "Kp": 0.45,
                "Kd": 0.1,
                "Ki": 0.09
        },
        "160": {
                "Kp": 0.4,
                "Kd": 0.05,
                "Ki": 0.06
        },
        "180": {
                "Kp": 0.28,
                "Kd": 0.02,
                "Ki": 0.05
        },
        "200": {
                "Kp": 0.28,
                "Kd": 0.03,
                "Ki": 0.04
        },
        "230": {
                "Kp": 0.26,
                "Kd": 0.04,
                "Ki": 0.05
        },
        "300": {
                "Kp": 0.205,
                "Kd": 0.008,
                "Ki": 0.017
        }
        }
        return conf

    # The idea and code for averaging points is from smooth_waypoint_following_local_planner.py
    def next_waypoint_smooth(self, current_speed: float):
        if current_speed > 70 and current_speed < 300:
            target_waypoint = self.average_point(current_speed)
        else:
            new_waypoint_index = self.get_lookahead_index(current_speed)
            target_waypoint = self.maneuverable_waypoints[new_waypoint_index]
        return target_waypoint

    def average_point(self, current_speed):
        next_waypoint_index = self.get_lookahead_index(current_speed)
        lookahead_value = self.get_lookahead_value(current_speed)
        num_points = lookahead_value * 2
        
        # if self.current_section in [12]:
        #     num_points = lookahead_value
        if self.current_section in [8,9]:
            # num_points = lookahead_value // 2
            num_points = lookahead_value * 2
            # num_points = lookahead_value
            # num_points = 1
        start_index_for_avg = (next_waypoint_index - (num_points // 2)) % len(self.maneuverable_waypoints)

        next_waypoint = self.maneuverable_waypoints[next_waypoint_index]
        next_location = next_waypoint.location
  
        sample_points = [(start_index_for_avg + i) % len(self.maneuverable_waypoints) for i in range(0, num_points)]
        if num_points > 3:
            location_sum = reduce(lambda x, y: x + y,
                                  (self.maneuverable_waypoints[i].location for i in sample_points))
            num_points = len(sample_points)
            new_location = location_sum / num_points
            shift_distance = np.linalg.norm(next_location - new_location)
            max_shift_distance = 2.0
            if self.current_section in [1,2]:
                max_shift_distance = 0.2
            if self.current_section in [6, 7]:
                max_shift_distance = 1.0
            if self.current_section in [8,9]:
                max_shift_distance = 2.8
            if self.current_section in [10,11]:
                max_shift_distance = 0.2
            if self.current_section in [12]:
                max_shift_distance = 0.4
            if shift_distance > max_shift_distance:
                uv = (new_location - next_location) / shift_distance
                new_location = next_location + uv*max_shift_distance

            target_waypoint = roar_py_interface.RoarPyWaypoint(location=new_location, 
                                                               roll_pitch_yaw=np.ndarray([0, 0, 0]), 
                                                               lane_width=0.0)
            # if next_waypoint_index > 1900 and next_waypoint_index < 2300:
            #   print("AVG: next_ind:" + str(next_waypoint_index) + " next_loc: " + str(next_location) 
            #       + " new_loc: " + str(new_location) + " shift:" + str(shift_distance)
            #       + " num_points: " + str(num_points) + " start_ind:" + str(start_index_for_avg)
            #       + " curr_speed: " + str(current_speed))

        else:
            target_waypoint =  self.maneuverable_waypoints[next_waypoint_index]

        return target_waypoint

class LatPIDController():
    def __init__(self, config: dict, dt: float = 0.05):
        self.config = config
        self.steering_boundary = (-1.0, 1.0)
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

    def run(self, vehicle_location, vehicle_rotation, current_speed, cur_section, next_waypoint) -> float:
        """
        Calculates a vector that represent where you are going.
        Args:
            next_waypoint ():
            **kwargs ():

        Returns:
            lat_control
        """
        # calculate a vector that represent where you are going
        v_begin = vehicle_location
        direction_vector = np.array([
            np.cos(normalize_rad(vehicle_rotation[2])),
            np.sin(normalize_rad(vehicle_rotation[2])),
            0])
        v_end = v_begin + direction_vector

        v_vec = np.array([(v_end[0] - v_begin[0]), (v_end[1] - v_begin[1]), 0])
        
        # calculate error projection
        w_vec = np.array(
            [
                next_waypoint.location[0] - v_begin[0],
                next_waypoint.location[1] - v_begin[1],
                0,
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(min(max(v_vec_normed @ w_vec_normed.T, -1), 1)) # makes sure arccos input is between -1 and 1, inclusive
        _cross = np.cross(v_vec_normed, w_vec_normed)

        if _cross[2] > 0:
            error *= -1
        self._error_buffer.append(error)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        k_p, k_d, k_i = self.find_k_values(cur_section, current_speed=current_speed, config=self.config)

        lat_control = float(
            np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        )
        
        # PIDFastController.sdprint("steer: " + str(lat_control) + " err" + str(error) + " k_p=" + str(k_p) + " de" + str(_de) + " k_d=" + str(k_d) 
        #     + " ie" + str(_ie) + " k_i=" + str(k_i) + " sum" + str(sum(self._error_buffer)))
        # print("cross " + str(_cross))
        # print("loc " + str(vehicle_location) + " rot "  + str(vehicle_rotation))
        # print(" next.loc " + str(next_waypoint.location))

        # print("steer: " + str(lat_control) + " speed: " + str(current_speed) + " err" + str(error) + " k_p=" + str(k_p) + " de" + str(_de) + " k_d=" + str(k_d) 
        #     + " ie" + str(_ie) + " k_i=" + str(k_i) + " sum" + str(sum(self._error_buffer)))
        # print("   err P " + str(k_p * error) + " D " + str(k_d * _de) + " I " + str(k_i * _ie))

        return lat_control
    
    def find_k_values(self, cur_section, current_speed: float, config: dict) -> np.array:
        k_p, k_d, k_i = 1, 0, 0
        if cur_section in [8, 9, 10, 11]:
        #   return np.array([0.3, 0.1, 0.25]) # ok for mu=1.2
        #   return np.array([0.2, 0.03, 0.15])
        #   return np.array([0.3, 0.06, 0.03]) # ok for mu=1.8
        #   return np.array([0.42, 0.05, 0.02]) # ok for mu=2.0
        #   return np.array([0.45, 0.05, 0.02]) # ok for mu=2.2
          return np.array([0.58, 0.05, 0.02]) # 
        # if cur_section in [12]:
        #   return np.array([0.4, 0.05, 0.02]) # 

        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        return np.array([k_p, k_d, k_i])

    
    def normalize_rad(rad : float):
        return (rad + np.pi) % (2 * np.pi) - np.pi

class SpeedData:
    def __init__(self, distance_to_section, current_speed, target_speed, recommended_speed):
        self.current_speed = current_speed
        self.distance_to_section = distance_to_section
        self.target_speed_at_distance = target_speed
        self.recommended_speed_now = recommended_speed
        self.speed_diff = current_speed - recommended_speed

class ThrottleController():
    display_debug = False
    debug_strings = deque(maxlen=1000)

    def __init__(self):
        self.max_radius = 10000
        self.max_speed = 300
        self.intended_target_distance = [0, 30, 60, 90, 120, 150, 180]
        self.target_distance = [0, 30, 60, 90, 120, 150, 180]
        self.close_index = 0
        self.mid_index = 1
        self.far_index = 2
        self.tick_counter = 0
        self.previous_speed = 1.0
        self.brake_ticks = 0

        # for testing how fast the car stops
        self.brake_test_counter = 0
        self.brake_test_in_progress = False

    def __del__(self):
        print("done")
        # for s in self.__class__.debug_strings:
        #     print(s)

    def run(self, cur_wp_index, waypoints, current_location, current_speed, current_section) -> (float, float, int):
        self.tick_counter += 1
        throttle, brake = self.get_throttle_and_brake(cur_wp_index, current_location, current_speed, current_section, waypoints)
        gear = max(1, (int)(current_speed / 60))
        if throttle == -1:
            gear = -1

        # self.dprint("--- " + str(throttle) + " " + str(brake) 
        #             + " steer " + str(steering)
        #             + "     loc x,z" + str(self.agent.vehicle.transform.location.x)
        #             + " " + str(self.agent.vehicle.transform.location.z)) 

        self.previous_speed = current_speed
        if self.brake_ticks > 0 and brake > 0:
            self.brake_ticks -= 1

        # throttle = 0.05 * (100 - current_speed)
        return throttle, brake, gear

    def get_throttle_and_brake(self, cur_wp_index, current_location, current_speed, current_section, waypoints):

        wp = self.get_next_interesting_waypoints(current_location, waypoints)
        r1 = self.get_radius(wp[self.close_index : self.close_index + 3])
        r2 = self.get_radius(wp[self.mid_index : self.mid_index + 3])
        r3 = self.get_radius(wp[self.far_index : self.far_index + 3])

        target_speed1 = self.get_target_speed(r1, current_section)
        target_speed2 = self.get_target_speed(r2, current_section)
        target_speed3 = self.get_target_speed(r3, current_section)

        close_distance = self.target_distance[self.close_index] + 3
        mid_distance = self.target_distance[self.mid_index]
        far_distance = self.target_distance[self.far_index]
        speed_data = []
        speed_data.append(self.speed_for_turn(close_distance, target_speed1, current_speed)) #distance curr target max
        speed_data.append(self.speed_for_turn(mid_distance, target_speed2, current_speed))
        speed_data.append(self.speed_for_turn(far_distance, target_speed3, current_speed))

        if current_speed > 100:
            # at high speed use larger spacing between points to look further ahead and detect wide turns.
            r4 = self.get_radius([wp[self.close_index], wp[self.close_index+3], wp[self.close_index+6]])
            target_speed4 = self.get_target_speed(r4, current_section)
            speed_data.append(self.speed_for_turn(close_distance, target_speed4, current_speed))

        update = self.select_speed(speed_data)
        # update = speed_data[2] # test

        # self.print_speed(" -- SPEED: ", 
        #                  speed_data[0].recommended_speed_now, 
        #                  speed_data[1].recommended_speed_now, 
        #                  speed_data[2].recommended_speed_now,
        #                  (0 if len(speed_data) < 4 else speed_data[3].recommended_speed_now), 
        #                  current_speed)

        t, b = self.speed_data_to_throttle_and_brake(update)
        self.dprint("--- (" + str(cur_wp_index) + ") throt " + str(t) + " brake " + str(b) + "---")
        return t, b

    def speed_data_to_throttle_and_brake(self, speed_data: SpeedData):
        percent_of_max = speed_data.current_speed / speed_data.recommended_speed_now

        # self.dprint("dist=" + str(round(speed_data.distance_to_section)) + " cs=" + str(round(speed_data.current_speed, 2)) 
        #             + " ts= " + str(round(speed_data.target_speed_at_distance, 2)) 
        #             + " maxs= " + str(round(speed_data.recommended_speed_now, 2)) + " pcnt= " + str(round(percent_of_max, 2)))

        percent_change_per_tick = 0.07 # speed drop for one time-tick of braking
        speed_up_threshold = 0.99
        throttle_decrease_multiple = 0.7
        throttle_increase_multiple = 1.25
        percent_speed_change = (speed_data.current_speed - self.previous_speed) / (self.previous_speed + 0.0001) # avoid division by zero

        if percent_of_max > 1:
            # Consider slowing down
            brake_threshold_multiplier = 1.0
            if speed_data.current_speed > 200:
                brake_threshold_multiplier = 1.0
            if percent_of_max > 1 + (brake_threshold_multiplier * percent_change_per_tick):
                if self.brake_ticks > 0:
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: counter" + str(self.brake_ticks))
                    return -1, 1
                # if speed is not decreasing fast, hit the brake.
                if self.brake_ticks <= 0 and not self.speed_dropping_fast(percent_change_per_tick, speed_data.current_speed):
                    # start braking, and set for how many ticks to brake
                    self.brake_ticks = math.floor((percent_of_max - 1) / percent_change_per_tick)
                    # TODO: try 
                    # self.brake_ticks = 1, or (1 or 2 but not more)
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: initiate counter" + str(self.brake_ticks))
                    return -1, 1
                else:
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: throttle early1: sp_ch=" + str(percent_speed_change))
                    self.brake_ticks = 0 # done slowing down. clear brake_ticks
                    return 1, 0
            else:
                if self.speed_dropping_fast(percent_change_per_tick, speed_data.current_speed):
                    # speed is already dropping fast, ok to throttle because the effect of throttle is delayed
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: throttle early2: sp_ch=" + str(percent_speed_change))
                    self.brake_ticks = 0 # done slowing down. clear brake_ticks
                    return 1, 0
                throttle_to_maintain = self.get_throttle_to_maintain_speed(speed_data.current_speed)
                if percent_of_max > 1.02 or percent_speed_change > (-percent_change_per_tick / 2):
                    self.dprint("tb: tick" + str(self.tick_counter) + " brake: throttle down: sp_ch=" + str(percent_speed_change))
                    return throttle_to_maintain * throttle_decrease_multiple, 0 # coast, to slow down
                else:
                    # self.dprint("tb: tick" + str(self.tick_counter) + " brake: throttle maintain: sp_ch=" + str(percent_speed_change))
                    return throttle_to_maintain, 0
        else:
            self.brake_ticks = 0 # done slowing down. clear brake_ticks
            # Consider speeding up
            if self.speed_dropping_fast(percent_change_per_tick, speed_data.current_speed):
                # speed is dropping fast, ok to throttle because the effect of throttle is delayed
                self.dprint("tb: tick" + str(self.tick_counter) + " throttle: full speed drop: sp_ch=" + str(percent_speed_change))
                return 1, 0
            if percent_of_max < speed_up_threshold:
                self.dprint("tb: tick" + str(self.tick_counter) + " throttle full: p_max=" + str(percent_of_max))
                return 1, 0
            throttle_to_maintain = self.get_throttle_to_maintain_speed(speed_data.current_speed)
            if percent_of_max < 0.98 or percent_speed_change < -0.01:
                self.dprint("tb: tick" + str(self.tick_counter) + " throttle up: sp_ch=" + str(percent_speed_change))
                return throttle_to_maintain * throttle_increase_multiple, 0 
            else:
                self.dprint("tb: tick" + str(self.tick_counter) + " throttle maintain: sp_ch=" + str(percent_speed_change))
                return throttle_to_maintain, 0

    # used to detect when speed is dropping due to brakes applied earlier. speed delta has a steep negative slope.
    def speed_dropping_fast(self, percent_change_per_tick: float, current_speed):
        percent_speed_change = (current_speed - self.previous_speed) / (self.previous_speed + 0.0001) # avoid division by zero
        return percent_speed_change < (-percent_change_per_tick / 2)

    # find speed_data with smallest recommended speed
    def select_speed(self, speed_data: [SpeedData]):
        min_speed = 1000
        index_of_min_speed = -1
        for i, sd in enumerate(speed_data): #distance curr target max
            if sd.recommended_speed_now < min_speed:
                min_speed = sd.recommended_speed_now
                index_of_min_speed = i

        if index_of_min_speed != -1:
            return speed_data[index_of_min_speed]
        else:
            return speed_data[0]
    
    def get_throttle_to_maintain_speed(self, current_speed: float):
        throttle = 0.6 + current_speed/1000
        return throttle

    def speed_for_turn(self, distance: float, target_speed: float, current_speed: float):
        d = (1/675) * (target_speed**2) + distance
        max_speed = math.sqrt(675 * d)
        return SpeedData(distance, current_speed, target_speed, max_speed)

    def speed_for_turn_fix_physics(self, distance: float, target_speed: float, current_speed: float):
        # fix physics
        braking_decceleration = 66.0 # try 11, 14, 56
        max_speed = math.sqrt((target_speed**2) + 2 * distance * (braking_decceleration + 9.81))
        return SpeedData(distance, current_speed, target_speed, max_speed)

    def get_next_interesting_waypoints(self, current_location, more_waypoints):
        # return a list of points with distances approximately as given 
        # in intended_target_distance[] from the current location.
        points = []
        dist = [] # for debugging
        start = roar_py_interface.RoarPyWaypoint(current_location, np.ndarray([0, 0, 0]), 0.0)
        # start = self.agent.vehicle.transform
        points.append(start)
        curr_dist = 0
        num_points = 0
        for p in more_waypoints:
            end = p
            num_points += 1
            # print("start " + str(start) + "\n- - - - -\n")
            # print("end " + str(end) +     "\n- - - - -\n")
            curr_dist += distance_p_to_p(start, end)
            # curr_dist += start.location.distance(end.location)
            if curr_dist > self.intended_target_distance[len(points)]:
                self.target_distance[len(points)] = curr_dist
                points.append(end)
                dist.append(curr_dist)
            start = end
            if len(points) >= len(self.target_distance):
                break

        self.dprint("wp dist " +  str(dist))
        return points

    def get_radius(self, wp):
        point1 = (wp[0].location[0], wp[0].location[1])
        point2 = (wp[1].location[0], wp[1].location[1])
        point3 = (wp[2].location[0], wp[2].location[1])

        # Calculating length of all three sides
        len_side_1 = round( math.dist(point1, point2), 3)
        len_side_2 = round( math.dist(point2, point3), 3)
        len_side_3 = round( math.dist(point1, point3), 3)
        small_num = 0.01
        if len_side_1 < small_num or len_side_2 < small_num or len_side_3 < small_num:
            return self.max_radius

        # sp is semi-perimeter
        sp = (len_side_1 + len_side_2 + len_side_3) / 2

        # Calculating area using Herons formula
        area_squared = sp * (sp - len_side_1) * (sp - len_side_2) * (sp - len_side_3)
        if area_squared < small_num:
            return self.max_radius
        # Calculating curvature using Menger curvature formula
        radius = (len_side_1 * len_side_2 * len_side_3) / (4 * math.sqrt(area_squared))
        return radius
    
    def get_target_speed(self, radius: float, current_section):
        if radius >= self.max_radius:
            return self.max_speed
        #self.section_indeces = [198, 438, 547, 691, 803, 884, 1287, 1508, 1854, 1968, 2264, 2662, 2770]
        #old section indeces = [0, 277, 554, 831, 1108, 1662, 1939, 2216, 2493]
        mu = 1.0 # SEE IF POSSIBLE TO SET SPEED TO LOOKAHEAD +-1 radius
        if current_section == 0: #CHNAGE TO ONLY SET SPEED FOR TURNS, OTHERWISE MAX SPEED, SO CHANGE SECTION DEFINITIONS, ALSO GO CHANGE ACCELERATION LIMIT BY SECTION, change braking style if possible
            mu = 10
        if current_section == 1:
            mu = 2.7
        if current_section == 2:
            mu = 10
        if current_section == 3:
            mu = 3
        if current_section == 4: #crash end of 4
            mu = 3
        if current_section == 5:
            mu = 3.2
        if current_section == 6:
            mu = 2.05
        if current_section == 7:
            mu = 1.2
        if current_section == 8:
            mu = 3.7
        if current_section == 9:
            mu = 10
        if current_section == 10: #3.8
            mu = 10
        if current_section == 11:
            mu = 2.2
        if current_section == 12:
            mu = 2.25
        '''old friction coefficients (goes with old sections): 
        if current_section == 6:
            mu = 1.1
        if current_section == 7:
            mu = 1.5
        if current_section == 9:
            mu = 1.5'''
        target_speed = math.sqrt(mu*9.81*radius) * 3.6
        return max(20, min(target_speed, self.max_speed))  # clamp between 20 and max_speed

    def print_speed(self, text: str, s1: float, s2: float, s3: float, s4: float, curr_s: float):
        self.dprint(text + " s1= " + str(round(s1, 2)) + " s2= " + str(round(s2, 2)) + " s3= " 
                    + str(round(s3, 2)) + " s4= " + str(round(s4, 2))
            + " cspeed= " + str(round(curr_s, 2)))

    # debug print
    def dprint(self, text):
        if self.display_debug:
            print(text)
            self.debug_strings.append(text)

# https://community.wolfram.com/groups/-/m/t/2963938 

SEC_2_WAYPOINTS = [ # f wider
    new_x_y(270.0, 907.0),
new_x_y(272.0511975518712469203129563022197502209425523761746826321165452640327736147175118052804049042224443, 907.0818289365759378425297348629986094528028166597009314892728355232594947231357522600984283836916318),
new_x_y(274.1023021007644117454670165342338270135897214859812623136462390297300262930099730485452732451805283, 907.1659559919179680303216785684833243924687598535757151268423076386732668813983829846234283238180229),
new_x_y(276.1532127155044030685731626137975407336613162164655173228888188926968868452964468083712200973851719, 907.2546789563339683709135051480657778513751623086267829558947772379296022766919699843841977573907636),
new_x_y(278.2038126104719057550613516453615334113378399788827673828703396010260232375256482641565437697430879, 907.3502949176308496679230466315965593156206035885046155703091650904810643831952006959750004763362869),
new_x_y(280.2539612236226839483033063563253520993134209164601659380311088722599247919666239986051117432838737, 907.4550997959190562089491038455862702625969112675060027442403341562940584866750174425075550396518491),
new_x_y(282.3034863018238022730128062487626782567174713195245521879797042066413680454036008568527307839142694, 907.5713877417325650474397661656358206418981992195412601097966899466464390086009340283424409264513723),
new_x_y(284.3521759976571928357070152310986747136207165114245464440895387658383959798199439720391429289616582, 907.7014503519968032989415273958895634733996482188177903217467173382021104977457208178121238772329633),
new_x_y(286.3997709833066478994364880922778994810229752395879935446094695663104522783860609162196588750011265, 907.8475756584801764114103411532879716858568028701309558637527708328278145136528771538146815561576636),
new_x_y(288.4459565889744077445626574804377558334065272474693186753160573717913756044659808820404120304295894, 908.0120468435224357463443092412862265705714421368487819114767584163453136342903860402389321758046449),
new_x_y(290.4903549754661954485294567906495597132730152609937445642646448269640358548434791902724788764023288, 908.1971406380638352645079689244705171137934262417914004234417072561288736800424730619967758026225798),
new_x_y(292.5325173531360845041502402699407995459235355629286796340571714538884183015906974903462069325087806, 908.4051253573255963076993932534676224033536350265531680940629207420137428325526783343043852163250445),
new_x_y(294.5719162622910488416045837309535015707776883066924685953887293941930817141150578289984930870334312, 908.6382585299409742950380617970505100428025869050341333189799664843786317300985062218826728968767628),
new_x_y(296.6079379334140006870222344026502010764904843025473618436596287338390174823009221889649244080484683, 908.8987840769372027068222037624496667339265110663760034448351928064818864131214810859118771826627029),
new_x_y(298.6398747491662401761726758135107029690734135281083297136291082707444988065803989606429436997455674, 909.1889289977553388444536138528054612376009710247621610444001192823686600140530541389436114953204429),
new_x_y(300.6669178340658773034084564085668348289721269088862216532515647007228210247483719258621988945223277, 909.5108995215045706960746931749986121515003205234423395559382848952583330625996072316661603188081748),
new_x_y(302.6881498019955183901784908591056201233856794094406504121050275581530116205161099391134696793594373, 909.8668766829201949822142071281406736022694921016532607334427314250187722557078499342910533240564458),
new_x_y(304.7025376962546290597265429206090204891012059697827115641753095796376055592530101423571256349806474, 910.2590112840737072663167743309463736334297657659998331157119358428626023418394371334739303382399086),
new_x_y(306.7089261617199628120360539096602961733370119454322690268388378394409418188400454406259650251521609, 910.6894182048156196737367980386009126094100343516505869368783540244467562553147570065687821946275553),
new_x_y(308.7060308937873494309599120841051820726360850389078810832630789628759183717077386430197146070637475, 911.1601700272657190868852389832041470439562831022224896174236111210520669464601956900679257132427269),
new_x_y(310.692432414111041442252675812327605906329672099728262232879569115403778247514870760192403908119285, 911.6732899424527383176608209163107420470089748273142569782960165847509556610153585425427301048868386),
new_x_y(312.6665702286981660692025760851498504488124412817399189458852663722982941956452900524980012269971985, 912.2307439104989128596039324347522276777714741247055826424400578475340009481699886370891492492459007),
new_x_y(314.6267374296148088382373280349637505459005448411619474291163389767896628576840302084894752807717061, 912.8344320495990405789868346807755694557861940945060090532234292960227110576355524524546324583776027),
new_x_y(316.5710758073691400102910835766856518630957110262751776753477293465863955181219984502104525520705365, 913.486179233513585273481652288393704296667031119170834053114790254152618084979534167665623615661716),
new_x_y(318.4975715469005144740509794253172567304031551158259719574404638772035096686367810982015666255946169, 914.187724882436233009978568977523943028266125947275161869371519343744940852893418490120690224218631),
new_x_y(320.4040515859581865972717310990621333276309762543945237057582781132709740791980362582680905372010664, 914.9407119379625033899070734583684830759826215127470889605047100484405315288500130858452886862699394),
new_x_y(322.2881807204269700082718582432335677820616219986366281684232378857131412420600749275570657360540085, 915.7466750195302208311729678079557251813119159579935040206045718640608687199561987622778050341925321),
new_x_y(324.1474595467683109626669429942121998423080123864495000554484847179162781794753093707692858477116541, 916.6070277671747799931244339996999396422231691280213487791807203444723685694472987010624415255777893),
new_x_y(325.9792233371024935309937280623585443239073123397405803086143245735886763473670278394337718471865763, 917.523049383788192931714761632485910124064791806703048504817278672415302856022820457063445681593095),
new_x_y(327.7806419474595130983536430905104347858543807435920712121243833173174522821792147149390359453852515, 918.4958703993316097940212912889305608539424884703009714883259028113931984533313885988212002132893059),
new_x_y(329.5487208642604806271760653700173588250821607279135923327919558501473580407037141352805528149431253, 919.5264576896603985703217643097269005861781704151294977541344726562902147240193366747306540757202832),
new_x_y(331.2803034980355150667261558769416979573731695997867369018134515644067630351249108231563166658585311, 920.6155987938046757054173600163728941005441342786557462571641310704173118139248137552298091467166799),
new_x_y(332.9720748366044989267143271435241308029064205489216772688705705721678684539548735730060018778772214, 921.7638855857220999769985060246672201992134921456220755514912656065939343997154158882860269211968097),
new_x_y(334.62056657229982953109530986513640821324410250799011471681029500271161241107672619945161285664897, 922.9716973697075755486010238888966063790903374954478129511610423448333606011609308829293611425602216),
new_x_y(336.2221638191412483931144012459053174085508184316370540857088501544254917336786173655237543304568342, 924.2391834827961707939847944276067211936509594831623513111385103412570185558241922813625287367176144),
new_x_y(337.773113536018276483759269635732469318681553605316981167320936457126090546194422913196481865023446, 925.5662455026049792503945002995127369003775733802206962533380967206557558721264557229132261464271475),
new_x_y(339.2695347707233811807990293030269489495467122402523032230725634888550840415930906406056795432093276, 926.9525191750825676424493746944533747870998938697707247921120347712732725240808055015856351679125837),
new_x_y(340.7074308369289488017387145754234502395317306426770901230389976902038966421931798562606206150047318, 928.3973561935063761336813109527791288826917923422612931818475957004028043940805463297034665654651689),
new_x_y(342.0827035317277320770673003113899143756547215902954732849429939105721601360474259385797415676447392, 929.8998059777014891997084803306199430523970138084731756424261686318069592933913746954836606359390167),
new_x_y(343.3911694949699896332720087489451722457687463579275533640332849095569629311375310500863755694135928, 931.4585976207360364052151025112437321783706114582899221015278992780694418320255265658682054939463272),
new_x_y(344.6285788031396759310937079598608283300691323460372986950825938798435180488171502728603865853684386, 933.0721221891391935628392703734606212674815312629868748924057095510284811003305543252989967228673467),
new_x_y(345.790635879726513670205267024047507441296783252085054627023719103687316941445487849201356908932423, 934.7384155818178392634234403310906580872878130785806488177074463686567737756929110909633070461955359),
new_x_y(346.8730227907846353918785154288482072701069837534357732050700801414807929706523568037092028234810686, 936.4551421721162377419785540008797916676530791060975813942074387035360928408016485843129929086509727),
new_x_y(347.8714249784438037081894290398664907359025505814593895984635154131897155532566588220318978598808204, 938.2195794766349923682451171372874110056477223530568321426590678295149354719689542314312945219208098),




new_x_y(349.8324751214893229886717197807415427134736192021860749708254250360510015113836795900772326722715292, 947.2959726273374343576758496295528790011894339674811320598281287632332211501570190786118468988365939),
new_x_y(350.2285623639767927073291206411586766918953863742880702272334681545959051143993768241080334175668138, 949.3564931858395739446107691896315072719555438160874249010218261428162581254996843605919545241490704),
new_x_y(350.7772401783673164976404899424796769153060707953928870453946360131461469710764319200384737791686269, 951.4106819839460526629197761734233042753648650195608645745500071626272640028579117291496275131015133),
new_x_y(351.4724989054193105145330110809507137707374416183740579201390416046849449864266906063896328722077971, 953.4509690897237138467980390775143272416545042581660204149709599116557001755040255279323719610541703),
new_x_y(352.3079575772411421885298855870783153275759756613045172820467861077977436170917114060998881545017366, 955.4704703462020628377299342030161737069633721066758191363758595507807725110545668959982396959440863),
new_x_y(353.2769551202859840270341732324618312406159542451594455992493894275122448995126580068406720655588996, 957.4629860864409238665538453746594350221819298200001591372414410981305833664823120059671877698994362),
new_x_y(354.3726347045687677853854453447561328489882246590412170866464444508873170162980743847667785180852958, 959.4229932662266321487168001102221504824347510010669122890118303673069682994272737631351297139389934),
new_x_y(355.588021021749321947608770628173607551719025814127893432559105583386870107811513402630952458067928, 961.3456319202066861302412769790698859553319897433402951293668470351798862844968785788438384242363404),
new_x_y(356.9160903994473607909007529292445017911648699708686736656502927694126986913065420109717777273242164, 963.2266868080286321862430292878053076566413862909206369123763801695745102611978997054846590911853208),
new_x_y(358.3498337674259475777917586138894015809999115166660643612920557816423367473887615805663846457106229, 965.0625650697236805305350509584657051116894479021369682068903813863668973031530930811966251265032549),
new_x_y(359.8823125836895694293347220093410839986630545812244982400074255676990650664242507458730993108978965, 966.8502706561923492996815961471250196347474804552389049222715876583432930679636880048798861428473753),
new_x_y(361.5067079058872730037747800942452163570168009287748459341049714349898340717465413362479487705958497, 968.5873762430212167594902156200706579606087423479495235394085550884571699823278121062076923449013694),
new_x_y(363.2163628566427002907402714713096029449699118785703289883300725551545119425960749938425840149909669, 970.2719932755870138665644716043239625422364694864279255725117078877470606717787411935268745912312395),
new_x_y(365.0048187816143263902404667851550042536086959253260356039847772482766392944415467705264001510275321, 971.9027407318774724575449977536196660587549025152745287767895306409659902411972129553749828239843357),
new_x_y(366.8658454373545500928773154768913486493221983798975007356652281468289396486901777628105359656444873, 973.4787131278654634936604089070335682795958755874234972429181740778378000362244980441015357017828914),
new_x_y(368.7934655735558717676057384384772952750095510395681528067365872828923958245940519641603656614688796, 974.9994482296092941972286000718971714993313823405373261906684608635794376991694693827894108931133332),
new_x_y(370.7819742922240485804726674608256726810452268387764053944247121936034579449147413102698385779203924, 976.4648948773327780166758346439562264470642249133535410420721865978102217636376526544059343041709912),
new_x_y(372.8259535758631900535878624605119848253375107180273051999404766797653671915689793429265906562099802, 977.875381270213130767503259755817770433182130587174444112289535982046582596664200926228501367692945),
new_x_y(374.9202823790217254112537725465524867039558085748954989013689218148313917128321081334239955156662841, 979.2315840069714863552096542482505665921471771571649774770962475606762119304328799617727433172455057),
new_x_y(377.0601426736055364745620019030558279431647171710466695788353740015981850534325630117324365547266544, 980.5344981269835275912912574104746651524951165201388237346532521298486615710260742825858036100086779),
new_x_y(379.2410218292275391971571809610335798107321074894576976697084143995785554545232870280459860433336705, 981.7854083497508843069781318620401031513462689559967209795568966804283432802950546720805891013226942),
new_x_y(381.4587116964736184501371265673467394254737228994184657889243801635480752167404335318631157659775192, 982.9858616673384137619475045176738090900785882118208224240679762571950233263257383634735653597919235),
new_x_y(383.7093047441898732461778912954712270391039775049781633494319087015072329525550744112053478321106195, 984.137641404840360881442704833120266042623930642301329823579858382659005837983740874657016240423807),
new_x_y(385.9891875825246852585816012150780903122430364468964755226097468548166354976518199132163756712295396, 985.2427428280672369089414506205888024647751229421968182926652588022997433018459992185098924239778271),
new_x_y(388.2950321822021398029344423971788448698503340482795876771194056938166703630252172920011315520459942, 986.3033503453612130167590084865295038947450319412287819934122713345461409487833846523326132732648114),
new_x_y(390.6237850779949726131011479954824054289709413600074963973897769544976626923006430242133804528026888, 987.3218163216178459622414892058804403908413449511997168307030954945111497630577819476317395259354509),
new_x_y(392.9726548211655637293326103692305018491507487660217635964435434507242783539494053465150602213355978, 988.3006414970448543190581865709370259780814992050692807689701255605529020581681391246497744485004532),
new_x_y(395.339097922242316660423319883140508996072949366708039034363129587043556019155784458262608464821704, 989.2424569807250887454180897217091919694713345778618999646394893402485885382557402355560162995142196),
new_x_y(397.7208035023200525608232718307996689269231373038442406974445137944559461411306488931297597687604825, 990.1500077694521302424928974309891699219524694618904939333202241314672318120930063017213232512661124),
new_x_y(400.1156768484801627874147100748242248288587850050139314911877072759158636174472151452645449636140088, 991.0261377253429510734248154797400325769523157221575738712254193245308391140034285414598209917333641),
new_x_y(402.5218220472273166505155147176713426704379364265263249113291643496834270145476899243154408984692311, 991.8737759311679198839238928473888482648088007131944979621468714087008627049236108583297927126439824),
new_x_y(404.9375238492929667017070388032033239181334895546640859889422337061328537478525103092259847858194301, 992.6959243299404176995277778747535283239429764715237134484505093118191588389326977424972611975765664),
new_x_y(407.3612288999760790495497008934152506642621704810129383503557065870959702319850582695393590838134059, 993.4956465448488851920210849825469860370243607570240258952628920090582902937822519830786720017033825),
new_x_y(409.7915264515540957679967341452504811960823250518548428794795475182196735257272827628901691735143967, 994.2760577668760228261587912327785936665523142685348234260278750789942496297845775473175831697077602),
new_x_y(412.2271286583442368823995014462778924431281350728303324958643810652470052166930035196447455556232962, 995.0403155902297480050999545518478247138943391698625038724120458539339648882659020012622768462640081),
new_x_y(414.6668505408402963172126148626188687842162618411984990052683322231953031953858941980834148785527971, 995.791611669821752500559946364588855024593952946320136529169046030988161938668662564456427389615751),
new_x_y(417.1095896930822249708023538358309819415605738359918869930897330017124552011775536619030439690714076, 996.5331640703045636736000942325486068922672336934519174275157673956114201353099342039576029849485575),
new_x_y(419.5543057971038489292437681190235203691729916893207549808498579632323177877447325318472482701081104, 997.2682101724703288832296444036508834851782125660413923517330189728927719833238690271735065303093788),
new_x_y(422.0, 998.0)
]

SEC_4_WAYPOINTS = [
    new_x_y(615.0, 1073.0),
new_x_y(617.793112530441705679044110339301080482052300945548190528314051474706645510145140283674954829564895, 1073.861076316288723592028288713090969186983164670660134061733303028081470430832021427924113667720643),
new_x_y(620.5881304932480726313217554295492525475777736290179112010699948871697929280981713983625771053641048, 1074.715959111369493866455474173239470559166340210146394059782035904509163730188484929782042995702617),
new_x_y(623.3869377075461693459882992482511809271178748171734378274433920837899862907737169872642553416483342, 1075.558448272168389839901319832600209649739043712816683026093401985920296998294526081334041171237453),
new_x_y(626.1913746915460058311512202322970259728532692815709948728131765854055487069535638057910327119053107, 1076.382330745866185133237038708124283531729094280221095915147398779190558943587389159376664098216469),
new_x_y(629.0032168279662241902933000775123030389698084711711015314551227915772442749492153239270089642404459, 1077.181374680585804648589437923421886509347813769272977481551109683751535309429453577816516799512632),
new_x_y(631.8241523141019302128915818032897370154927101337968105678767721336639891025963232330214245980464881, 1077.94932430037342261251350255420938324452756630112629534964052309926809847902596385030747777243807),
new_x_y(634.6557598340864442913420271485388708164741149072292376140296179736019430988932845612244222371118282, 1078.679895761818956688398304414190926739143772711775986730620228679615190995551586229574787575243691),
new_x_y(637.4994858989727586005273178590071496837707016646911015964641890308551481668966142009346895464582787, 1079.366774241631102993621190725587444884292273331267597299889823551665485115587354410389248021240408),
new_x_y(640.3566218104371083711969464258367841560619910403672093485670641981450744093807308290948686576482769, 1080.003612506638979234849130629829443951197127371286620950763488290266127813952146225220681226346143),
new_x_y(643.2282802162366005041363784457602513964467951200050015915796361136988311016835425860398691998906024, 1080.584031219832292242815575721501451663363783808149003981419417514001573784746263497496797812308431),
new_x_y(646.1153712400898240989578301071008145178893214288856992372513521500165232509513123054215513927674282, 1081.101621237928022618322307784951260891083096424737021903270392126196465376918623950307320694520651),
new_x_y(649.0185781854493062928692354191665302750320577543627379533172076538059324799574149193323053862887113, 1081.549948157274761828524464093132020377806382164652005560561743109770939450197373967609959081958969),
new_x_y(651.9383328317502193206746720408797977363027134114705650709876831276588812115199544247979646593144184, 1081.922559365344240171834346590872544238316336473792415859848794677799812396799919314257399293383604),
new_x_y(654.8747903631962199558571347624539557509305866023719897289258184544753433763100327172806084147535472, 1082.212993854238885183112223501580386599651630146287122998708410087717396023366129303569906224967208),
new_x_y(657.827803994013644797031424539143234508590799655716732533390047956318150268894056362871393452683285, 1082.414795050148040148556393842254518040673069427513246157065811284785469870135077063364858111942367),
new_x_y(660.7968993803843318769012335503138997715958383740582377816833368442787545183178650315692721946234089, 1082.521526908056332283896643838879190277880375038333050681506180229668801242763082018250572425938458),
new_x_y(663.7812489379454947869051633454531667981519004259584310348447853310864220182230616605219314728413532, 1082.526793513748949968312490043614142258442218607094883253675718066906325902210532487215958726992811),
new_x_y(666.7796462147813633213352501720093898004272661201708329649815162806255904720192933145681284654883375, 1082.424262424736986512381797218031435674403607721323810412123660887614460581335780349639973677105888),
new_x_y(669.7904805031458292946101577388044394330186708613317409607799051398923078802689048116072455362109748, 1082.207691967575305549523189315721309893773335130698874491349816974370715207924521031414703187697061),
new_x_y(672.8117119086211950181681773339852993686451745146226336512536600933900974944304119607912751851289706, 1081.870962690571374356484659927164632438297059081773111149228918005534566651120185991100874345945295),
new_x_y(675.8408471328528527013483434630214709767581350362100377738337964372617197906857296952392577223365292, 1081.408113147470336915493982633624889571049391154267555065173397925211820879972398543369779865632305),
new_x_y(678.8749162651563465730798522854028060928330431748821281196733332133696910498534450910177461118278972, 1080.813380158719856538003947501005818445844966888304749128603759760541587513477759377084802572527048),
new_x_y(681.910450918851089354399293883755918102659381572709064005026100869802369151109207853781964426800936, 1080.081243661734835328742976707822556615456416700960052446928011307869649470255866251349820378972026),
new_x_y(684.9434640897302845590484955445396119991320206545811822558404907479994130095053502413065986722933643, 1079.206476219572201548778992734213617543212196796822457314196949259233314323329340028357486995955956),
new_x_y(687.9694321561333475631697284886915467384491132388080330132834148998329225194944102189166683518957803, 1078.184197207987195432600520562867392663511559845612742137071516209045476804784163614870765573596205),
new_x_y(690.9832794820481585791280451699935572522953156222682538004566524596381506689171258947632157475999239, 1077.009931643411715851986402351699624967709653250380454356789061770881090239482296453931830292426381),
new_x_y(693.9793661258292033707870714534486426048683248579287007077872906729314760770073974642997065717002657, 1075.679673548467283141921336115656997065416216685905456618437890274549715601088517001704545567384106),
new_x_y(696.9514791966506443224536755599418414792783427828958613043580443265813309853298253027735998185413686, 1074.189953676774990368098899131728315484140572951046775770483401949618818915983356231924134785209696),
new_x_y(699.8928284377743710873429223994755024391933972288486744496816363067213062540984705073585093443249234, 1072.53791133473187366883881703185329738295988223061891404806138250635570687951063929321853675460359),
new_x_y(702.7960466490287740267601768534027647057322270791627906086399905360818959316810784382753708716103982, 1070.721369944398377986970721604257540342121740139515612779351384042816940995670294186366937531327683),
new_x_y(705.6531955893619221223537156205728525830074173755789950380407925427763503996856058086380368931647055, 1068.738915888657085119587722249317227002233909987157371561374177985768587205935234518794154138328449),
new_x_y(708.4557780226222171427175516456001761968196657256249228343337672398331707287975218931986204752814614, 1066.589980067523700959963409108535683404199870190820965246469771943773169200864656526460373857696345),
new_x_y(711.1947565843754186982914874661765533674048164629559294170692109895961792402086210492295230796360288, 1064.274921473309492673233866571832992174170016091540997639037073941774294909296530476051272611280348),
new_x_y(713.860580153018022106422232264287003099752696586207120586525710092412838638280210458521870610904779, 1061.795111962904224246400091218726982833810767785755369247592095618363520554781090500715104369999024),
new_x_y(716.4432184030187396723982464892946524886561975197574654233208469958056064263814123126897429305089471, 1059.153021268723126771345197754235237480629759938180699986899948201981558676098574500222555205783263),
new_x_y(718.9322052000522340643172794247095162588563840437886176228565610675878480238671618154334501803038211, 1056.352301147128488154737900335614479616805950268602433362601516509020375977680910714047587554676931),
new_x_y(721.3166914652607083477133399946033409074008231992840721637902139506486892305755149764491721706075478, 1053.397867416054648820042240946859080505767177398975621118289305946424365111441400770030931156591258),
new_x_y(723.5855080870368025513502431350617722970393963765006911935753181808554470576061050917048912344101168, 1050.295978484197257429491704166562125320472701319683398976203296149189893813158028786879301304447848),
new_x_y(725.7272393917193406512447842086708695524856810268948781297098575820698392903660791643663693164770947, 1047.054308824970686607665173910614745409149332154406542969493369734909231675879787887315756432348712),
new_x_y(727.7303075976373848405238019450006579086961896470258817666901815188815999233156048547455182911068315, 1043.68201570244823855719932756869570657625515011490637931810490979510054251588192403903074595933949),
new_x_y(729.5830685683383060506409440866495687730699208399482082473338497380190242668304818841691828316280626, 1040.189797317113062772023189881051847890214053880519546148126226106919932814280946260331646944869276),
new_x_y(731.273919049069083115682678327553132659567330785439357599488079175814230652469740852797636554666583, 1036.589940410386426999375426222030694869106901448052757009163568495604441863356150043732870175231008),
new_x_y(732.7914154143597451932672151579065839952497705993732622661150679495929142068706277261560898206702128, 1032.896355252973905026562410409017710200448438727675487892830765460233024322270388705661478330652116),
new_x_y(734.1244037729104928002559187380852287826647893237302429825012097371008672848044080496525431592811357, 1029.124595847961684863352069658077453176981103373649334857988026440591495344488273052161369275675172),
new_x_y(735.2621610683352051213725509712572327011080614741028985023047903182984794073141677446067494497957821, 1025.291863110629925639400617253366943184719473889219846209013870473020783212958537576377444314651205),
new_x_y(736.1945465805800722571722665401751529283191131594829000601899130140192100905960287974018857229604915, 1021.416988748848184119758253901467085282353409561944336619920745544499521428528006281450156018767117)


]

SEC_8_WAYPOINTS = [
  new_x_y(-104.11528778076172, -726.1124877929688),
  new_x_y(-104.1638900756836, -728.1100463867188),
  new_x_y(-104.21249237060549, -730.1076049804688),
  new_x_y(-104.26109466552734, -732.1051635742188),
  new_x_y(-104.30969696044922, -734.1027221679688),
  new_x_y(-104.3582992553711, -736.100341796875),
  new_x_y(-104.40690155029296, -738.097900390625),
  new_x_y(-104.45550384521485, -740.095458984375),
  new_x_y(-104.50410614013671, -742.093017578125),
  new_x_y(-104.55270843505859, -744.090576171875),
  new_x_y(-104.60131072998048, -746.088134765625),
  new_x_y(-104.6499053955078, -748.085693359375),
  new_x_y(-104.69850769042968, -750.083251953125),
  new_x_y(-104.74710998535156, -752.0808715820312),
  new_x_y(-104.79571228027343, -754.0784301757812),
  new_x_y(-104.84431457519533, -756.0759887695312),
  new_x_y(-104.8929168701172, -758.0735473632812),
  new_x_y(-104.94151916503907, -760.0711059570312),
  new_x_y(-104.99012145996093, -762.0687255859375),
  new_x_y(-105.0387237548828, -764.0662841796875),
  new_x_y(-105.08732604980467, -766.0638427734375),
  new_x_y(-105.13592834472657, -768.0614013671875),
  new_x_y(-105.18453063964844, -770.0589599609375),
  new_x_y(-105.23313293457032, -772.0565185546875),
  new_x_y(-105.2817352294922, -774.0540771484375),
  new_x_y(-105.33033752441406, -776.0516357421875),
  new_x_y(-105.37893981933594, -778.0491943359375),
  new_x_y(-105.4275421142578, -780.0468139648438),
  new_x_y(-105.47614440917967, -782.0443725585938),
  new_x_y(-105.52474670410156, -784.0419311523438),
  new_x_y(-105.57334899902344, -786.03955078125),
  new_x_y(-105.62195129394533, -788.037109375),
  new_x_y(-105.67055358886721, -790.03466796875),
  new_x_y(-105.71915588378906, -792.0322265625),
  new_x_y(-105.76775817871093, -794.02978515625),
  new_x_y(-105.8163604736328, -796.02734375),
  new_x_y(-105.86496276855468, -798.02490234375),
  new_x_y(-105.91356506347657, -800.0224609375),
  new_x_y(-105.96216735839843, -802.02001953125),
  new_x_y(-106.01076965332032, -804.017578125),
  new_x_y(-106.0593719482422, -806.0151977539062),
  new_x_y(-106.10797424316407, -808.0127563476562),
  new_x_y(-106.15657653808594, -810.0103149414062),
  new_x_y(-106.20517883300779, -812.0079345703125),
  new_x_y(-106.25378112792967, -814.0054931640625),
  new_x_y(-106.30238342285156, -816.0030517578125),
  new_x_y(-106.35098571777344, -818.0006103515625),
  new_x_y(-106.39958801269533, -819.9981689453125),
  new_x_y(-106.4481903076172, -821.9957275390625),
  new_x_y(-106.49679260253906, -823.9932861328125),
  new_x_y(-106.54539489746094, -825.9908447265625),
  new_x_y(-106.6439971923828, -827.9884033203125),
  new_x_y(-106.74259948730467, -829.9859619140625),
  new_x_y(-106.84120178222656, -831.9835815429688),
  new_x_y(-106.93980407714844, -833.9811401367188),
  new_x_y(-107.03840637207033, -835.9786987304688),
  new_x_y(-107.13700103759766, -837.976318359375),
  new_x_y(-107.23560333251952, -839.973876953125),
  new_x_y(-107.33421325683594, -841.971435546875),
  new_x_y(-107.43280792236328, -843.968994140625),
  new_x_y(-107.53141021728516, -845.966552734375),
  new_x_y(-107.63001251220705, -847.964111328125),
  new_x_y(-107.7286148071289, -849.961669921875),
  new_x_y(-107.82721710205078, -851.959228515625),
  new_x_y(-107.92581939697266, -853.956787109375),
  new_x_y(-108.02442169189452, -855.954345703125),
  new_x_y(-108.1230239868164, -857.9519653320312),
  new_x_y(-108.22162628173828, -859.9495239257812),
  new_x_y(-108.32022857666016, -861.9470825195312),
  new_x_y(-108.41883087158205, -863.9447021484375),
  new_x_y(-108.5174331665039, -865.9422607421875),
  new_x_y(-108.61603546142578, -867.9398193359375),
  new_x_y(-108.71463775634766, -869.9373779296875),
  new_x_y(-108.81324005126952, -871.9349365234375),
  new_x_y(-108.9118423461914, -873.9324951171875),
  new_x_y(-109.01044464111328, -875.9300537109375),
  new_x_y(-109.10904693603516, -877.9276123046875),
  new_x_y(-109.20764923095705, -879.9251708984375),
  new_x_y(-109.3062515258789, -881.9227294921875),
  new_x_y(-109.40485382080078, -883.9203491210938),
  new_x_y(-109.50345611572266, -885.9179077148438),
  new_x_y(-109.60205841064452, -887.9154663085938),
  new_x_y(-109.7006607055664, -889.9130859375),
  new_x_y(-109.79926300048828, -891.91064453125),
  new_x_y(-109.89786529541016, -893.908203125),
  new_x_y(-109.99646759033205, -895.90576171875),
  new_x_y(-110.0950698852539, -897.9033203125),
  new_x_y(-110.19367218017578, -899.90087890625),
  new_x_y(-110.29227447509766, -901.8984375),
  new_x_y(-110.390869140625, -903.89599609375),
  new_x_y(-110.48947143554688, -905.8935546875),
  new_x_y(-110.58807373046876, -907.89111328125),
  new_x_y(-110.68667602539062, -909.8887329101562),
  new_x_y(-110.7852783203125, -911.8862915039062),
  new_x_y(-110.88388061523438, -913.8838500976562),
  new_x_y(-110.98248291015624, -915.8814697265624),
  new_x_y(-111.08108520507812, -917.8790283203124),
  new_x_y(-111.1796875, -919.8765869140624),
  new_x_y(-111.27828979492188, -921.8741455078124),
  new_x_y(-111.37689208984376, -923.8717041015624),
  new_x_y(-111.47549438476562, -925.8692626953124),
  new_x_y(-111.5740966796875, -927.8668212890624),
  new_x_y(-111.67269897460938, -929.8643798828124),
  new_x_y(-111.77130126953124, -931.8619384765624),
  new_x_y(-111.86990356445312, -933.8594970703124),
  new_x_y(-111.968505859375, -935.8571166992188),
  new_x_y(-112.06756591796876, -937.8546752929688),
  new_x_y(-112.17355346679688, -939.8518676757812),
  new_x_y(-112.2884521484375, -941.8485717773438),
  new_x_y(-112.4122314453125, -943.8447265625),
  new_x_y(-112.54489135742188, -945.84033203125),
  new_x_y(-112.68646240234376, -947.8353271484376),
  new_x_y(-112.8369140625, -949.8296508789062),
  new_x_y(-112.99624633789062, -951.8233032226562),
  new_x_y(-113.16445922851562, -953.8162231445312),
  new_x_y(-113.3515537600983, -955.8665279809941),
  new_x_y(-113.53996902717849, -957.9167117971767),
  new_x_y(-113.73102552986727, -959.9666510205336),
  new_x_y(-113.9260432892828, -962.0162169747531),
  new_x_y(-114.12634158750711, -964.0652733298231),
  new_x_y(-114.33323868289689, -966.1136735545402),
  new_x_y(-114.54805149255013, -968.1612583724475),
  new_x_y(-114.77209523374661, -970.2078532223375),
  new_x_y(-115.006683016202, -972.2532657246397),
  new_x_y(-115.25312537700373, -974.297283155235),
  new_x_y(-115.51272975013325, -976.3396699284953),
  new_x_y(-115.78679986252514, -978.3801650916428),
  new_x_y(-116.07663504867058, -980.4184798328554),
  new_x_y(-116.38352947584305, -982.4542950059051),
  new_x_y(-116.70877127210922, -984.4872586745246),
  new_x_y(-117.05364154939126, -986.5169836801274),
  new_x_y(-117.41941331396944, -988.5430452369803),
  new_x_y(-117.80735025695967, -990.5649785594276),
  new_x_y(-118.21870541747198, -992.5822765263035),
  new_x_y(-118.65471971135594, -994.5943873882336),
  new_x_y(-119.11662031867073, -996.6007125241229),
  new_x_y(-119.605618923285, -998.6006042537522),
  new_x_y(-120.12290979831764, -1000.5933637140547),
  new_x_y(-120.66966773147875, -1002.578238807317),
  new_x_y(-121.24704578476556, -1004.5544222302485),
  new_x_y(-121.85617288341206, -1006.5210495935763),
  new_x_y(-122.49815122949076, -1008.4771976425546),
  new_x_y(-123.17405353612175, -1010.4218825895213),
  new_x_y(-123.88492007886275, -1012.3540585703897),
  new_x_y(-124.63175556153985, -1014.2726162377194),
  new_x_y(-125.41552579453258, -1016.1763815037664),
  new_x_y(-126.23715418435646, -1018.064114447671),
  new_x_y(-127.0975180342922, -1019.9345084016783),
  new_x_y(-127.99744465679828, -1021.7861892320115),
  new_x_y(-129.29904174804688, -1024.0030168456112),
  new_x_y(-130.23733520507812, -1025.3253234392039),
  new_x_y(-131.21519470214844, -1026.6535923650738),
  new_x_y(-132.2321319580078, -1027.9847638155609),
  new_x_y(-133.28762817382812, -1029.3159715472532),
  new_x_y(-134.38116455078125, -1030.6445645242186),
  new_x_y(-135.51214599609375, -1031.9679987239151),
  new_x_y(-136.68003845214844, -1033.283986323702),
  new_x_y(-137.8842315673828, -1034.590318591403),
  new_x_y(-139.12411499023438, -1035.8849434285803),
  new_x_y(-140.3990478515625, -1037.1659199800672),
  new_x_y(-141.70838928222656, -1038.431439475185),
  new_x_y(-143.05148315429688, -1039.679798522091),
  new_x_y(-144.42764282226562, -1040.9093769333458),
  new_x_y(-145.83311462402344, -1042.1160772946891),
  new_x_y(-147.24366760253906, -1043.2797508701397),
  new_x_y(-148.6542205810547, -1044.398189605361),
  new_x_y(-150.0647735595703, -1045.4734179410927),
  new_x_y(-151.9115447998047, -1046.818879204404),
  new_x_y(-153.56297302246094, -1047.9649649606927),
  new_x_y(-155.27511596679688, -1049.0989259660173),
  new_x_y(-157.0152130126953, -1050.1971999910343),
  new_x_y(-158.78709411621094, -1051.261679878713),
  new_x_y(-160.56326293945312, -1052.2763142840188),
  new_x_y(-162.36941528320312, -1053.256244237683),
  new_x_y(-164.17236328125, -1054.1841280003594),
  new_x_y(-166.06207275390625, -1055.1045175025336),
  new_x_y(-167.92999267578125, -1055.963504145741),
  new_x_y(-169.81085205078125, -1056.7789530599423),
  new_x_y(-171.71499633789062, -1057.555335983251),
  new_x_y(-173.6272430419922, -1058.2865596520683),
  new_x_y(-175.54446411132812, -1058.9721381371337),
  new_x_y(-177.5172576904297, -1059.6290460067014),
  new_x_y(-179.49436950683594, -1060.2390685602497),
  new_x_y(-181.5004119873047, -1060.80958016515),
  new_x_y(-183.51683044433597, -1061.3347975878194),
  new_x_y(-185.5426788330078, -1061.8146020439071),
  new_x_y(-187.5782928466797, -1062.2491491193605),
  new_x_y(-189.625, -1062.6386692708334),
  new_x_y(-191.70413208007807, -1062.9863178358407),
  new_x_y(-193.7519073486328, -1063.2819196173116),
  new_x_y(-195.80239868164065, -1063.5318193259066),
  new_x_y(-197.89547729492188, -1063.7397222603213),
  new_x_y(-199.98516845703125, -1063.9000472976386),
  new_x_y(-202.0535888671875, -1064.0124941221452),
  new_x_y(-204.19088745117188, -1064.0805283660864),
  new_x_y(-206.28028869628903, -1064.0998263672855),
  new_x_y(-208.3740997314453, -1064.0723702297396),
  new_x_y(-210.478515625, -1063.9975325926928),
  new_x_y(-212.55540466308597, -1063.8771264852148),
  new_x_y(-214.64825439453125, -1063.708837278481),
  new_x_y(-216.74729919433597, -1063.4924460365803),
  new_x_y(-218.8514862060547, -1063.2273400866045),
  new_x_y(-220.93853759765625, -1062.9163286544263),
  new_x_y(-223.02557373046875, -1062.5569658086474),
  new_x_y(-225.13626098632807, -1062.1437710248788),
  new_x_y(-227.19061279296875, -1061.6929057110094),
  new_x_y(-229.30372619628903, -1061.1782614004462)
]

SEC_12_WAYPOINTS = [
 new_x_y(-343.2425231933594, 57.59950256347656),
  new_x_y(-343.2458117675781, 59.59837341308594),
  new_x_y(-343.24910034179686, 61.59727478027344),
  new_x_y(-343.2523889160156, 63.59614562988281),
  new_x_y(-343.2556469726562, 65.59504699707031),
  new_x_y(-343.258935546875, 67.59391784667969),
  new_x_y(-343.2622241210937, 69.59281921386719),
  new_x_y(-343.2655126953125, 71.59169006347656),
  new_x_y(-343.26880126953125, 73.59059143066406),
  new_x_y(-343.27208984375, 75.58946228027344),
  new_x_y(-343.27537841796874, 77.58836364746094),
  new_x_y(-343.2786669921875, 79.58723449707031),
  new_x_y(-343.2819555664062, 81.58613586425781),
  new_x_y(-343.2852136230469, 83.58500671386719),
  new_x_y(-343.28850219726564, 85.58390808105469),
  new_x_y(-343.2917907714844, 87.58277893066406),
  new_x_y(-343.29507934570313, 89.58168029785156),
  new_x_y(-343.2983679199219, 91.58058166503906),
  new_x_y(-343.3016564941406, 93.57945251464844),
  new_x_y(-343.30494506835936, 95.57835388183594),
  new_x_y(-343.3082336425781, 97.57722473144533),
  new_x_y(-343.3115222167969, 99.5761260986328),
  new_x_y(-343.3147802734375, 101.5749969482422),
  new_x_y(-343.31806884765626, 103.57389831542967),
  new_x_y(-343.321357421875, 105.57276916503906),
  new_x_y(-343.3246459960937, 107.57167053222656),
  new_x_y(-343.3279345703125, 109.57054138183594),
  new_x_y(-343.33122314453124, 111.56944274902344),
  new_x_y(-343.33451171875, 113.5683135986328),
  new_x_y(-343.3378002929687, 115.56721496582033),
  new_x_y(-343.3410888671875, 117.56608581542967),
  new_x_y(-343.34434692382814, 119.5649871826172),
  new_x_y(-343.3476354980469, 121.56385803222656),
  new_x_y(-343.3509240722656, 123.56275939941406),
  new_x_y(-343.35421264648437, 125.56166076660156),
  new_x_y(-343.3575012207031, 127.56053161621094),
  new_x_y(-343.36078979492186, 129.55943298339844),
  new_x_y(-343.3640783691406, 131.5583038330078),
  new_x_y(-343.3673669433594, 133.5572052001953),
  new_x_y(-343.370625, 135.5560760498047),
  new_x_y(-343.37391357421876, 137.5549774169922),
  new_x_y(-343.3072021484375, 139.55384826660156),
  new_x_y(-343.24049072265626, 141.55274963378906),
  new_x_y(-343.173779296875, 143.55162048339844),
  new_x_y(-343.1070678710937, 145.55052185058594),
  new_x_y(-343.0403564453125, 147.5493927001953),
  new_x_y(-342.97364501953126, 149.5482940673828),
  new_x_y(-342.90693359375, 151.5471649169922),
  new_x_y(-342.84019165039064, 153.5460662841797),
  new_x_y(-342.7734802246094, 155.54493713378906),
  new_x_y(-342.70676879882814, 157.54383850097656),
  new_x_y(-342.6400573730469, 159.54270935058594),
  new_x_y(-342.57334594726564, 161.54161071777344),
  new_x_y(-342.5066345214844, 163.54051208496094),
  new_x_y(-342.43992309570314, 165.5393829345703),
  new_x_y(-342.3732116699219, 167.5382843017578),
  new_x_y(-342.30650024414064, 169.5371551513672),
  new_x_y(-342.23975830078126, 171.5360565185547),
  new_x_y(-342.173046875, 173.53492736816406),
  new_x_y(-342.10633544921876, 175.53382873535156),
  new_x_y(-342.0396240234375, 177.53269958496094),
  new_x_y(-341.97291259765626, 179.53160095214844),
  new_x_y(-341.906201171875, 181.5304718017578),
  new_x_y(-341.8394897460937, 183.5293731689453),
  new_x_y(-341.7727783203125, 185.5282440185547),
  new_x_y(-341.70606689453126, 187.5271453857422),
  new_x_y(-341.6393249511719, 189.5260162353516),
  new_x_y(-341.57261352539064, 191.52491760253903),
  new_x_y(-341.5059020996094, 193.52378845214844),
  new_x_y(-341.43919067382814, 195.52268981933597),
  new_x_y(-341.3724792480469, 197.52159118652344),
  new_x_y(-341.30576782226564, 199.5204620361328),
  new_x_y(-341.2390563964844, 201.5193634033203),
  new_x_y(-341.17234497070314, 203.5182342529297),
  new_x_y(-341.1056335449219, 205.5171356201172),
  new_x_y(-341.0388916015625, 207.51600646972656),
  new_x_y(-340.97218017578126, 209.5149078369141),
  new_x_y(-340.90546875, 211.51377868652344),
  new_x_y(-340.83875732421876, 213.5126800537109),
  new_x_y(-340.7720458984375, 215.5115509033203),
  new_x_y(-340.70533447265626, 217.5104522705078),
  new_x_y(-340.638623046875, 219.5093231201172),
  new_x_y(-340.5719116210937, 221.5082244873047),
  new_x_y(-340.5052001953125, 223.5070953369141),
  new_x_y(-340.43845825195314, 225.5059967041016),
  new_x_y(-340.3717468261719, 227.5048675537109),
  new_x_y(-340.30503540039064, 229.50376892089844),
  new_x_y(-340.2383239746094, 231.50267028808597),
  new_x_y(-340.17161254882814, 233.5015411376953),
  new_x_y(-340.1049011230469, 235.5004425048828),
  new_x_y(-340.03818969726564, 237.4993133544922),
  new_x_y(-339.9714782714844, 239.4982147216797),
  new_x_y(-339.90476684570314, 241.49708557128903),
  new_x_y(-339.8380249023437, 243.49598693847656),
  new_x_y(-339.7713134765625, 245.49485778808597),
  new_x_y(-339.70460205078126, 247.49375915527344),
  new_x_y(-339.637890625, 249.4926300048828),
  new_x_y(-339.57117919921876, 251.4915313720703),
  new_x_y(-339.5044677734375, 253.4904022216797),
  new_x_y(-339.43775634765626, 255.4893035888672),
  new_x_y(-339.371044921875, 257.4881591796875),
  new_x_y(-339.3043334960937, 259.487060546875),
  new_x_y(-339.2375915527344, 261.4859619140625),
  new_x_y(-339.17088012695314, 263.48486328125),
  new_x_y(-339.1041687011719, 265.48370361328125),
  new_x_y(-339.03745727539064, 267.48260498046875),
  new_x_y(-338.9707458496094, 269.48150634765625),
  new_x_y(-338.90403442382814, 271.4804077148437),
  new_x_y(-338.8373229980469, 273.479248046875),
  new_x_y(-338.77061157226564, 275.4781494140625),
  new_x_y(-338.70386962890626, 277.47705078125),
  new_x_y(-338.637158203125, 279.4759521484375),
  new_x_y(-338.5704467773437, 281.47479248046875),
  new_x_y(-338.5037353515625, 283.47369384765625),
  new_x_y(-338.43702392578126, 285.4725952148437),
  new_x_y(-338.3699462890625, 287.4714660644531),
  new_x_y(-338.2862670898437, 289.4696960449219),
  new_x_y(-338.177685546875, 291.4667358398437),
  new_x_y(-337.74420166015625, 293.4622802734375),
  new_x_y(-337.561151551239, 295.57938104829776),
  new_x_y(-337.3687309999671, 297.6956472447914),
  new_x_y(-337.15758115312997, 299.81011987990416),
  new_x_y(-336.91836908806914, 301.9215914330887),
  new_x_y(-336.6418076495959, 304.0284823421357),
  new_x_y(-336.318683504231, 306.12871865486255),
  new_x_y(-335.9398960929732, 308.21961161765273),
  new_x_y(-335.4965100939307, 310.29774031792573),
  new_x_y(-334.9798238937089, 312.3588389125448),
  new_x_y(-334.38145639508144, 314.39769046277337),
  new_x_y(-333.69345423906987, 316.40802995065553),
  new_x_y(-332.90842117070656, 318.38245965990507),
  new_x_y(-332.01967080638593, 320.3123807501063),
  new_x_y(-331.02140344302677, 322.18794551404585),
  new_x_y(-329.9089067619942, 323.99803545528476),
  new_x_y(-328.67877930271305, 325.7302709198313),
  new_x_y(-327.3291743951622, 327.37105851663097),
  new_x_y(-325.8600608366352, 328.90568291218955),
  new_x_y(-324.27349497551563, 330.3184497216768),
  new_x_y(-322.57389703549154, 331.5928860707134),
  new_x_y(-320.7683225063256, 332.7120048904),
  new_x_y(-318.86671729112027, 333.6586380505436),
  new_x_y(-320.55279541015625, 332.7275085449219),
  new_x_y(-318.68414306640625, 333.4375305175781),
  new_x_y(-316.74951171875, 333.9408569335937),
  new_x_y(-314.7717590332031, 334.2315979003906),
  new_x_y(-312.7741394042969, 334.30633544921875),
  new_x_y(-310.7801818847656, 334.16412353515625),
  new_x_y(-308.80279541015625, 333.8643798828125),
  new_x_y(-307.814697265625, 333.7107238769531),
  new_x_y(-306.8265686035156, 333.5570678710937),
  new_x_y(-305.83843994140625, 333.4034423828125),
  new_x_y(-304.8503112792969, 333.2497863769531),
  new_x_y(-303.8621826171875, 333.0961303710937),
  new_x_y(-301.877197265625, 332.8551025390625),
  new_x_y(-299.88006591796875, 332.75640869140625),
  new_x_y(-297.8809814453125, 332.8007202148437),
  new_x_y(-295.8901672363281, 332.9877319335937),
  new_x_y(-293.9178161621094, 333.3164978027344),
  new_x_y(-291.9739990234375, 333.7853698730469),
  new_x_y(-290.0686340332031, 334.3919372558594),
  new_x_y(-287.36890955136124, 336.4502300627805),
  new_x_y(-285.53462028523313, 337.43563207506764),
  new_x_y(-283.8206015721414, 338.6180705351394),
  new_x_y(-282.2413695974792, 339.97542718319943),
  new_x_y(-280.8073997467029, 341.48555109159435),
  new_x_y(-279.52547159263526, 343.12681286859697),
  new_x_y(-278.3990548588583, 344.87855052651463),
  new_x_y(-277.4287151511894, 346.7214132459724),
  new_x_y(-276.6125219176662, 348.63761172938075),
  new_x_y(-275.9464446647921, 350.61108536348434),
  new_x_y(-275.424726785363, 352.62759715614163),
  new_x_y(-275.04022934994344, 354.6747675289015),
  new_x_y(-274.78473982892007, 356.7420576792383),
  new_x_y(-274.6492429260599, 358.8207125056033),
  new_x_y(-274.62415252333415, 360.9036721291275),
  new_x_y(-274.6995051839878, 362.98545994418),
  new_x_y(-274.86511677200605, 365.06205396436957),
  new_x_y(-275.11070456392184, 367.1307470623927),
  new_x_y(-275.42597779922124, 369.1900005776115),
  new_x_y(-275.8006999845239, 371.23929471755116),
  new_x_y(-276.22472647836435, 373.2789782309583),
  new_x_y(-276.68802097838517, 375.31011899439363),
  new_x_y(-277.18065454720767, 377.334356438785),
  new_x_y(-277.6927907782895, 379.3537561495886),
  new_x_y(-278.21466064453125, 381.3706665039063),
  new_x_y(-278.75653076171875, 383.2958679199219),
  new_x_y(-279.324462890625, 385.2135009765625),
  new_x_y(-279.9183959960937, 387.1232604980469),
  new_x_y(-280.5381774902344, 389.0247802734375),
  new_x_y(-281.29998779296875, 391.6999816894531),
  new_x_y(-282.1494140625, 393.758056640625)
]



# Section 0: 319
# Section 1: 175
# Section 2: 114
# Section 3: 146
# Section 4: 124
# Section 5: 77
# Section 6: 313
# Section 7: 211
# Section 8: 254
# Section 9: 75
# Section 10: 238
# Section 11: 189
# Section 12: 161
# Section 0: 207
# Section 1: 162
# Section 2: 112
# Section 3: 146
# Section 4: 124
# Section 5: 75
# Section 6: 315
# Section 7: 213
# Section 8: 252
# Section 9: 75
# Section 10: 241
# Section 11: 188
# Section 12: 163
# Section 0: 207
# Section 1: 162
# Section 2: 112
# Section 3: 146
# Section 4: 124
# Section 5: 77
# Section 6: 314
# Section 7: 212
# Section 8: 252
# Section 9: 75
# Section 10: 239
# Section 11: 188
# Section 12: 164
# end of the loop
# done
# Solution finished in 347.2500000000506 seconds