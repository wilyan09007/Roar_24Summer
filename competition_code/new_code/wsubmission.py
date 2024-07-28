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
        startInd2 = 480
        endInd2 = 515
        startInd_8 = 1800
        endInd_8 = 2006
        startInd_12 = 2586
        
            # maneuverable_waypoints[:startInd2] + SEC_2_WAYPOINTS \
            # + maneuverable_waypoints[endInd2:startInd_8] + SEC_8_WAYPOINTS \
            # + maneuverable_waypoints[endInd_8:startInd_12] \
            # + SEC_12_WAYPOINTS
        temp = maneuverable_waypoints # so indexes don't change
        self.maneuverable_waypoints = \
        temp[:startInd2] + SEC_2_WAYPOINTS \
            + temp[endInd2:startInd_8] + SEC_8_WAYPOINTS \
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
            90: 12,
            110: 13,
            130: 14,
            160: 16,
            180: 20,
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
        mu = 1.0
        if current_section == 0: #CHNAGE TO ONLY SET SPEED FOR TURNS, OTHERWISE MAX SPEED, SO CHANGE SECTION DEFINITIONS, ALSO GO CHANGE ACCELERATION LIMIT BY SECTION, change braking style if possible
            mu = 10
        if current_section == 1:
            mu = 2.3
        if current_section == 2:
            mu = 10
        if current_section == 3:
            mu = 10
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

SEC_2_WAYPOINTS = [
    new_x_y(296.0, 913.0),
new_x_y(298.0499988831381560493562048839717509734566719676894649599814925549690215747523046149145945203694936, 913.0021398907243335942791063091723710514310300549935211181226623496450363666542493923767436713551149),
new_x_y(300.0999863419044289116023301221908734147756309528440501466218566884378753120951875189254401748818474, 913.0100297675864182181339371259858459636521633934812540487538888565860228367213551366086798611045975),
new_x_y(302.1499184187581084439479521664978106587640689230098630621749978479272375342251760607520092695017915, 913.0294194610945218288419935576710049702337593378669298816611950194591172082835089288378677195837504),
new_x_y(304.199686092448348994655850971148640013389913852243033988021208085481369291268270280212019159371303, 913.0660580342467501004278059656572463889149621314240026203792488217989759632913529971462566560411195),
new_x_y(306.2490827587511113890012822470297277510409708472136227996738832877390755883499047418729177345644944, 913.1256922582218425382550014550480597358193448833365836864754410660834334840689682139853972569631401),
new_x_y(308.2977717431796835208946397641900557531645980567329224687794421497273190820039092505107857249963301, 913.2140637197638920890744429383141794505979388511216069339713153640211558477414168431575290512172918),
new_x_y(310.3452538844232687216041352797187977314862691048766583456051775439538885888987873657517557152429806, 913.3369041052131310649313441561418405349971205825328795126140690697199313062294146953479055425814826),
new_x_y(312.3908352513237493564570230489571135636286470274951238099223192139868368437451151652883370488547148, 913.4999282079502522165079485141845553175092325822670215856520626633622300225377391062352717060958006),
new_x_y(314.4335950862139606035239409756810381660744930159863128131828871601666878360900954833212268570107696, 913.7088242094294262199134261035361723643101127374949838334991947824955564205021214676400692032836432),
new_x_y(316.4723541033394859024976503036520662453240981363061609052182374065032314991057670219296701330157484, 913.9692407897315481409145602346650868252530663920544709740431571533774582185143741242736905133724059),
new_x_y(318.5056433127489850733080696027750605444259117741711566315051852860213443470725786281777436039567615, 914.2867706325775196940085200908228618398326276063976763513957016090104274920660469709868626315293312),
new_x_y(320.5316735872766391564922309069780665903872504357801968895249350672534511289492374479184117410901372, 914.6669299030472261682021648699975683344817343981423547033636599714557564232186462447481667047194665),
new_x_y(322.5483062427765347805490363424192722716536742912077615448241010017561592123842221714346735688442163, 915.1151332950345077778035908246690674143224000989188352336423297863386845854148995231437298999380071),
new_x_y(324.5530249592114363784241698693805105032910370708330759785860393829941819095052956429768069176890647, 915.6366642710384217025374932663972972715323042330333500721979799893921111406708674994696581268783617),
new_x_y(326.5429094320161528711986261227891389997458355929799269441547833595471893538572989321266915618981018, 916.2366401506640918362164302033064374202347580760208810918583195192982090675311354830053645336890771),
new_x_y(328.514611208648780901479827538413543366806766547699174148779142229726372669742896347849021646146478, 916.9199717476918183515822575742321554972450379370347181972271949992053636565366221768256414831633036),
new_x_y(330.4643322335130959497104225522870395973614289543850657560834301307377289532356884723821398971352276, 917.6913173103465482222910727767375576112652040056275361397847304357485379356851269522621713994569144),
new_x_y(332.3878066943545299356014256669621966250034668955682062487523188200854431971818281114812603440792814, 918.5550305870707356319544212493292573148494641065446671567101114254988042752524519254654753709543805),
new_x_y(334.2802868334128972259683175482579213638763166094328127986331077990191862721882338045981083461448893, 919.5151029222745765279658181836325095695764873974853589644891423044505750526685598399588301943205917),
new_x_y(336.1365334553805804737526006919257157007370536231706362044246531590737941685899650434598939764338676, 920.5750993847543564764926874163043399131174520684443531835432012816577437876241856384005174638255189),
new_x_y(337.9508119295729162008017456690129586571538413552067241370139110602471233520791678126004004605467289, 921.7380890471613478770055959192178232173253836038209071438428655194273922193348214126578077098082268),
new_x_y(339.7168945433379274309179275153999999204476366150079819312653883202143465890020448408694602992311262, 923.0065696693121720889481276429145624699031530586562325289559823307678346051890573317753952967613864),
new_x_y(341.4280701149316477985564798991684591853242871173040071235002672509339609809951389064612974342484433, 924.382387192229412840265988827236721551661613681543827116520943950849940138310407347589796715056157),
new_x_y(343.0771618138184772948855904211436671011635636905942251738038451263189580922423669218979953033370732, 925.8666506241985245649529419871493294722837468618959516765333052775269633695966122824879183726644263),
new_x_y(344.6565541612215308734881945895025128378574628371054209388526881226052977422294201795062178771364528, 927.4596430949665171297942463254178327461653984735044289995319584806845365629024442901462613848209266),
new_x_y(346.1582301900043022792717586033597634559848245678740027007385641416291470621584203021592006173090779, 929.1607300690501843558954887340015998571381850680234520498926624425403574382419571846485507958660417),
new_x_y(347.5738197265652155173369870665242552428764146032766018527444991669937956340480023095751296330629137, 930.9682659428213860165121040774634989137196922415671538675165012925730215102656944130154941083437298),
new_x_y(348.8946597140723832879511923661643672915624822373926835155801836782714361096283940600682936282768357, 932.8795005006121397663070757092174569543458367245608454323805523788860752674371586958315817637118296),
new_x_y(350.1118674215844125324974078878705265919847144120328416271781784102387830523072828263529885121364984, 934.890486969580345895509236627214525375254617118061337439058464089494300506166846972320880789656713),
new_x_y(351.2164272728504284920743295516833474515389036671670729184191278555788070000585739739608641116435735, 936.9959936874410971356126413064823022068057942416994869444306184445929702031183668412438130193404907),
new_x_y(352.1992918773755478801144146273556876364519560800736216404200843532769833575477249409632977898525244, 939.1894216761104073863373880230278986474395743632080858003280450523303785033074413007714112614796928),
new_x_y(353.0514976504180060820943619760710854522173618343604580201386561906181336341748539223734035560096957, 941.4627306911931896325064853488595241649634942911595092032840014639572075344228901392728319879744271),
new_x_y(353.7642951641130571377967260678800591445822942162846586565700029218756230905309921883508914692606315, 943.8063765840000508583402514556059258464679481726816684408483525236822710117650516521546981657399121),
new_x_y(354.3292940757099516084677183233132380520211889995589501533944451031171550997875315955485294559591808, 946.2092630598170997589886057813037828972727659985603751209093326441171272466321528448140058901180096)
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