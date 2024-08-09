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
        startInd_2 = 336
        endInd_2 = 547
        startInd_4 = 663
        endInd_4 = 742
        startInd_5 = 800
        endInd_5 = 871
        startInd_6 = 1306
        endInd_6 = 1445
        startInd_7 = 1445
        endInd_7 = 1483
        startInd_8 = 1882
        endInd_8 = 2012
        startInd_12 = 2648
        
        # temp[:startInd_2] + SEC_2_WAYPOINTS \
        #     + temp[endInd_2:startInd_4] + SEC_4_WAYPOINTS \
        #     + temp[endInd_4:startInd_8] + SEC_8_WAYPOINTS \
        #     + temp[endInd_8:startInd_12] \
        #     + SEC_12_WAYPOINTS
        temp = maneuverable_waypoints # so indexes don't change
        self.maneuverable_waypoints = \
        temp[:startInd_2] + SEC_2_WAYPOINTS \
            + temp[endInd_2:startInd_4] + SEC_4_WAYPOINTS \
            + temp[endInd_4:startInd_5] + SEC_5_WAYPOINTS \
            + temp[endInd_5:startInd_6] + SEC_6_WAYPOINTS \
            + temp[endInd_6:startInd_7] + SEC_7_WAYPOINTS \
            + temp[endInd_7:startInd_8] + SEC_8_WAYPOINTS \
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
        self.section_indeces = [5, 425, 547, 691, 803, 884, 1345, 1508, 1854, 1968, 2264, 2565, 2750]
        print(f"1 lap length: {len(self.maneuverable_waypoints)}")
        print(f"indexes: {self.section_indeces}")

        # Receive location, rotation and velocity data 
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation ()
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
            "throttle": np.clip(throttle, 0.0, 1.0),
            "steer": np.clip(steer_control, -1.0, 1.0),
            "brake": np.clip(brake, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": gear
        }

        print("--- " + str(throttle) + " " + str(brake)
                    + " steer " + str(steer_control)
                    + " loc: " + str(vehicle_location)
                    + " cur_ind: " + str(self.current_waypoint_idx)
                    + " cur_tar: " + str(self.get_lookahead_index(current_speed_kmh))
                    + " cur_sec: " + str(self.current_section)
                    , end="\t"
                    ) 


        await self.vehicle.apply_action(control)
        return control

    def get_lookahead_value(self, speed):
        speed_to_lookahead_dict = {
            70: 12,
            90: 12,
            110: 12,
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
        self.max_speed = 400
        self.intended_target_distance = [i * 15 for i in range(1,14)]
        self.target_distance = [i * 15 for i in range(1,14)]
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
        # speed_data.append(SpeedData(close_distance, current_speed, target_speed1, target_speed1))
        # speed_data.append(SpeedData(mid_distance, current_speed, target_speed2, target_speed2))
        # speed_data.append(SpeedData(far_distance, current_speed, target_speed3, target_speed3))
        speed_data.append(self.speed_for_turn(close_distance, target_speed1, current_speed, current_section))
        speed_data.append(self.speed_for_turn(mid_distance, target_speed2, current_speed, current_section))
        speed_data.append(self.speed_for_turn(far_distance, target_speed3, current_speed, current_section))

        # speed_data.append(self.speed_for_turn(close_distance, target_speed1, current_speed)) #distance curr target max
        # speed_data.append(self.speed_for_turn(mid_distance, target_speed2, current_speed))
        # speed_data.append(self.speed_for_turn(far_distance, target_speed3, current_speed))

        if current_speed > 100:
            # at high speed use larger spacing between points to look further ahead and detect wide turns.
            r4 = self.get_radius([wp[self.mid_index+3], wp[self.mid_index+4], wp[self.mid_index+5]])
            target_speed4 = self.get_target_speed(r4, current_section)
            # speed_data.append(SpeedData(mid_distance, current_speed, target_speed4, target_speed4))
            speed_data.append(self.speed_for_turn(mid_distance, target_speed4, current_speed, current_section))

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

        percent_change_per_tick = 0.05 # speed drop for one time-tick of braking
        speed_up_threshold = 0.99
        throttle_decrease_multiple = 0.7
        throttle_increase_multiple = 1.25
        percent_speed_change = (speed_data.current_speed - self.previous_speed) / (self.previous_speed + 0.0001) # avoid division by zero
        print(percent_of_max, speed_data.recommended_speed_now)

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
            print(sd.recommended_speed_now, end="  ")
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

    def speed_for_turn(self, distance: float, target_speed: float, current_speed: float, current_section):
        # a = (target_speed**2 - current_speed**2) / (2 * distance)
        # speed = math.sqrt(current_speed**2 + 2 * a * distance)
        # return SpeedData(distance, current_speed, target_speed, speed)
        # return speed
        if(current_section == 11):
            return SpeedData(distance, current_speed, target_speed, target_speed)
        if(current_section == 1):
            return SpeedData(distance, current_speed, target_speed, target_speed)
        if(current_section == 4):
            return SpeedData(distance, current_speed, target_speed, target_speed)
        # if(current_section == 6):
        #     return SpeedData(distance, current_speed, target_speed, target_speed)
        if(current_section == 8):
            return SpeedData(distance, current_speed, target_speed, target_speed)
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

        # print(len(points))
        # self.dprint("wp dist " +  str(dist))
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
            mu = 1.5
        if current_section == 2:
            mu = 2.7
        if current_section == 3:
            mu = 2.6
        if current_section == 4: #crash end of 4
            mu = 3
        if current_section == 5: # not decel
            mu = 1
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
        if(current_section==1):
            target_speed = 125
        if(current_section==11):
            target_speed = max(target_speed, 95)
        if(current_section==4):
            target_speed = max(target_speed, 154)
        if(current_section==6):
            target_speed = max(target_speed, 120)
        if(current_section==8):
            target_speed = max(target_speed, 180)
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
    new_x_y(5.0, 884.0),
new_x_y(7.044573774995567958203582739500835090561590894994125929941374269135132369663503069725764967411827876, 884.1839169656777704838752752040095311177987574535970989068353154732072891311621748007102301623945674),
new_x_y(9.089147549991135916407165479001670181123181789988251859882748538270264739327006139451529934823655753, 884.3678339313555409677505504080190622355975149071941978136706309464145782623243496014204603247891349),
new_x_y(11.13372132498670387461074821850250527168477268498237778982412280740539710899050920917729490223548363, 884.5517508970333114516258256120285933533962723607912967205059464196218673934865244021306904871837023),
new_x_y(13.17829509998227183281433095800334036224636357997650371976549707654052947865401227890305986964731151, 884.7356678627110819355011008160381244711950298143883956273412618928291565246486992028409206495782697),
new_x_y(15.22286887497783979101791369750417545280795447497062964970687134567566184831751534862882483705913938, 884.9195848283888524193763760200476555889937872679854945341765773660364456558108740035511508119728371),
new_x_y(17.26744264997340774922149643700501054336954536996475557964824561481079421798101841835458980447096726, 885.1035017940666229032516512240571867067925447215825934410118928392437347869730488042613809743674046),
new_x_y(19.31201642496897570742507917650584563393113626495888150958961988394592658764452148808035477188279514, 885.287418759744393387126926428066717824591302175179692347847208312451023918135223604971611136761972),
new_x_y(21.35659019996454366562866191600668072449272715995300743953099415308105895730802455780611973929462301, 885.4713357254221638710022016320762489423900596287767912546825237856583130492973984056818412991565394),
new_x_y(23.40116397496011162383224465550751581505431805494713336947236842221619132697152762753188470670645089, 885.6552526910999343548774768360857800601888170823738901615178392588656021804595732063920714615511069),
new_x_y(25.44573774995567958203582739500835090561590894994125929941374269135132369663503069725764967411827876, 885.8391696567777048387527520400953111779875745359709890683531547320728913116217480071023016239456743),
new_x_y(27.49031152495124754023941013450918599617749984493538522935511696048645606629853376698341464153010664, 886.0230866224554753226280272441048422957863319895680879751884702052801804427839228078125317863402417),
new_x_y(29.53488529994681549844299287401002108673909073992951115929649122962158843596203683670917960894193452, 886.2070035881332458065033024481143734135850894431651868820237856784874695739460976085227619487348092),
new_x_y(31.57945907494238345664657561351085617730068163492363708923786549875672080562553990643494457635376239, 886.3909205538110162903785776521239045313838468967622857888591011516947587051082724092329921111293766),
new_x_y(33.62403284993795141485015835301169126786227252991776301917923976789185317528904297616070954376559027, 886.574837519488786774253852856133435649182604350359384695694416624902047836270447209943222273523944),
new_x_y(35.66860662493351937305374109251252635842386342491188894912061403702698554495254604588647451117741815, 886.7587544851665572581291280601429667669813618039564836025297320981093369674326220106534524359185114),
new_x_y(37.71318039992908733125732383201336144898545431990601487906198830616211791461604911561223947858924602, 886.9426714508443277420044032641524978847801192575535825093650475713166260985947968113636825983130789),
new_x_y(39.7577541749246552894609065715141965395470452149001408090033625752972502842795521853380044460010739, 887.1265884165220982258796784681620290025788767111506814162003630445239152297569716120739127607076463),
new_x_y(41.80232794992022324766448931101503163010863610989426673894473684443238265394305525506376941341290178, 887.3105053821998687097549536721715601203776341647477803230356785177312043609191464127841429231022137),
new_x_y(43.84690172491579120586807205051586672067022700488839266888611111356751502360655832478953438082472965, 887.4944223478776391936302288761810912381763916183448792298709939909384934920813212134943730854967812),
new_x_y(45.89147549991135916407165479001670181123181789988251859882748538270264739327006139451529934823655753, 887.6783393135554096775055040801906223559751490719419781367063094641457826232434960142046032478913486),
new_x_y(47.93604927490692712227523752951753690179340879487664452876885965183777976293356446424106431564838541, 887.862256279233180161380779284200153473773906525539077043541624937353071754405670814914833410285916),
new_x_y(49.98062304990249508047882026901837199235499968987077045871023392097291213259706753396682928306021328, 888.0461732449109506452560544882096845915726639791361759503769404105603608855678456156250635726804835),
new_x_y(52.02519682489806303868240300851920708291659058486489638865160819010804450226057060369259425047204116, 888.2300902105887211291313296922192157093714214327332748572122558837676500167300204163352937350750509),
new_x_y(54.06977059989363099688598574802004217347818147985902231859298245924317687192407367341835921788386904, 888.4140071762664916130066048962287468271701788863303737640475713569749391478921952170455238974696183),
new_x_y(56.11434437488919895508956848752087726403977237485314824853435672837830924158757674314412418529569691, 888.5979241419442620968818801002382779449689363399274726708828868301822282790543700177557540598641857),
new_x_y(58.15891814988476691329315122702171235460136326984727417847573099751344161125107981286988915270752479, 888.7818411076220325807571553042478090627676937935245715777182023033895174102165448184659842222587532),
new_x_y(60.20349192488033487149673396652254744516295416484140010841710526664857398091458288259565412011935267, 888.9657580732998030646324305082573401805664512471216704845535177765968065413787196191762143846533206),
new_x_y(62.24806569987590282970031670602338253572454505983552603835847953578370635057808595232141908753118054, 889.149675038977573548507705712266871298365208700718769391388833249804095672540894419886444547047888),
new_x_y(64.29263947487147078790389944552421762628613595482965196829985380491883872024158902204718405494300842, 889.3335920046553440323829809162764024161639661543158682982241487230113848037030692205966747094424555),
new_x_y(66.3372132498670387461074821850250527168477268498237778982412280740539710899050920917729490223548363, 889.5175089703331145162582561202859335339627236079129672050594641962186739348652440213069048718370229),
new_x_y(68.38178702486260670431106492452588780740931774481790382818260234318910345956859516149871398976666417, 889.7014259360108850001335313242954646517614810615100661118947796694259630660274188220171350342315903),
new_x_y(70.42636079985817466251464766402672289797090863981202975812397661232423582923209823122447895717849205, 889.8853429016886554840088065283049957695602385151071650187300951426332521971895936227273651966261577),
new_x_y(72.47093457485374262071823040352755798853249953480615568806535088145936819889560130095024392459031992, 890.0692598673664259678840817323145268873589959687042639255654106158405413283517684234375953590207252),
new_x_y(74.5155083498493105789218131430283930790940904298002816180067251505945005685591043706760088920021478, 890.2531768330441964517593569363240580051577534223013628324007260890478304595139432241478255214152926),
new_x_y(76.56008212484487853712539588252922816965568132479440754794809941972963293822260744040177385941397568, 890.43709379872196693563463214033358912295651087589846173923604156225511959067611802485805568380986),
new_x_y(78.60465589984044649532897862203006326021727221978853347788947368886476530788611051012753882682580355, 890.6210107643997374195099073443431202407552683294955606460713570354624087218382928255682858462044275),
new_x_y(80.64922967483601445353256136153089835077886311478265940783084795799989767754961357985330379423763143, 890.8049277300775079033851825483526513585540257830926595529066725086696978530004676262785160085989949),
new_x_y(82.69380344983158241173614410103173344134045400977678533777222222713503004721311664957906876164945931, 890.9888446957552783872604577523621824763527832366897584597419879818769869841626424269887461709935623),
new_x_y(84.73837722482715036993972684053256853190204490477091126771359649627016241687661971930483372906128718, 891.1727616614330488711357329563717135941515406902868573665773034550842761153248172276989763333881298),
new_x_y(86.78295099982271832814330958003340362246363579976503719765497076540529478654012278903059869647311506, 891.3566786271108193550110081603812447119502981438839562734126189282915652464869920284092064957826972),
new_x_y(88.82752477481828628634689231953423871302522669475916312759634503454042715620362585875636366388494294, 891.5405955927885898388862833643907758297490555974810551802479344014988543776491668291194366581772646),
new_x_y(90.87209854981385424455047505903507380358681758975328905753771930367555952586712892848212863129677081, 891.724512558466360322761558568400306947547813051078154087083249874706143508811341629829666820571832),
new_x_y(92.91667232480942220275405779853590889414840848474741498747909357281069189553063199820789359870859869, 891.9084295241441308066368337724098380653465705046752529939185653479134326399735164305398969829663995),
new_x_y(94.96124609980499016095764053803674398470999937974154091742046784194582426519413506793365856612042657, 892.0923464898219012905121089764193691831453279582723519007538808211207217711356912312501271453609669),
new_x_y(97.00581987480055811916122327753757907527159027473566684736184211108095663485763813765942353353225444, 892.2762634554996717743873841804289003009440854118694508075891962943280109022978660319603573077555343),
new_x_y(99.05039364979612607736480601703841416583318116972979277730321638021608900452114120738518850094408232, 892.4601804211774422582626593844384314187428428654665497144245117675353000334600408326705874701501018),
new_x_y(101.0949674247916940355683887565392492563947720647239187072445906493512213741846442771109534683559102, 892.6440973868552127421379345884479625365416003190636486212598272407425891646222156333808176325446692),
new_x_y(103.1395411997872619937719714960400843469563629597180446371859649184863537438481473468367184357677381, 892.8280143525329832260132097924574936543403577726607475280951427139498782957843904340910477949392366),
new_x_y(105.1841149747828299519755542355409194375179538547121705671273391876214861135116504165624834031795659, 893.0119313182107537098884849964670247721391152262578464349304581871571674269465652348012779573338041),
new_x_y(107.2286887497783979101791369750417545280795447497062964970687134567566184831751534862882483705913938, 893.1958482838885241937637602004765558899378726798549453417657736603644565581087400355115081197283715),
new_x_y(109.2732625247739658683827197145425896186411356447004224270100877258917508528386565560140133380032217, 893.3797652495662946776390354044860870077366301334520442486010891335717456892709148362217382821229389),
new_x_y(111.3178362997695338265863024540434247092027265396945483569514619950268832225021596257397783054150496, 893.5636822152440651615143106084956181255353875870491431554364046067790348204330896369319684445175063),
new_x_y(113.3624100747651017847898851935442597997643174346886742868928362641620155921656626954655432728268775, 893.7475991809218356453895858125051492433341450406462420622717200799863239515952644376421986069120738),
new_x_y(115.4069838497606697429934679330450948903259083296828002168342105332971479618291657651913082402387053, 893.9315161465996061292648610165146803611329024942433409691070355531936130827574392383524287693066412),
new_x_y(117.4515576247562377011970506725459299808874992246769261467755848024322803314926688349170732076505332, 894.1154331122773766131401362205242114789316599478404398759423510264009022139196140390626589317012086),
new_x_y(119.4961313997518056594006334120467650714490901196710520767169590715674127011561719046428381750623611, 894.2993500779551470970154114245337425967304174014375387827776664996081913450817888397728890940957761),
new_x_y(121.540705174747373617604216151547600162010681014665178006658333340702545070819674974368603142474189, 894.4832670436329175808906866285432737145291748550346376896129819728154804762439636404831192564903435),
new_x_y(123.5852789497429415758077988910484352525722719096593039365997076098376774404831780440943681098860168, 894.6671840093106880647659618325528048323279323086317365964482974460227696074061384411933494188849109),
new_x_y(125.6298527247385095340113816305492703431338628046534298665410818789728098101466811138201330772978447, 894.8511009749884585486412370365623359501266897622288355032836129192300587385683132419035795812794784),
new_x_y(127.6744264997340774922149643700501054336954536996475557964824561481079421798101841835458980447096726, 895.0350179406662290325165122405718670679254472158259344101189283924373478697304880426138097436740458),
new_x_y(129.7190002747296454504185471095509405242570445946416817264238304172430745494736872532716630121215005, 895.2189349063439995163917874445813981857242046694230333169542438656446370008926628433240399060686132),
new_x_y(131.7635740497252134086221298490517756148186354896358076563652046863782069191371903229974279795333283, 895.4028518720217700002670626485909293035229621230201322237895593388519261320548376440342700684631806),
new_x_y(133.8081478247207813668257125885526107053802263846299335863065789555133392888006933927231929469451562, 895.5867688376995404841423378526004604213217195766172311306248748120592152632170124447445002308577481),
new_x_y(135.8527215997163493250292953280534457959418172796240595162479532246484716584641964624489579143569841, 895.7706858033773109680176130566099915391204770302143300374601902852665043943791872454547303932523155),
new_x_y(137.897295374711917283232878067554280886503408174618185446189327493783604028127699532174722881768812, 895.9546027690550814518928882606195226569192344838114289442955057584737935255413620461649605556468829),
new_x_y(139.9418691497074852414364608070551159770649990696123113761307017629187363977912026019004878491806398, 896.1385197347328519357681634646290537747179919374085278511308212316810826567035368468751907180414504),
new_x_y(141.9864429247030531996400435465559510676265899646064373060720760320538687674547056716262528165924677, 896.3224367004106224196434386686385848925167493910056267579661367048883717878657116475854208804360178),
new_x_y(144.0310166996986211578436262860567861581881808596005632360134503011890011371182087413520177840042956, 896.5063536660883929035187138726481160103155068446027256648014521780956609190278864482956510428305852),
new_x_y(146.0755904746941891160472090255576212487497717545946891659548245703241335067817118110777827514161235, 896.6902706317661633873939890766576471281142642981998245716367676513029500501900612490058812052251526),
new_x_y(148.1201642496897570742507917650584563393113626495888150958961988394592658764452148808035477188279514, 896.8741875974439338712692642806671782459130217517969234784720831245102391813522360497161113676197201),
new_x_y(150.1647380246853250324543745045592914298729535445829410258375731085943982461087179505293126862397792, 897.0581045631217043551445394846767093637117792053940223853073985977175283125144108504263415300142875),
new_x_y(152.2093117996808929906579572440601265204345444395770669557789473777295306157722210202550776536516071, 897.2420215287994748390198146886862404815105366589911212921427140709248174436765856511365716924088549),
new_x_y(154.253885574676460948861539983560961610996135334571192885720321646864662985435724089980842621063435, 897.4259384944772453228950898926957715993092941125882201989780295441321065748387604518468018548034224),
new_x_y(156.2984593496720289070651227230617967015577262295653188156616959159997953550992271597066075884752629, 897.6098554601550158067703650967053027171080515661853191058133450173393957060009352525570320171979898),
new_x_y(158.3430331246675968652687054625626317921193171245594447456030701851349277247627302294323725558870907, 897.7937724258327862906456403007148338349068090197824180126486604905466848371631100532672621795925572),
new_x_y(160.3876068996631648234722882020634668826809080195535706755444444542700600944262332991581375232989186, 897.9776893915105567745209155047243649527055664733795169194839759637539739683252848539774923419871247),
new_x_y(162.4321806746587327816758709415643019732424989145476966054858187234051924640897363688839024907107465, 898.1616063571883272583961907087338960705043239269766158263192914369612630994874596546877225043816921),
new_x_y(164.4767544496543007398794536810651370638040898095418225354271929925403248337532394386096674581225744, 898.3455233228660977422714659127434271883030813805737147331546069101685522306496344553979526667762595),
new_x_y(166.5213282246498686980830364205659721543656807045359484653685672616754572034167425083354324255344022, 898.5294402885438682261467411167529583061018388341708136399899223833758413618118092561081828291708269),
new_x_y(168.5659019996454366562866191600668072449272715995300743953099415308105895730802455780611973929462301, 898.7133572542216387100220163207624894239005962877679125468252378565831304929739840568184129915653944),
new_x_y(170.610475774641004614490201899567642335488862494524200325251315799945721942743748647786962360358058, 898.8972742198994091938972915247720205416993537413650114536605533297904196241361588575286431539599618),
new_x_y(172.6550495496365725726937846390684774260504533895183262551926900690808543124072517175127273277698859, 899.0811911855771796777725667287815516594981111949621103604958688029977087552983336582388733163545292),
new_x_y(174.6996233246321405308973673785693125166120442845124521851340643382159866820707547872384922951817137, 899.2651081512549501616478419327910827772968686485592092673311842762049978864605084589491034787490967),
new_x_y(176.7441970996277084891009501180701476071736351795065781150754386073511190517342578569642572625935416, 899.4490251169327206455231171368006138950956261021563081741664997494122870176226832596593336411436641),
new_x_y(178.7887708746232764473045328575709826977352260745007040450168128764862514213977609266900222300053695, 899.6329420826104911293983923408101450128943835557534070810018152226195761487848580603695638035382315),
new_x_y(180.8333446496188444055081155970718177882968169694948299749581871456213837910612639964157871974171974, 899.8168590482882616132736675448196761306931410093505059878371306958268652799470328610797939659327989),
new_x_y(182.8779184246144123637116983365726528788584078644889559048995614147565161607247670661415521648290253, 900.0007760139660320971489427488292072484918984629476048946724461690341544111092076617900241283273664),
new_x_y(184.9224921996099803219152810760734879694199987594830818348409356838916485303882701358673171322408531, 900.1846929796438025810242179528387383662906559165447038015077616422414435422713824625002542907219338),
new_x_y(186.967065974605548280118863815574323059981589654477207764782309953026780900051773205593082099652681, 900.3686099453215730648994931568482694840894133701418027083430771154487326734335572632104844531165012),
new_x_y(189.0116397496011162383224465550751581505431805494713336947236842221619132697152762753188470670645089, 900.5525269109993435487747683608578006018881708237389016151783925886560218045957320639207146155110687),
new_x_y(191.0562135245966841965260292945759932411047714444654596246650584912970456393787793450446120344763368, 900.7364438766771140326500435648673317196869282773360005220137080618633109357579068646309447779056361),
new_x_y(193.1007872995922521547296120340768283316663623394595855546064327604321780090422824147703770018881646, 900.9203608423548845165253187688768628374856857309330994288490235350706000669200816653411749403002035),
new_x_y(195.1453610745878201129331947735776634222279532344537114845478070295673103787057854844961419692999925, 901.104277808032655000400593972886393955284443184530198335684339008277889198082256466051405102694771),
new_x_y(197.1899348495833880711367775130784985127895441294478374144891812987024427483692885542219069367118204, 901.2881947737104254842758691768959250730832006381272972425196544814851783292444312667616352650893384),
new_x_y(199.2345086245789560293403602525793336033511350244419633444305555678375751180327916239476719041236483, 901.4721117393881959681511443809054561908819580917243961493549699546924674604066060674718654274839058),
new_x_y(201.2790823995745239875439429920801686939127259194360892743719298369727074876962946936734368715354761, 901.6560287050659664520264195849149873086807155453214950561902854278997565915687808681820955898784732),
new_x_y(203.323656174570091945747525731581003784474316814430215204313304106107839857359797763399201838947304, 901.8399456707437369359016947889245184264794729989185939630256009011070457227309556688923257522730407),
new_x_y(205.3682299495656599039511084710818388750359077094243411342546783752429722270233008331249668063591319, 902.0238626364215074197769699929340495442782304525156928698609163743143348538931304696025559146676081),
new_x_y(207.4128037245612278621546912105826739655974986044184670641960526443781045966868039028507317737709598, 902.2077796020992779036522451969435806620769879061127917766962318475216239850553052703127860770621755),
new_x_y(209.4573774995567958203582739500835090561590894994125929941374269135132369663503069725764967411827877, 902.391696567777048387527520400953111779875745359709890683531547320728913116217480071023016239456743),
new_x_y(211.5019512745523637785618566895843441467206803944067189240788011826483693360138100423022617085946155, 902.5756135334548188714027956049626428976745028133069895903668627939362022473796548717332464018513104),
new_x_y(213.5465250495479317367654394290851792372822712894008448540201754517835017056773131120280266760064434, 902.7595304991325893552780708089721740154732602669040884972021782671434913785418296724434765642458778),
new_x_y(215.5910988245434996949690221685860143278438621843949707839615497209186340753408161817537916434182713, 902.9434474648103598391533460129817051332720177205011874040374937403507805097040044731537067266404452),
new_x_y(217.6356725995390676531726049080868494184054530793890967139029239900537664450043192514795566108300992, 903.1273644304881303230286212169912362510707751740982863108728092135580696408661792738639368890350127),
new_x_y(219.680246374534635611376187647587684508967043974383222643844298259188898814667822321205321578241927, 903.3112813961659008069038964210007673688695326276953852177081246867653587720283540745741670514295801),
new_x_y(221.7248201495302035695797703870885195995286348693773485737856725283240311843313253909310865456537549, 903.4951983618436712907791716250102984866682900812924841245434401599726479031905288752843972138241475),
new_x_y(223.7693939245257715277833531265893546900902257643714745037270467974591635539948284606568515130655828, 903.679115327521441774654446829019829604467047534889583031378755633179937034352703675994627376218715),
new_x_y(225.8139676995213394859869358660901897806518166593656004336684210665942959236583315303826164804774107, 903.8630322931992122585297220330293607222658049884866819382140711063872261655148784767048575386132824),
new_x_y(227.8585414745169074441905186055910248712134075543597263636097953357294282933218346001083814478892385, 904.0469492588769827424049972370388918400645624420837808450493865795945152966770532774150877010078498),
new_x_y(229.9031152495124754023941013450918599617749984493538522935511696048645606629853376698341464153010664, 904.2308662245547532262802724410484229578633198956808797518847020528018044278392280781253178634024173),
new_x_y(231.9476890245080433605976840845926950523365893443479782234925438739996930326488407395599113827128943, 904.4147831902325237101555476450579540756620773492779786587200175260090935590014028788355480257969847),
new_x_y(233.9922627995036113188012668240935301428981802393421041534339181431348254023123438092856763501247222, 904.5987001559102941940308228490674851934608348028750775655553329992163826901635776795457781881915521),
new_x_y(236.03683657449917927700484956359436523345977113433623008337529241226995777197584687901144131753655, 904.7826171215880646779060980530770163112595922564721764723906484724236718213257524802560083505861195),
new_x_y(238.0814103494947472352084323030952003240213620293303560133166666814050901416393499487372062849483779, 904.966534087265835161781373257086547429058349710069275379225963945630960952487927280966238512980687),
new_x_y(240.1259841244903151934120150425960354145829529243244819432580409505402225113028530184629712523602058, 905.1504510529436056456566484610960785468571071636663742860612794188382500836501020816764686753752544),
new_x_y(242.1705578994858831516155977820968705051445438193186078731994152196753548809663560881887362197720337, 905.3343680186213761295319236651056096646558646172634731928965948920455392148122768823866988377698218),
new_x_y(244.2151316744814511098191805215977055957061347143127338031407894888104872506298591579145011871838615, 905.5182849842991466134071988691151407824546220708605720997319103652528283459744516830969290001643893),
new_x_y(246.2597054494770190680227632610985406862677256093068597330821637579456196202933622276402661545956894, 905.7022019499769170972824740731246719002533795244576710065672258384601174771366264838071591625589567),
new_x_y(248.3042792244725870262263460005993757768293165043009856630235380270807519899568652973660311220075173, 905.8861189156546875811577492771342030180521369780547699134025413116674066082988012845173893249535241),
new_x_y(250.3488529994681549844299287401002108673909073992951115929649122962158843596203683670917960894193452, 906.0700358813324580650330244811437341358508944316518688202378567848746957394609760852276194873480916),
new_x_y(252.3934267744637229426335114796010459579524982942892375229062865653510167292838714368175610568311731, 906.253952847010228548908299685153265253649651885248967727073172258081984870623150885937849649742659),
new_x_y(254.4380005494592909008370942191018810485140891892833634528476608344861490989473745065433260242430009, 906.4378698126879990327835748891627963714484093388460666339084877312892740017853256866480798121372264),
new_x_y(256.4825743244548588590406769586027161390756800842774893827890351036212814686108775762690909916548288, 906.6217867783657695166588500931723274892471667924431655407438032044965631329475004873583099745317938),
new_x_y(258.5271480994504268172442596981035512296372709792716153127304093727564138382743806459948559590666567, 906.8057037440435400005341252971818586070459242460402644475791186777038522641096752880685401369263613),
new_x_y(260.5717218744459947754478424376043863201988618742657412426717836418915462079378837157206209264784846, 906.9896207097213104844094005011913897248446816996373633544144341509111413952718500887787702993209287),
new_x_y(262.6162956494415627336514251771052214107604527692598671726131579110266785776013867854463858938903124, 907.1735376753990809682846757052009208426434391532344622612497496241184305264340248894890004617154961),
new_x_y(264.6608694244371306918550079166060565013220436642539931025545321801618109472648898551721508613021403, 907.3574546410768514521599509092104519604421966068315611680850650973257196575961996901992306241100636),
new_x_y(266.7054431994326986500585906561068915918836345592481190324959064492969433169283929248979158287139682, 907.541371606754621936035226113219983078240954060428660074920380570533008788758374490909460786504631),
new_x_y(268.7500169744282666082621733956077266824452254542422449624372807184320756865918959946236807961257961, 907.7252885724323924199105013172295141960397115140257589817556960437402979199205492916196909488991984),
new_x_y(270.7945907494238345664657561351085617730068163492363708923786549875672080562553990643494457635376239, 907.9092055381101629037857765212390453138384689676228578885910115169475870510827240923299211112937658),
new_x_y(272.8391645244194025246693388746093968635684072442304968223200292567023404259189021340752107309494518, 908.0931225037879333876610517252485764316372264212199567954263269901548761822448988930401512736883333),
new_x_y(274.8837382994149704828729216141102319541299981392246227522614035258374727955824052038009756983612797, 908.2770394694657038715363269292581075494359838748170557022616424633621653134070736937503814360829007),
new_x_y(276.9283120744105384410765043536110670446915890342187486822027777949726051652459082735267406657731076, 908.4609564351434743554116021332676386672347413284141546090969579365694544445692484944606115984774681),








new_x_y(277.5, 908.700000000000045474735088646411895751953125),
new_x_y(279.5511975518712469203129563022197502209425523761746826321165452640327736147175118052804049042224443, 908.7818289365759833172648235094105052047559416597009314892728355232594947231357522600984283836916318),
new_x_y(281.6022774044789002167961452863614500768874668722438039530984836972182247318695835891894164126362964, 908.8665554834448776742337221295013367312360716563584284362324905501332545230164196478993110651834267),
new_x_y(283.6531092545864988761694549919948790050704681873297651166024673049613844134799692942742249904649205, 908.9570767236778716296049169678472041269005352669498479693527639836880340111607593009686554563873116),
new_x_y(285.7035375949887794070951924686569514083523593314531821807895198143634019588450170266675351946164095, 909.0562885945303385239497889798929652964313092013245303541627289211615029578998916828141327125944516),
new_x_y(287.7533691233969414931031989484747853927554763475738430769418297738555607537130584752751083329963889, 909.1670850861460227450653475018603467140148763131932298684610575767178764116183264264002178491566651),
new_x_y(289.8023601669634698436406215038671037184203550696718261435669560656672461313272393720684519687266249, 909.2923571663263835459318714053398038401969662310103232810350099534944486118488555419630866682939324),
new_x_y(291.8502041319843420968066088368179589001444845884171161916120481536867636416312971350081906437537716, 909.4349913403077872818733960757778455186453798895988366715549880175740655951878533017537504698023359),
new_x_y(293.8965189920187869449187050619548558967047661341562737461868153029071320398350327986219207102365762, 909.5978677547796636584901605113029256525629213831962439100766541251675981308853286695017526806556998),
new_x_y(295.9408348322876814455037588087451606271415192225736650260751142437320634453329471787461662742768819, 909.7838577558304220895011516400383480734368118255562668821764606250072421336212767122946158154137759),
new_x_y(297.982581473744410921540174498776594321798901311576126788596844892148398566437407328925794881114806, 909.9958208111850409418599847701402368164227103312883685556468947745840330838147539962066612866609392),
new_x_y(300.0210762066463075928479110028541074901547092634165645296972223749353226232480187353668990004890272, 910.2366007080708449792640103291471633275815418122442966616969123397958978148942763319261827031134823),
new_x_y(302.055511670775766807083249278292966652467531176303962729112664889614526896807742700186608130080789, 910.5090209393997622465350558676953865009866498790782488212956973360711078611284577233562720583504488),
new_x_y(304.0849439276469977828234925826181616365204659809509582064066433294639415477683868185823156639022621, 910.8158791927813521505399729295907235731561854258078024980794549359505559612932233548583107116987216),
new_x_y(306.108280779058830715074587565038074206288350785783708580518958919145507198925712501596743501887865, 911.1599408592871567566525238441779122131189043556980650539629789941455568244638991296415556048735302),
new_x_y(308.1242703961786760974324741371489974945782705550551568315047034155149093799723317147194491102187181, 911.5439314819899390183549587595815460835265798635766046430764665649019705241261885138375595939801517),
new_x_y(310.1314903339192528741733475818042036761443852255030374737657764811211338985079626335173801056007311, 911.9705280682273972351113892701213270557404850242493773677443761671869768784944111829014073351684225),
new_x_y(312.12833701663676640281224539211689308582988126385354297642018198940672611562617103477202775910054, 912.442349194424113095002021328984744259519507119316044658570641135611351402694731190725647151863773),
new_x_y(314.1130157930604784719313429693601774671052160417249287885705053688004301441038610862537423061711158, 912.9619438382906675001256937631645229826005747268096481127322362886904798323062464721709109838730476),
new_x_y(316.0835316707654802235129298773450595862898713652757816882206683386588020456773289696790644989733624, 913.5317788804542217576570813528603274742857110749648532277500005707397151788413139356020389197134002),
new_x_y(318.0376808533098511862541605562328793858428114283749763311581443477701437143812932978037673506565547, 914.1542252262141729786566503248451783435830119843868448830175571746361802534539341847546034012071313),
new_x_y(319.973043216239345220295179995774418882224956537970616277616382049443229141789494074906504164273563, 914.8315425083159963329566763051394539814515858933179244402973467049853389874607194738282900946479249),
new_x_y(321.8869758713582641555233024027891625274161066001243543495744134402476253538092134158941244786736813, 915.5658623435523192512480143563426627960440792838401873020531356371253732000333656668480478203369831),
new_x_y(323.7766079817889139548699684391632122879138277891420035710759677599763156799102823495909056734493994, 916.3591701297859237529433406926894688538628424580447928231781425261086070296730056447792231681551698),
new_x_y(325.6388370031802157004232738682306888868135882475305924892797912439642974991816341831414415702551265, 917.2132853857916879522669060544236176713936648243696505591111891117063702621873760798865623091036335),
new_x_y(327.470326538734582942096494765120392683985923291725628544124477970086007464679507880749700248123542, 918.129840654270129363710499371624183811714076784109901086253910297788541881084785579542120801054684),
new_x_y(329.2675060072250713542179169595596724776796945565031783123333216106263390611664599850695517743885906, 919.1102590086161683696895564208859610711945694708327851491447654463552179099663624052541607106327782),
new_x_y(331.0265723335628943099066145474480000438142545343887562664849906139107327999374679240291452523151339, 920.1557302266352460463418124948199758093931495229575554186534964864146530504513421905996903851046509),
new_x_y(332.7434938804056071601064100690319805242571055325366874781583646981629228447505617456056654013894855, 921.2671857194619997078748954464157181622767857474315907206216184937840475061430044047409071321064471),
new_x_y(334.4140168463914570844957031356712847404000702687704090606970661308165151799022728563061453023188819, 922.4452723314999369910592117579460283371132100933150265423337742710696079595171169217581041673508056),
new_x_y(336.0336743614349267580740017501022442017736305823373247622754941420406471925587867692452505678870377, 923.6903251572715590748906391380103481582742539666492066961464875819391345936229817845150742846551139),
new_x_y(337.5977985116796391850691482294289472475926604698089640654675549181302117579158053627398597614110347, 925.002339553609592057842182635117758649736229051030916380771370581792015823571644941218928091441588),
new_x_y(339.1015355257051119984984444531088357923405965666319704455006680759547633322305763001943108540738412, 926.3809425605410957652712188795547883528169863411468993258716000527459192490742280780192117686695599),
new_x_y(340.5398643489248074188805020536961351963067722594107815055040450680285323046283197767062708931035103, 927.8253639813662305168603676615012036863287924069404925173980330708453561446433322680101993404163779),
new_x_y(341.9076188242746624782006247051221687767645636098042600387295220481998730830868118648353620652260539, 929.3344074115925309961555955083399322104604383649061195737280576881180977294640619682730712137851963),
new_x_y(343.1995136837389466998025220146116846963544811166342677916889459466914500329661142605812974966463949, 930.9064215472566416980382468238413834712671033223687665487980757842020672215646895154995852260376578),
new_x_y(344.4101745364518595949061495300214202376117189035962874482324736296489458360626780887171181845350884, 932.5392721453662259165110804871569915387690948099297239828474805337527054755706402618628032774288777),
new_x_y(345.5341720145091605682681735793766649734189512510101774560703080130418815746232720301250989733357236, 934.2303150522494508003100140792582769899157238949827947724366857680450651258403958289096345376000466),
new_x_y(346.5660602066985902477388039798778038169140067557104431899255948712314695762533139135872954391720899, 935.9763707589315993656922685352001566253852518095548440085627073825779549300885982437627925561670648),
new_x_y(347.5004194726123626883986342150271887285109482515321767235590487810231895519049294938195369714485821, 937.7737009855841355991548578394101898751010222781515305937873751273085336234718523082399293841702512),
new_x_y(348.3319036845836361206130727904818477475098651169653685664414426617691543879591455229061878758188246, 939.6179878388142642366029479385900536882248194616933665437693850301028365835280884736816638592966118),
new_x_y(349.055291892195632580237635222353338495014257221322541083585578192836964743599177521789137867600159, 941.5043161251691797669969569692525478902850689397183940065593717445034614062511589982317284157664876),
new_x_y(349.6655443434303593794465313375656704255702455630732951556139305083211970437107536483301847667992421, 943.4271594406864155396666866302590767048466486576415849902547693104283508463815155691870188096770778),
new_x_y(350.1578627276378484124696763968307489130086828668439647951828241856794901198284867745493640358640602, 945.3803706884789926152785096358225091926799374450309284154527779297745049662199841578636547805460252),
new_x_y(350.5277544283244687695350827619931249291097013799043538415415716827893740170782678921720535664774797, 947.3571777029349740250060270081508775846071422097748704730649405056301974608965807765751138252319115),
new_x_y(350.7711004883367998348137431439798610591741248184010881071080972381464901177424914400338199144001314, 949.3501846787600226090248031705548921806894242895433027114677819157503660292325149753221328888364924),






# # new_x_y(336.5594500302790230882513139793115180280060733295531819248512298863056611040170877940902297528385642, 958.7204257222718650404928549152077130137143443156979755304509444889272199716502370505491163460726904),
# # new_x_y(338.4077127282111020868504994717033986942281991497190044339805754436658887055406260830729245186665235, 959.6137522981088195636596349959187016960363388503694362363168025129125021943224555100642097374048699),
# # new_x_y(340.2559754261431810854496849640952793604503249698848269431099210010261163070641643720556192844944828, 960.5070788739457740868264150766296903783583333850408969421826605368977844169946739695793031287370494),
# new_x_y(342.1042381240752600840488704564871600266724507900506494522392665583863439085877026610383140503224421, 961.400405449782728609993195157340679060680327919712357648048518560883066639666892429094396520069229),
# new_x_y(343.9525008220073390826480559488790406928945766102164719613686121157465715101112409500210088161504014, 962.2937320256196831331599752380516677430023224543838183539143765848683488623391108886094899114014085),
# new_x_y(345.8007635199394180812472414412709213591167024303822944704979576731067991116347792390037035819783607, 963.187058601456637656326755318762656425324316989055279059780234608853631085011329348124583302733588),
# new_x_y(347.64902621787149707984642693366280202533882825054811697962730323046702671315831752798639834780632, 964.0803851772935921794935353994736451076463115237267397656460926328389133076835478076396766940657675),
# new_x_y(349.4972889158035760784456124260546826915609540707139394887566487878272543146818558169690931136342793, 964.973711753130546702660315480184633789968306058398200471511950656824195530355766267154770085397947),
# new_x_y(351.3455516137356550770447979184465633577830798908797619978859943451874819162053941059517878794622386, 965.8670383289675012258270955608956224722903005930696611773778086808094777530279847266698634767301265),
# new_x_y(353.1938143116677340756439834108384440240052057110455845070153399025477095177289323949344826452901979, 966.7603649048044557489938756416066111546122951277411218832436667047947599757002031861849568680623061),
# new_x_y(355.0420770095998130742431689032303246902273315312114070161446854599079371192524706839171774111181572, 967.6536914806414102721606557223175998369342896624125825891095247287800421983724216457000502593944856),
# new_x_y(356.8903397075318920728423543956222053564494573513772295252740310172681647207760089728998721769461165, 968.5470180564783647953274358030285885192562841970840432949753827527653244210446401052151436507266651),
# new_x_y(358.7386024054639710714415398880140860226715831715430520344033765746283923222995472618825669427740758, 969.4403446323153193184942158837395772015782787317555040008412407767506066437168585647302370420588446),
# new_x_y(360.5868651033960500700407253804059666888937089917088745435327221319886199238230855508652617086020351, 970.3336712081522738416609959644505658839002732664269647067070988007358888663890770242453304333910241),
# new_x_y(362.4351278013281290686399108727978473551158348118746970526620676893488475253466238398479564744299944, 971.2269977839892283648277760451615545662222678010984254125729568247211710890612954837604238247232036),
# new_x_y(364.2833904992602080672390963651897280213379606320405195617914132467090751268701621288306512402579537, 972.1203243598261828879945561258725432485442623357698861184388148487064533117335139432755172160553832),
# # new_x_y(366.131653197192287065838281857581608687560086452206342070920758804069302728393700417813346006085913, 973.0136509356631374111613362065835319308662568704413468243046728726917355344057324027906106073875627),
# # new_x_y(367.9799158951243660644374673499734893537822122723721645800501043614295303299172387067960407719138723, 973.9069775115000919343281162872945206131882514051128075301705308966770177570779508623057039987197422),
# # new_x_y(369.8281785930564450630366528423653700200043380925379870891794499187897579314407769957787355377418316, 974.8003040873370464574948963680055092955102459397842682360363889206622999797501693218207973900519217),
# # new_x_y(371.6764412909885240616358383347572506862264639127038095983087954761499855329643152847614303035697909, 975.6936306631740009806616764487164979778322404744557289419022469446475822024223877813358907813841012),
# new_x_y(373.5247039889206030602350238271491313524485897328696321074381410335102131344878535737441250693977502, 976.5869572390109555038284565294274866601542350091271896477681049686328644250946062408509841727162808),
# new_x_y(375.3729666868526820588342093195410120186707155530354546165674865908704407360113918627268198352257095, 977.4802838148479100269952366101384753424762295437986503536339629926181466477668247003660775640484603),
# new_x_y(377.2212293847847610574333948119328926848928413732012771256968321482306683375349301517095146010536688, 978.3736103906848645501620166908494640247982240784701110594998210166034288704390431598811709553806398),
# new_x_y(379.0694920827168400560325803043247733511149671933670996348261777055908959390584684406922093668816281, 979.2669369665218190733287967715604527071202186131415717653656790405887110931112616193962643467128193),
# new_x_y(380.9177547806489190546317657967166540173370930135329221439555232629511235405820067296749041327095874, 980.1602635423587735964955768522714413894422131478130324712315370645739933157834800789113577380449988),
# new_x_y(382.7660174785809980532309512891085346835592188336987446530848688203113511421055450186575988985375467, 981.0535901181957281196623569329824300717642076824844931770973950885592755384556985384264511293771783),
# new_x_y(384.614280176513077051830136781500415349781344653864567162214214377671578743629083307640293664365506, 981.9469166940326826428291370136934187540862022171559538829632531125445577611279169979415445207093579),
# new_x_y(386.4625428744451560504293222738922960160034704740303896713435599350318063451526215966229884301934653, 982.8402432698696371659959170944044074364081967518274145888291111365298399838001354574566379120415374),
# new_x_y(388.3108055723772350490285077662841766822255962941962121804729054923920339466761598856056831960214246, 983.7335698457065916891626971751153961187301912864988752946949691605151222064723539169717313033737169),
# new_x_y(390.1590682703093140476276932586760573484477221143620346896022510497522615481996981745883779618493839, 984.6268964215435462123294772558263848010521858211703360005608271845004044291445723764868246947058964),
# new_x_y(392.0073309682413930462268787510679380146698479345278571987315966071124891497232364635710727276773431, 985.5202229973805007354962573365373734833741803558417967064266852084856866518167908360019180860380759),
# new_x_y(393.8555936661734720448260642434598186808919737546936797078609421644727167512467747525537674935053024, 986.4135495732174552586630374172483621656961748905132574122925432324709688744890092955170114773702554),
# new_x_y(395.7038563641055510434252497358516993471140995748595022169902877218329443527703130415364622593332617, 987.306876149054409781829817497959350848018169425184718118158401256456251097161227755032104868702435),
# new_x_y(397.552119062037630042024435228243580013336225395025324726119633279193171954293851330519157025161221, 988.2002027248913643049965975786703395303401639598561788240242592804415333198334462145471982600346145),
# new_x_y(399.4003817599697090406236207206354606795583512151911472352489788365533995558173896195018517909891803, 989.093529300728318828163377659381328212662158494527639529890117304426815542505664674062291651366794),
# new_x_y(401.2486444579017880392228062130273413457804770353569697443783243939136271573409279084845465568171396, 989.9868558765652733513301577400923168949841530291991002357559753284120977651778831335773850426989735),
# new_x_y(403.0969071558338670378219917054192220120026028555227922535076699512738547588644661974672413226450989, 990.880182452402227874496937820803305577306147563870560941621833352397379987850101593092478434031153),
# new_x_y(404.9451698537659460364211771978111026782247286756886147626370155086340823603880044864499360884730582, 991.7735090282391823976637179015142942596281420985420216474876913763826622105223200526075718253633325),
# new_x_y(406.7934325516980250350203626902029833444468544958544372717663610659943099619115427754326308543010175, 992.6668356040761369208304979822252829419501366332134823533535494003679444331945385121226652166955121),
# new_x_y(408.6416952496301040336195481825948640106689803160202597808957066233545375634350810644153256201289768, 993.5601621799130914439972780629362716242721311678849430592194074243532266558667569716377586080276916),
# new_x_y(410.4899579475621830322187336749867446768911061361860822900250521807147651649586193533980203859569361, 994.4534887557500459671640581436472603065941257025564037650852654483385088785389754311528519993598711),
# new_x_y(412.3382206454942620308179191673786253431132319563519047991543977380749927664821576423807151517848954, 995.3468153315870004903308382243582489889161202372278644709511234723237911012111938906679453906920506),
# # new_x_y(414.1864833434263410294171046597705060093353577765177273082837432954352203680056959313634099176128547, 996.2401419074239550134976183050692376712381147718993251768169814963090733238834123501830387820242301),
# # new_x_y(416.034746041358420028016290152162386675557483596683549817413088852795447969529234220346104683440814, 997.1334684832609095366643983857802263535601093065707858826828395202943555465556308096981321733564096),
# # new_x_y(417.8830087392904990266154756445542673417796094168493723265424344101556755710527725093287994492687733, 998.0267950590978640598311784664912150358821038412422465885486975442796377692278492692132255646885892),
# # new_x_y(419.7312714372225780252146611369461480080017352370151948356717799675159031725763107983114942150967326, 998.9201216349348185829979585472022037182040983759137072944145555682649199919000677287283189560207687),
# new_x_y(421.5795341351546570238138466293380286742238610571810173448011255248761307740998490872941889809246919, 999.8134482107717731061647386279131924005260929105851680002804135922502022145722861882434123473529482),
# new_x_y(423.4277968330867360224130321217299093404459868773468398539304710822363583756233873762768837467526512, 1000.706774786608727629331518708624181082848087445256628706146271616235484437244504647758505738685128),
# new_x_y(425.2760595310188150210122176141217900066681126975126623630598166395965859771469256652595785125806105, 1001.600101362445682152498298789335169765170081979928089412012129640220766659916723107273599130017307),
# new_x_y(427.1243222289508940196114031065136706728902385176784848721891621969568135786704639542422732784085698, 1002.493427938282636675665078870046158447492076514599550117877987664206048882588941566788692521349487),
# new_x_y(428.9725849268829730182105885989055513391123643378443073813185077543170411801940022432249680442365291, 1003.386754514119591198831858950757147129814071049271010823743845688191331105261160026303785912681666),
# new_x_y(430.8208476248150520168097740912974320053344901580101298904478533116772687817175405322076628100644884, 1004.280081089956545721998639031468135812136065583942471529609703712176613327933378485818879304013846),
# new_x_y(432.6691103227471310154089595836893126715566159781759523995771988690374963832410788211903575758924477, 1005.173407665793500245165419112179124494458060118613932235475561736161895550605596945333972695346025),
# new_x_y(434.517373020679210014008145076081193337778741798341774908706544426397723984764617110173052341720407, 1006.066734241630454768332199192890113176780054653285392941341419760147177773277815404849066086678205),
# new_x_y(436.3656357186112890126073305684730740040008676185075974178358899837579515862881553991557471075483663, 1006.960060817467409291498979273601101859102049187956853647207277784132459995950033864364159478010384),
# new_x_y(438.2138984165433680112065160608649546702229934386734199269652355411181791878116936881384418733763256, 1007.853387393304363814665759354312090541424043722628314353073135808117742218622252323879252869342564),
# new_x_y(440.0621611144754470098057015532568353364451192588392424360945810984784067893352319771211366392042849, 1008.746713969141318337832539435023079223746038257299775058938993832103024441294470783394346260674743),
# # new_x_y(441.9104238124075260084048870456487160026672450790050649452239266558386343908587702661038314050322442, 1009.640040544978272860999319515734067906068032791971235764804851856088306663966689242909439652006923),
# # new_x_y(443.7586865103396050070040725380405966688893708991708874543532722131988619923823085550865261708602035, 1010.533367120815227384166099596445056588390027326642696470670709880073588886638907702424533043339102),
# # new_x_y(445.6069492082716840056032580304324773351114967193367099634826177705590895939058468440692209366881628, 1011.426693696652181907332879677156045270712021861314157176536567904058871109311126161939626434671282),
# # new_x_y(447.4552119062037630042024435228243580013336225395025324726119633279193171954293851330519157025161221, 1012.320020272489136430499659757867033953034016395985617882402425928044153331983344621454719826003461),
# # new_x_y(449.3034746041358420028016290152162386675557483596683549817413088852795447969529234220346104683440814, 1013.213346848326090953666439838578022635356010930657078588268283952029435554655563080969813217335641),
# # new_x_y(451.1517373020679210014008145076081193337778741798341774908706544426397723984764617110173052341720407, 1014.10667342416304547683321991928901131767800546532853929413414197601471777732778154048490660866782),
# # new_x_y(453.0, 1015.0),




# new_x_y(349.7294041423643415400564664585191652643724515306864232637702783699660531223704252428551627319031584, 949.5190798654238465753729975946611508640224657260235367082763888853337423923272007444149967142645556),
# new_x_y(350.1353256705494165738034470355241804931334532027032962437043709719094385866007731012932672209206549, 951.5019310713495519016892111914055394183910221195582028717174469956638073124696009608835961157572371),
# new_x_y(350.6527977708149315995005411018955520964469872894594102825452564553251975145031207929765364953434657, 953.4613525614729675600542069094623562583375375532826001751035660153034639959102985497375760871895185),
# new_x_y(351.2774313054684974192233518770283798482714297701912267080371853299796034110540433526011951074594971, 955.3918904116264119960885560378540171830096347821220035575760121871842093434496730516783774288158856),
# new_x_y(352.0046366191428443544079226010191460078765630575664304308226264742900647646840906189617587219646765, 957.2886614722967454121791096936618450106944907735594158182813593624964885493565918142519125580347333),
# new_x_y(352.8296764868295586083490255171320265236725384486104096543156199259079499958530484423577754885613065, 959.1473422721494141565743072462521061055970147226422794811384221095365979140941254221463449742043111),
# new_x_y(353.74771500773970760686416307191666687152669253783457055224600424127137527460613511998573011987583, 960.9641542813922196028561353594355415023788342792736735703740964247829646736384360974391114507770604),
# new_x_y(354.7538623765854671138334908114060688741016511216381599860050642117096700121832346767791711784725086, 962.7358461195669071605807064553974155513602623418064795385472123566555116475889899119721733455069437),
# new_x_y(355.8432155212220907121398386920132763520358850457371349609772205631602193665883831990925702626965129, 964.4596732573985409651344330287981199092813515707582268618500902007230302383542277118308062860225468),
# new_x_y(357.010894645280530828661446735682112533419245592352052988862569018932152187545616563822245200501638, 966.1333757244756848431980460805301024145959317025747146954283777942626899390430992096839377377836277),
# new_x_y(358.2520757567875770935623014338251193640525484309950912033339872701410363593382778088373871113310303, 967.7551542947579454923661569081827203942673509119527242107370050616923330778608569624783055573261765),
# new_x_y(359.5620192992147960539736307826863794574081176976954791870425692709672487972792839606681115248223785, 969.3236455810735715221964551004691153988968474504084252658326540241152283291805026713151211891491494),
# new_x_y(360.9360950303759071341873391257465186264889770838837674176865599204018317609901083174547061577500387, 970.8378964286263356344247262200912823685193541501032778296810586927773738527250957667098947847721193),
# new_x_y(362.3698033175977212140114028261438203786959476305005576015761489785284173033797478349398533093372478, 972.297337956715973066607645965912241095048523442950198296775607936412244952152171327836607644720483),
# new_x_y(363.8587930351382393449403348569449000396851516167258301341156939528911771179163167328959164886336284, 973.7017595579243945167932529927029196501153979080706126651082627491660057841070042085336172863467412),
# new_x_y(365.398876262442836988170763723868460339139609233586211255208381365811977724803417714643246989025941, 975.0512831253684025385958359622897058180169812629718078302245979322521604142046643499561046078099142),
# new_x_y(366.9860399900409716906497729660334824024580546628057451245586469649315866562594459832896928485111002, 976.3463377416172924303762647511505133020873712444761655344578817538720093315959465478587974810407279),
# new_x_y(368.6164550442074722535898773804311578886026654001831466987621225791955380820358307668765150042600986, 977.5876350277880947705629173686670175858067088524147151335865853208792884865021165914070504485386673),
# new_x_y(370.2864824424435860707266036332059561020084851630148206786799865276385839850769867283376976723177582, 978.7761453183571397737473068750898515833897598542626480744054051123282047181969259031043499116759079),
# new_x_y(371.9926773898508049538932670336190645318350456948073492460995762584005191750328308607717580586845439, 979.9130747964944018448836695884632271461432956082017870177632806539354995889648002357410865047873351),
# new_x_y(373.7317911220258842900796184797653604244580138823246218574616162188204733806572757661502049544817336, 980.9998436963105644389192429725796986729199396856184005670767441656547451986349694279433843265763001),
# new_x_y(375.5007707936198118885847455665678603777271341795414710801823472357268664651805691607853323700562576, 982.0380656523310940349252886838551757693821269673076027100547539496051652178696657633448748167129797),
# new_x_y(377.2967576035668055961321974424872268332568011851195455112648497349451761682350795063729212401844417, 983.0295282527606516691320648639829891873663405014900521232585638502729637053675911955233644800410662),
# new_x_y(379.1170833385593492121956900465191343780005915992756751053481354164348412880322105767595259692342178, 983.9761748316243175140375857653348841890744812988223018877164460569305729477224563562624436871675243),
# new_x_y(380.9592655059468139429584178209598716682073173673238561528264875266265836964952235199051474616048949, 984.8800875155907288902358014711329637574875501394944479805443351677451489666650303133928383061245601),
# new_x_y(382.821001216161128379614355663297835274292672820021592813328083071509903130638196848490160127703155, 985.7434715240955283244594833336876931203673928030094643052982124642225528924399084535007541378551321),
# new_x_y(384.700159963284744571510733347125160361201952682644583726752313971727395011096100216200258946443796, 986.5686407061737946060191258382694656748807648949333254987505708830427668024431755115590925931956238),
# new_x_y(386.5947754407053720495302338440212199109731981226309314500171449477859159379781331488589162236448325, 987.3580042840475673848336646406295630469820816593444550278144491651687283834508318664119632784241024),
# new_x_y(388.5030365171519518153945476583646595934979157995258529975962690713129044213449274115228104429638051, 988.1140547618614411796620062567020314459566911244510864054247562984851829228156393825948339805480833),
# new_x_y(390.4232774869541267488410108630285665144825318521287296337735613407155778586016612794058785987260348, 988.8393569478735696463603490795341953905682247511865245555009001624870513612591254528044381358626983),
# new_x_y(392.353967697265781548606584362203903744343953580552880190443555482155186642023966932829191592088933, 989.5365380297483850322887632629435883136321640468342003259508162076926245004736644865873500450128091),
# new_x_y(394.2937006443727059109056445226174269681062730886994663421236825382245482162086779110367931977548277, 990.2082786352198144393662408728995460445665541447109546045962415897602694283999060924438647764309985),
# new_x_y(396.2411826211757635094874035754001329555047255614597902625365147087364167152352290139815068222826467, 990.8573048041628678354745715396496322749381950424093548374040065186335100552803524432561918984112268),
# new_x_y(398.195220988597006559766347376285274972352002995362220475712548030558458504090396866501944306069985, 991.4863807928964805180297898452202252574919914917625232825523604854246865514458838118839097886848423),
# new_x_y(400.1547121350741087619124026441439616812431725415652904992172324883728447187068389077993552622830438, 992.0983026272185843446420148221287464991194445837397901038833679418149517138501172673409178638590552),
# new_x_y(402.1186291805516818542011252528855279721505885155673006778051018676103108334231714810156190449730709, 992.6958923171319784895459900175246746031579598038820338002275867135794734382961438556749528382522475),
# new_x_y(404.0860094744979446737111084295279585738100805708478815291308790890056785910964718879673084652315559, 993.2819926433534775665321914641907200929980811373278502158783829168313409935283759088896775510095805),
# new_x_y(406.0559419315130257682293643878259394905375796634161635391303600523813391789240386391325801589193153, 993.8594624234171282094238451142761861270503978587708226379507325645547765934952982587201480587780613),
# new_x_y(408.0275542430833421270897269719632257949308720425373037263506275029040737960113288354000953023054489, 994.4311721634051141487368616552062934771856613411579664237945989152473539543117066390871752219564854),
# # new_x_y(410.0, 995.0),








# # new_x_y(336.2715654706026577699085531998930602190367992988854549770096310334716282793562309563877906238177155, 962.7946277999295976802706698493295709507401806760739344511632537461840390220736241234540923045646184),
# # new_x_y(338.1620381508436152629878210665624689313691890604524945929837430582544070414240199062240011206429023, 963.594765548649351585904755237808299900721201684379731003697529291153679047148659402339884809575782),
# # new_x_y(340.0525108310845727560670889332318776437015788220195342089578550830371858034918088560602116174680891, 964.3949032973691054915388406262870288507022226926855275562318048361233190722236946812256773145869456),
# # new_x_y(341.9429835113255302491463567999012863560339685835865738249319671078199645655595978058964221142932758, 965.1950410460888593971729260147657578006832437009913241087660803810929590972987299601114698195981093),
# # new_x_y(343.8334561915664877422256246665706950683663583451536134409060791326027433276273867557326326111184626, 965.9951787948086133028070114032444867506642647092971206613003559260625991223737652389972623246092729),
# # new_x_y(345.7239288718074452353048925332401037806987481067206530568801911573855220896951757055688431079436494, 966.7953165435283672084410967917232157006452857176029172138346314710322391474488005178830548296204365),
# # new_x_y(347.6144015520484027283841603999095124930311378682876926728543031821683008517629646554050536047688362, 967.5954542922481211140751821802019446506263067259087137663689070160018791725238357967688473346316002),
# # new_x_y(349.504874232289360221463428266578921205363527629854732288828415206951079613830753605241264101594023, 968.3955920409678750197092675686806736006073277342145103189031825609715191975988710756546398396427638),
# # new_x_y(351.3953469125303177145426961332483299176959173914217719048025272317338583758985425550774745984192097, 969.1957297896876289253433529571594025505883487425203068714374581059411592226739063545404323446539274),
# # new_x_y(353.2858195927712752076219639999177386300283071529888115207766392565166371379663315049136850952443965, 969.9958675384073828309774383456381315005693697508261034239717336509107992477489416334262248496650911),
# # new_x_y(355.1762922730122327007012318665871473423606969145558511367507512812994159000341204547498955920695833, 970.7960052871271367366115237341168604505503907591318999765060091958804392728239769123120173546762547),
# # new_x_y(357.0667649532531901937804997332565560546930866761228907527248633060821946621019094045861060888947701, 971.5961430358468906422456091225955894005314117674376965290402847408500792978990121911978098596874183),
# # new_x_y(358.9572376334941476868597675999259647670254764376899303686989753308649734241696983544223165857199569, 972.396280784566644547879694511074318350512432775743493081574560285819719322974047470083602364698582),
# # new_x_y(360.8477103137351051799390354665953734793578661992569699846730873556477521862374873042585270825451437, 973.1964185332863984535137798995530473004934537840492896341088358307893593480490827489693948697097456),
# # new_x_y(362.7381829939760626730183033332647821916902559608240096006471993804305309483052762540947375793703304, 973.9965562820061523591478652880317762504744747923550861866431113757589993731241180278551873747209092),
# # new_x_y(364.6286556742170201660975711999341909040226457223910492166213114052133097103730652039309480761955172, 974.7966940307259062647819506765105052004554958006608827391773869207286393981991533067409798797320728),
# new_x_y(366.519128354457977659176839066603599616355035483958088832595423429996088472440854153767158573020704, 975.5968317794456601704160360649892341504365168089666792917116624656982794232741885856267723847432365),
# new_x_y(368.4096010346989351522561069332730083286874252455251284485695354547788672345086431036033690698458908, 976.3969695281654140760501214534679631004175378172724758442459380106679194483492238645125648897544001),
# new_x_y(370.3000737149398926453353747999424170410198150070921680645436474795616459965764320534395795666710776, 977.1971072768851679816842068419466920503985588255782723967802135556375594734242591433983573947655637),
# new_x_y(372.1905463951808501384146426666118257533522047686592076805177595043444247586442210032757900634962644, 977.9972450256049218873182922304254210003795798338840689493144891006071994984992944222841498997767274),
# new_x_y(374.0810190754218076314939105332812344656845945302262472964918715291272035207120099531120005603214511, 978.797382774324675792952377618904149950360600842189865501848764645576839523574329701169942404787891),
# new_x_y(375.9714917556627651245731783999506431780169842917932869124659835539099822827797989029482110571466379, 979.5975205230444296985864630073828789003416218504956620543830401905464795486493649800557349097990546),
# new_x_y(377.8619644359037226176524462666200518903493740533603265284400955786927610448475878527844215539718247, 980.3976582717641836042205483958616078503226428588014586069173157355161195737244002589415274148102183),
# new_x_y(379.7524371161446801107317141332894606026817638149273661444142076034755398069153768026206320507970115, 981.1977960204839375098546337843403368003036638671072551594515912804857595987994355378273199198213819),
# new_x_y(381.6429097963856376038109819999588693150141535764944057603883196282583185689831657524568425476221983, 981.9979337692036914154887191728190657502846848754130517119858668254553996238744708167131124248325455),
# new_x_y(383.533382476626595096890249866628278027346543338061445376362431653041097331050954702293053044447385, 982.7980715179234453211228045612977947002657058837188482645201423704250396489495060955989049298437092),
# new_x_y(385.4238551568675525899695177332976867396789330996284849923365436778238760931187436521292635412725718, 983.5982092666431992267568899497765236502467268920246448170544179153946796740245413744846974348548728),
# new_x_y(387.3143278371085100830487855999670954520113228611955246083106557026066548551865326019654740380977586, 984.3983470153629531323909753382552526002277479003304413695886934603643196990995766533704899398660364),
# new_x_y(389.2048005173494675761280534666365041643437126227625642242847677273894336172543215518016845349229454, 985.1984847640827070380250607267339815502087689086362379221229690053339597241746119322562824448772001),
# new_x_y(391.0952731975904250692073213333059128766761023843296038402588797521722123793221105016378950317481322, 985.9986225128024609436591461152127105001897899169420344746572445503035997492496472111420749498883637),
# new_x_y(392.985745877831382562286589199975321589008492145896643456232991776954991141389899451474105528573319, 986.7987602615222148492932315036914394501708109252478310271915200952732397743246824900278674548995273),
# new_x_y(394.8762185580723400553658570666447303013408819074636830722071038017377699034576884013103160253985057, 987.5988980102419687549273168921701684001518319335536275797257956402428797993997177689136599599106909),
# new_x_y(396.7666912383132975484451249333141390136732716690307226881812158265205486655254773511465265222236925, 988.3990357589617226605614022806488973501328529418594241322600711852125198244747530477994524649218546),
# new_x_y(398.6571639185542550415243927999835477260056614305977623041553278513033274275932663009827370190488793, 989.1991735076814765661954876691276263001138739501652206847943467301821598495497883266852449699330182),
# new_x_y(400.5476365987952125346036606666529564383380511921648019201294398760861061896610552508189475158740661, 989.9993112564012304718295730576063552500948949584710172373286222751517998746248236055710374749441818),
# new_x_y(402.4381092790361700276829285333223651506704409537318415361035519008688849517288442006551580126992529, 990.7994490051209843774636584460850842000759159667768137898628978201214398996998588844568299799553455),
# # new_x_y(404.3285819592771275207621963999917738630028307152988811520776639256516637137966331504913685095244397, 991.5995867538407382830977438345638131500569369750826103423971733650910799247748941633426224849665091),
# # new_x_y(406.2190546395180850138414642666611825753352204768659207680517759504344424758644221003275790063496264, 992.3997245025604921887318292230425421000379579833884068949314489100607199498499294422284149899776727),
# # new_x_y(408.1095273197590425069207321333305912876676102384329603840258879752172212379322110501637895031748132, 993.1998622512802460943659146115212710500189789916942034474657244550303599749249647211142074949888364),
# # new_x_y(410.0, 994.0),




# new_x_y(380.9177547806489190546317657967166540173370930135329221439555232629511235405820067296749041327095874, 980.1602635423587735964955768522714413894422131478130324712315370645739933157834800789113577380449988),
# new_x_y(382.7660174785809980532309512891085346835592188336987446530848688203113511421055450186575988985375467, 981.0535901181957281196623569329824300717642076824844931770973950885592755384556985384264511293771783),
# new_x_y(384.614280176513077051830136781500415349781344653864567162214214377671578743629083307640293664365506, 981.9469166940326826428291370136934187540862022171559538829632531125445577611279169979415445207093579),
# new_x_y(386.4625428744451560504293222738922960160034704740303896713435599350318063451526215966229884301934653, 982.8402432698696371659959170944044074364081967518274145888291111365298399838001354574566379120415374),
# new_x_y(388.3108055723772350490285077662841766822255962941962121804729054923920339466761598856056831960214246, 983.7335698457065916891626971751153961187301912864988752946949691605151222064723539169717313033737169),
# new_x_y(390.1590682703093140476276932586760573484477221143620346896022510497522615481996981745883779618493839, 984.6268964215435462123294772558263848010521858211703360005608271845004044291445723764868246947058964),
# new_x_y(392.0073309682413930462268787510679380146698479345278571987315966071124891497232364635710727276773431, 985.5202229973805007354962573365373734833741803558417967064266852084856866518167908360019180860380759),
# new_x_y(393.8555936661734720448260642434598186808919737546936797078609421644727167512467747525537674935053024, 986.4135495732174552586630374172483621656961748905132574122925432324709688744890092955170114773702554),
# new_x_y(395.7038563641055510434252497358516993471140995748595022169902877218329443527703130415364622593332617, 987.306876149054409781829817497959350848018169425184718118158401256456251097161227755032104868702435),
# new_x_y(397.552119062037630042024435228243580013336225395025324726119633279193171954293851330519157025161221, 988.2002027248913643049965975786703395303401639598561788240242592804415333198334462145471982600346145),
# new_x_y(399.4003817599697090406236207206354606795583512151911472352489788365533995558173896195018517909891803, 989.093529300728318828163377659381328212662158494527639529890117304426815542505664674062291651366794),
# new_x_y(401.2486444579017880392228062130273413457804770353569697443783243939136271573409279084845465568171396, 989.9868558765652733513301577400923168949841530291991002357559753284120977651778831335773850426989735),
# new_x_y(403.0969071558338670378219917054192220120026028555227922535076699512738547588644661974672413226450989, 990.880182452402227874496937820803305577306147563870560941621833352397379987850101593092478434031153),
# new_x_y(404.9451698537659460364211771978111026782247286756886147626370155086340823603880044864499360884730582, 991.7735090282391823976637179015142942596281420985420216474876913763826622105223200526075718253633325),
# new_x_y(406.7934325516980250350203626902029833444468544958544372717663610659943099619115427754326308543010175, 992.6668356040761369208304979822252829419501366332134823533535494003679444331945385121226652166955121),
# new_x_y(408.6416952496301040336195481825948640106689803160202597808957066233545375634350810644153256201289768, 993.5601621799130914439972780629362716242721311678849430592194074243532266558667569716377586080276916),
# new_x_y(410.4899579475621830322187336749867446768911061361860822900250521807147651649586193533980203859569361, 994.4534887557500459671640581436472603065941257025564037650852654483385088785389754311528519993598711),
# new_x_y(412.3382206454942620308179191673786253431132319563519047991543977380749927664821576423807151517848954, 995.3468153315870004903308382243582489889161202372278644709511234723237911012111938906679453906920506),
# new_x_y(414.1864833434263410294171046597705060093353577765177273082837432954352203680056959313634099176128547, 996.2401419074239550134976183050692376712381147718993251768169814963090733238834123501830387820242301),
# new_x_y(416.034746041358420028016290152162386675557483596683549817413088852795447969529234220346104683440814, 997.1334684832609095366643983857802263535601093065707858826828395202943555465556308096981321733564096),
# new_x_y(417.8830087392904990266154756445542673417796094168493723265424344101556755710527725093287994492687733, 998.0267950590978640598311784664912150358821038412422465885486975442796377692278492692132255646885892),
# new_x_y(419.7312714372225780252146611369461480080017352370151948356717799675159031725763107983114942150967326, 998.9201216349348185829979585472022037182040983759137072944145555682649199919000677287283189560207687),
# new_x_y(421.5795341351546570238138466293380286742238610571810173448011255248761307740998490872941889809246919, 999.8134482107717731061647386279131924005260929105851680002804135922502022145722861882434123473529482),
# new_x_y(423.4277968330867360224130321217299093404459868773468398539304710822363583756233873762768837467526512, 1000.706774786608727629331518708624181082848087445256628706146271616235484437244504647758505738685128),
# new_x_y(425.2760595310188150210122176141217900066681126975126623630598166395965859771469256652595785125806105, 1001.600101362445682152498298789335169765170081979928089412012129640220766659916723107273599130017307),
# new_x_y(427.1243222289508940196114031065136706728902385176784848721891621969568135786704639542422732784085698, 1002.493427938282636675665078870046158447492076514599550117877987664206048882588941566788692521349487),
# new_x_y(428.9725849268829730182105885989055513391123643378443073813185077543170411801940022432249680442365291, 1003.386754514119591198831858950757147129814071049271010823743845688191331105261160026303785912681666),
# new_x_y(430.8208476248150520168097740912974320053344901580101298904478533116772687817175405322076628100644884, 1004.280081089956545721998639031468135812136065583942471529609703712176613327933378485818879304013846),
# new_x_y(432.6691103227471310154089595836893126715566159781759523995771988690374963832410788211903575758924477, 1005.173407665793500245165419112179124494458060118613932235475561736161895550605596945333972695346025),
# new_x_y(434.517373020679210014008145076081193337778741798341774908706544426397723984764617110173052341720407, 1006.066734241630454768332199192890113176780054653285392941341419760147177773277815404849066086678205),
# new_x_y(436.3656357186112890126073305684730740040008676185075974178358899837579515862881553991557471075483663, 1006.960060817467409291498979273601101859102049187956853647207277784132459995950033864364159478010384),
# new_x_y(438.2138984165433680112065160608649546702229934386734199269652355411181791878116936881384418733763256, 1007.853387393304363814665759354312090541424043722628314353073135808117742218622252323879252869342564),
# new_x_y(440.0621611144754470098057015532568353364451192588392424360945810984784067893352319771211366392042849, 1008.746713969141318337832539435023079223746038257299775058938993832103024441294470783394346260674743),
# new_x_y(441.9104238124075260084048870456487160026672450790050649452239266558386343908587702661038314050322442, 1009.640040544978272860999319515734067906068032791971235764804851856088306663966689242909439652006923),
# new_x_y(443.7586865103396050070040725380405966688893708991708874543532722131988619923823085550865261708602035, 1010.533367120815227384166099596445056588390027326642696470670709880073588886638907702424533043339102),
# new_x_y(445.6069492082716840056032580304324773351114967193367099634826177705590895939058468440692209366881628, 1011.426693696652181907332879677156045270712021861314157176536567904058871109311126161939626434671282),
# new_x_y(447.4552119062037630042024435228243580013336225395025324726119633279193171954293851330519157025161221, 1012.320020272489136430499659757867033953034016395985617882402425928044153331983344621454719826003461),
# new_x_y(449.3034746041358420028016290152162386675557483596683549817413088852795447969529234220346104683440814, 1013.213346848326090953666439838578022635356010930657078588268283952029435554655563080969813217335641),
# new_x_y(451.1517373020679210014008145076081193337778741798341774908706544426397723984764617110173052341720407, 1014.10667342416304547683321991928901131767800546532853929413414197601471777732778154048490660866782),
# new_x_y(453.0, 1015.0),




new_x_y(352.0, 954.0),
new_x_y(353.114665666888176284295417618357787642613260411632072306995661012268466211249819303267743454006956, 955.7238410085400807585174528351996599603861233028120966172013265786127577632418166511035101615478253),
new_x_y(354.2335266315478304302719452775339449375794912469502933135987509437137152664356979737192678106739245, 957.4449618215747240369312147868460137547745094910873793484252002312798363229040463967358489827984211),
new_x_y(355.3607577247964924282875014870096244632059863229617051687495340126266518624278128028081216814099464, 959.1606108214665129571975627071882571526571789905647808619000100301116125361526074833672797341556881),
new_x_y(356.5004924515158171223196410247053309801188819831233554605218275644614403444952430552630236334493637, 960.867973803294973686569003583440971215091788852326934645066170856360862686713442641243604691366446),
new_x_y(357.6568013521365427926290304682159036667747477450869308047252868150707999287741061968112127240539857, 962.5641433305108540507014991258336981720504234454209112865551208960484499621632893761681364813819761),
new_x_y(358.8336692062844012339482578999639250037211524022128757317119029905765689804273153610467937233441191, 964.2460888888048495148448594023584969610314592179099130398749382256279011667540979964338383533640581),
new_x_y(360.034970714460782491263676802220121388545874926609859978296736805780056407129706271331939651258838, 965.9106281357164465620139099766961225684234234981789136980922712492047068264815783674933343232954797),
new_x_y(361.2644443132412664275801320612901752046639258554333407108058881163828999167669975923170898765023368, 967.5543995698329383421736824707419923158587880909908737770407395002922897128202576163472854240010941),
new_x_y(362.5256638050988378279744120579669018986206838088970847128371236596668947753571668159525880186467119, 969.1738369755094882202402748468520900154452501377895619320894185954651532463780792088456696837702348),
new_x_y(363.8220075163046139729211434557313887983910786637809150617749881860661761228182772696934727735639231, 970.7651460362839058184750248535367043330092843150645983743539932067122573876927884751486411121347327),
new_x_y(365.1566247362413818027066416785620698423272249373012452652479594080804896645482992667272928148352699, 972.3242835518079315393498199482378287828230680688545594910431961305683487122114009310912360466000724),
new_x_y(366.5323992397829506225466108106402709830816202677040331171218695178048682807720813323624285732080272, 973.8469397382298158823316354258219376814621283169061141872982663276598011751972633956077624929773278),
new_x_y(367.9519097521016557876717798862681825452363989808374518146862350056111254108777337150360355461212009, 975.3285241393928863332897328607846476112680393762264760842629165499080675434114424824712099562688372),
new_x_y(369.4173872833480768027778361040026897070574738169274116503893310539281646404563318729892833015656673, 976.7641557245816175165348199661885224222496075343994197586724936459189097830289474309946680543212478),
new_x_y(370.930669340066228408467341333035236598041925449103185625086354819636657597054164170254023838632531, 978.1486577962139491258990052178551246764458617199610799513538714636067501819549909353819455107916704),
new_x_y(372.493151111865788365471459211536378337482601781679027256238933424804058235021140627979666059515552, 979.4765583759302631424859501440358999846797132529526093899993317450013157020556631489820289386560143),
new_x_y(374.1057338365526036529248705213962208876197782798021570448408494555364197653798873941160238846328276, 980.7420967777503112957644075856819872480549551858409026394060344135837527861163429549800008365165637),
new_x_y(375.7687706652179934060657250552553103238647722325943119069876800898614566898966964834884264347999055, 981.9392371098302447688909248385610910550780134613945190416697674495982867227477041721330110954512052),
new_x_y(377.4820104810468155395381089630752568867603365480831472221242967349127262603679915400317345474920893, 983.061689469002025457622315843438065820234498594229596109590768034743613743593010835708787613864557),
new_x_y(379.2445402718242996930203414037723627793671143812657639208155864071042450563571212427328186987655377, 984.102939601546980140976765286905129649827126147222923888460102873835347778893687463339478667441681),
new_x_y(381.0547268158716204945058971278984715514299619955156655054634095044295085594067055361729325675209715, 985.0562877960697959892568107607290733638497057726123418441902003844372001719280172714117712580535586),
new_x_y(382.9101586134600087270049943202564676043641564336311331438711463613028554376643172360743825599151716, 985.9148977461505647605778550988441376248721944004145661858821651828130687922749763381811898358721944),
new_x_y(384.8075891790492075789035960610212105439059362632455195263704308054449051593013043111599691551302543, 986.6718560676874344705150813483616902828418541213037093332049577393031202841766812064825769423009201),
new_x_y(386.7428830016333980258767519869555866751201871178859265673611881458039506569485790539713404065386617, 987.3202430743742954504197347526810423839143288641574385978861946098362083383202519967798154828043417),
new_x_y(388.7109656778724012721789280954724478942619381370576784587369470903200579005462041178926873221262236, 987.8532153004030016858388158736826724363788488015511019229966447507304764760661369875605755975806004),
new_x_y(390.7057799214011180208789546372609134456210930324824340824314682558793425059092116544863872768536716, 988.2641001081223574607005197371495392068592575961801252735552604845293044542746683016324266447369753),
new_x_y(392.7202493465612728916268423493262572818501714203319885768567159912940510667007917713955590698060434, 988.5465025261354726770952966550327388406985586900751028687202598104781757077443192466609277349693238),
new_x_y(394.7462521094754295629112637736324235166248426033563028869062423230108172831444941396095864091203809, 988.6944242266965142904915233272264486512543387995536748163647830242498354376443345508721796431363213),
new_x_y(396.7746066563903011094298743876540210713950490362554345822421832318936345290580135247760977694681155, 988.7023942674396839429487869663602823954517034265086870708735349183445696762474789636806161388917666),
new_x_y(398.7950719698546399568347850254689643660292468764709928789865761926330152088944141899418041167586762, 988.5656108894980009351957290872742637335004856612493297649898118315467899865618132822669383595077424),
# new_x_y(400.7963648076780727814328043616038973941679912446596130757784286237530217830115779441420455016671092, 988.2800932811984770436741565897272027415591925343268254290128360287554893193586668698098226649582638),
# new_x_y(402.7661964867347621540006629667241261539832992390529755776851935038812388344205770006527919083008032, 987.8428417845177398923830865015437214508627507791925478819143910092941391198653124362871652441866609),
# new_x_y(404.6913317615396395109884583058497179996471667578046348370530137371339358696935566587576197994638387, 987.2520045429711340586848010752270245610263388365165645550253156718366111503760098113038680899967123),
# new_x_y(406.5576722733712204921177573495904853633521094509170321707852535663913173000025118083549945717830885, 986.5070480694314647342765515560793976460682074804563728678082352702450785780333657300984689339486734),


]

SEC_4_WAYPOINTS = [
    



new_x_y(615.0, 1072.0),
new_x_y(617.7799259492933820062945402733453066927292467325344463623000807470770843284190020510991967910069226, 1072.902741240128706581518651344912453149199002398926892015566916701006875346187465843136233735616386),
new_x_y(620.5618866493955629816284270282880217744058770948470987681624660200787273838723662148982642617568449, 1073.799203964120746161121375131175126506414567783116343005466308274245716426582529593592091391583977),
new_x_y(623.3478945343227769592209857596164591871987437950382516276134143949749844276022094615009187432033195, 1074.683102484348474775278165864591099495987605668200242496273028306950149467017676694090130759349517),
new_x_y(626.1399173220197502444555175308550724268469474134189784990361913020905802352459828493838395642641516, 1075.548137026795870671619799226948604162031737637442727522490878914955834115006332159583671997722193),
new_x_y(628.9398554522386984086159342157136388354383656351122066614742266783328640126469360379316234469258923, 1076.387987330018596647368759300927966756882169139372756319720558547676808902289706399072802486189334),
new_x_y(631.7495192854959454578115343102675039102046804547398683158502274942452498715856018087709545516783877, 1077.196307016521969600706435361419223959761140188518460949670274403531248672466290647462177875308706),
new_x_y(634.5706059934732531704268017762061830679875547636129367848456166330300940939569905618119722466293642, 1077.96671899695177935382601901484205065768926413465999787053748738820986922916512160967579275395141),
new_x_y(637.4046760798900070736028467387126188363858443282996747688639846195738212149819736691637893856903414, 1078.692812169732459393875578995631583907465642997750208924615339684499263201697498661926300804638436),
new_x_y(640.2531294817932207616374551565893383737067282985072478157652610622757059957922174473507289088564289, 1079.368139681253825079756330016905107576402812780834990985835338327179704505480332747691794187772551),
new_x_y(643.1171812144531320904982417570969308090804458140347938859533123861042293381893385304450196265555878, 1079.986219014177990386323682555623710883766802027705135704577934008108823390526570161549402543495167),
new_x_y(645.9978365386763694873326296267835317110653293454583364181960391843623955025502889974559205690665532, 1080.5405341736425896180281345807005284422622630768668346201005246236360977089999136359150358326601),
new_x_y(648.8958656474211903299760153856026901513662932849318950902581936412751228217742059956639873085340334, 1081.024540242758967412896727618885350373745191105050107596188053198969627743739725174742283227079257),
new_x_y(651.8117778891823020567653903703176524813274340657225630196397771655125695768400849227322342383575769, 1081.431670579481662725836110759154330952991701615970060709584186016981325473011027272607971821571593),
new_x_y(654.7457955687606956905004523275157810967579235654853041936896016651057780885695617827545273091805017, 1081.755346926248673904240760954569917201102580124353659103414479757963791472774558472248094071770944),
new_x_y(657.6978273917877569462922911524013830913848740827506693488018558146137033603803966942967551658326662, 1081.988992701304804409278634551429525951775747441821383421563517025878592821296549606674198652527569),
new_x_y(660.6674416477538658830640524163605580939022945406453776889499628231004497620613465257157014282070423, 1082.126049735821914802216943022583356661629097431656156319008951503057777785146431178480098804622126),
new_x_y(663.6538392572940272252139879073430304713025785753565720851643859977381368610883512821861899642702078, 1082.15999871327604983444779099656677271389647526641543070390172354396299053451399278332233295810083),
new_x_y(666.6558268430663423610886032780004489593433542727443035740870369432375857174005574760981737717716623, 1082.084383556447820144949061522154611058748052020652393745495751985538827737215847431374155475231377),
new_x_y(669.6717900196396936555372050736590914954346654497539145976965566977463401002364187565807726827531224, 1081.892839992258596665660704018660906159309243729979476888104179197119536472087058909372263443926503),
new_x_y(672.6996671362489165954118497172932901072087987254203819960830305217283505420991427600384175705304347, 1081.579128504789900933588588882858641142485604048579904251516536101300262825013560085382830417887479),
new_x_y(675.7369237468810999542183373532698012104534939863278718698326318841097537836731044041869635131726483, 1081.137171861582252968874809745877455480016611690688071861641337076484953696674842856138736390179025),
new_x_y(678.780528124655567969443851454268730570703011332594100436564750809551349617190753966796443634704863, 1080.56109736698364457502099610960294467017835206937364015042744606498926976157248649225973545551497),
new_x_y(681.8269281815002411790101728487883041158396234136180224137747660883534221936221011843673158911555916, 1079.845283958224418049031997028088409060777381794084331449199377492072950986241778168545610788794025),
new_x_y(684.8720301992632079650707844377965820527098553955114076546169974538422949318128193664848255390187455, 1078.984414214352513940142777361445513401052257966560829668573361836997904772121830043117988020179362),
new_x_y(687.9111798240819151984224478329453701140869589507943877097276528661748434413475655926266436511801293, 1077.973531294514844464564665581388284286356163546594515510892199437804380537596857704100008765880158),
new_x_y(690.9391458214015095460899417379873006050043860550729763384365503980357410743832095151976629981775359, 1076.808100759705918553716611935234112707384260596314015422627038962113601259221921158963276179270135),
new_x_y(693.9501071337039764327946729717280624844681688305119062593916364900855677744509460270413919893016359, 1075.484077160479269617744275096560661301491135939893751995518401300521282766300497483721866261680908),
new_x_y(696.9376438258653832685152857785336917844784324346206219558578184366339092527236058515734369771807104, 1073.997975191777277050160969563974265649459405986669462384269199569853302856658409683253272239744884),
new_x_y(699.8947325430466946578579927918605619862162118889036697157580737014753098358720272097491118923392275, 1072.346945124645836835434632742480863575574273340814120472973267236195235086278143827187028133509275),
new_x_y(702.8137471419489859688052671229452431915829502766423426857211480735598581095498368892735910329791745, 1070.528852122976403125267122784912567464414081336323900521508408768496676584670242299252104208143047),
new_x_y(705.6864651867867943419307041024604924371057945045031743715375920367242248378797134902686731318486544, 1068.542358941556037960508267918031960937654177372472387539367981989485963172347556705887344519644873),
new_x_y(708.5040810249709795114441988650719179593810210123819318056391238341766699735508873423329981251623054, 1066.387011379821520335744353607791581287915622247583685040122799240493790727541970527319981009721331),
new_x_y(711.2572261726238058968470942930688859799509248893307412182063574078225410327175748874602729914912954, 1064.063325734277877331873066604512153872179460429593637206458144351857781366382417298100794762160826),
new_x_y(713.935997744924433860820487831384986573429495849048203532153402415242118081186857787523257568049463, 1061.57287735232256553480011334615585104354678004751040800494965580366738478310884248706508407822705),
new_x_y(716.5299956590394414024221192056007661790265384183559501961225457024390643953890784491080834345886761, 1058.918389242318113853123689328785959365952839350227828869242694014137405676538346960264726392462732),
new_x_y(719.028369316074627108794396849805286148575140690794020628979217988382568946790098871496632500447938, 1056.103819540659467316024425882482159672904229768605024864101659673215953572839592529548694429730148),
new_x_y(721.4198744310707146581633978096880583801642713431786636310358359309249634557869671191535830718874878, 1053.134446478184935416462419358397429250062119223856520168361368692553124818738753299726622551085934),
new_x_y(723.6929406245069368391353851529319075994095268777373790423262722523458361926304896490791485125892015, 1050.016949327931872544229979164591901384589993234688331994522599801632991096690102488271386447300422),
new_x_y(725.8357503130374196314089362867512404114373195504443541526521955827708082617937120181660395255376512, 1046.759483656775892583460623796763286563696127951560147222996407158005800897747790674781526004739798),
new_x_y(727.8363293392970758357352280048223249955900413692770017596325357967768890784626706113230112019055428, 1043.371749048263526174572399489420461414307869467496654958128791185476203266570911302457397034175551),
new_x_y(729.6826496587358674528823089744079538757255517344333528880645639724531474231632026814084404827705466, 1039.86504731683149337681775109233819063512634761909062103922986181451931791801716827588340609157429),
new_x_y(731.3627442539313788709537718824954963063403448570328796650331656287314153481188810030302387186748098, 1036.252329099018733654377361020787135234210596785587488178473006225582698801509426233220802041850472),
new_x_y(732.8648342723275389649865629268803341503279340444525046844009385211802332783044620129584925369564228, 1032.548226590171796971525721984641775196129870697205913019048720416192643594293891030063742662417124),
new_x_y(734.177468180858519753847707359633889919468078296868494313064799107183566059488489306468816066990479, 1028.769070100986032407424931138329294533665298279134763833334371795004299241579980632618969559597991),
# new_x_y(735.2896724999137992802559236474802656109997676021970272872616631622089219106466736072627974382051956, 1024.93288604295560999913601840282133816350989114922720576469702587460777749474356084984963356514584),
# new_x_y(736.1911134196265057977638675716229027819284839694501328096814749625280881809618899856221716203129075, 1021.059373921782252781309550860295307121767237365788908265549250970403383574639570621586098517807521),




]

SEC_5_WAYPOINTS = [
    

new_x_y(758.0, 890.0),
new_x_y(758.378096280178902392093174091171867888191253447165802987925631901989762446464005962810778453677994, 887.982290760644977742181312528447700456445869190527855932404103386638144042864270386571156919996109),
new_x_y(758.7560942693197554411152196263186205071715826900715343994332643792026339286225472198637644631848866, 885.9645631128162042669591649859388132209376964462289435699536029565675090357915897078623140058173648),
new_x_y(759.133895665337459870609506598043206975595347205569938170191097270519408118415477296930471155083377, 883.9467987070141822798860828400932010377675163390637895450508404417863922696147012053828238715289157),
new_x_y(759.5114021440675612976803463350377479094053334941338608413339840818530317124263000937893673366565518, 881.9289793116906777060488233137784230313205406364020435264161591842423809687485488477860515922354019),
new_x_y(759.8885153482634384442651312784617285025296736478264336866069338445266535369507491922830395137902441, 879.9110868722312190923781103633744184795386738419275398675621174946849952827840832797816431138739393),
new_x_y(760.2651368766377410746258123606223713277775947135346884018200916883273020670144783070883577983864144, 877.8931035699457805630192461678482333677598297236949311694045145447693994127653538435620702625226655),
new_x_y(760.6411682729628455554795099645755750644085709521958362308878945957324621932726194840553064501254272, 875.8750118810702798437879210986044992390985156707628419549698609393411675074801549645721379560733229),
new_x_y(761.0165110152451112975089021063169121097895826678250440374374567853714072581850772775637974726405826, 873.8567946357814402886178133006478733932279114789994896220280174500034233063911939725919130480583999),
new_x_y(761.3910665049877404676150769661722241253873965398829389696555109632470604897635730293556270676374084, 871.8384350772274625988473047800770039638433537327392670897751098409701773912539829607017047312543248),
new_x_y(761.7647360565570662189344720844775656047657733279720943251595233568950512805978373182592690284998844, 869.8199169205768280179276957882482541681958897111060087779481169541248035628063645256684728337307913),
new_x_y(762.1374208866671212149627381650797504251998401777311778846310637446373202539654636668550373955204895, 867.8012244120874102000531078349278555588762655016650626124030678342963114818187144334737023401257007),
new_x_y(762.5090221039973683460994735734965274517719990540014422054522530615315700966473369793982214821534612, 865.7823423881979076778843636298309324703186871947972076813669007882429641446534262475114988389015303),
new_x_y(762.879440698958509198460511128025326120546688237263206520841015570529161210606648294925962354704857, 863.7632563346434228837395577040930156238084598905286578269421147024113777182349065533222982659258519),
new_x_y(763.2485775336213229300599477422345324552392387988520865697187420318964931249649941383332711139725414, 861.7439524455968069933476458618950055583252750640141041369284263599180100981616294024653816030463541),
new_x_y(763.6163333318235286752578210711954828280997251134661326635080558351126812168472642654974005806850902, 859.7244176828371624528553911594280909969425945295344022642504688052534186227548661538375518498212531),
new_x_y(763.9826086694697082778260051538308730602294615112516403860165509333288982388762243137907228292228929, 857.7046398349466468944530860813089673735960317474120612312392457795887978601304065014400580236327645),
new_x_y(764.3473039650393730045066239197941000972437975814205841112491450487048907805857232738074971313705775, 855.6846075765364532410520563268960632169811648897374932171441053150273936554723239888842241965800445),
new_x_y(764.7103194703183077394223818088294865134161069946231150980969729153258457360087798009946785822288047, 853.6643105275025511216050207580576527399216073460196700533374085215621483747217174800805823886127834),
new_x_y(765.071555261368378874184705578219777959644879823444994482841084321728457484916867047541307223488749, 851.6437393123114642515406415815375165311455565687903909011625085521222921980350729428334089770651369),
new_x_y(765.4309112297510475955996069722430600224883538052025062606691292589273466703861768751729249645242673, 849.6228856193160271705547868111235138024084119953674931619963965507554747960665050502863235551317743),
new_x_y(765.7882870740198883231369889925965572825912899420421909602163067064700395920293680871838684412520933, 847.6017422601007126494200510552697216422102474620900649466075345220770532523842596229165848582802381),
new_x_y(766.1435822914974725334246082838508404917437586904103282313132545329028733425537721361159420990781928, 845.5803032288557481712560224636789624466092408839655019611167378453450121998046370875362127967984902),
new_x_y(766.49669617035204097031322967875778587331588219364485455362960518399975356164333855180204516477983, 843.5585637617788461509700401572012873281377417640307942193999445101224326393683217555068339846600222),
new_x_y(766.8475277819894520254637420751630830248254007359515077089392516678970567836970157470347065640492588, 841.5365203965029579598055501965304305786993605053700507479186609879624002079852849705591415649590623),
new_x_y(767.195975973775960812223083366815086802043124385491831555218554783705860517086032513075217112725682, 839.5141710315480263796159447263270800961296559478702400279480675365199741675556848225113688676935575),
new_x_y(767.5419393621074518515712628449836195637905847333749692309630210899449128349230949234122910940075521, 837.4915149857942548048962607895345904863538718102514394436359211907772657136611435314762113550353418),
new_x_y(767.8853163258408181081943440091332005742439169861288259750855974226393468996314916716594850219368058, 835.468553057973934336240188026806509604062398918553733702232807267439611086530338498703909469661681),
new_x_y(768.2260050001032503020349280926926659785093650079013225516989558555399769424976917917121740942303619, 833.4452875861783718891763745760758615994266565006352325271520898099035641227097418380221942549713677),
new_x_y(768.5639032704952725350127741007910855471560595919192108572932445697063984324927649605524921983515137, 831.4217225073759435554888172435257542227405150079942206733683525872361260927151231375636683341496595),
new_x_y(768.8989087677034331655999606545270980176798201735982199737441674177432645839810883933553598797658151, 829.3978634169367577239424164773272816195446009047858161107451826448608546028430328520051470892646309),
new_x_y(769.2309188625386334019470350901969111166672626561509377364767391452877372960973557355118637486193917, 827.3737176281588519201303886160582721196075257653833228899427719248938200341344418167743393162326634),
new_x_y(769.5598306614161497463252964869887342158585932167922764115887124429996961593051097723134536467360734, 825.349294231790265952833002424018940390123471839989296548896781094345879134493626437850832460162619),
new_x_y(769.8855410022934801481691151491180005413486236156950653445319762936203000922517705428013457209854701, 823.3246041555407318079926668441840644349640301745812606410886053617655965551922411447564707241729928),
new_x_y(770.2079464510822172313571396877475723372546985639809470373060655448140102873660778663933073384281734, 821.2996602235760978411241433784044067785819482870117389577560058128557519339703909396015230195534374),
new_x_y(770.526943298550224717440208246246472429009016424580828605982737795171768183674925663156727532563941, 819.2744772159879612045320142281240498649834132898156695356620297076475054960615937801617737836965189),
new_x_y(770.8424275577304652481304386299496619386191814100110518661149357751125754750629123804621862816609049, 817.249071928230318188418910573103314440227167294876690019700896621330347343806372512429147167710702),
new_x_y(771.1542949618528985144847558108252801189663631372958022203246970471625363217248725984984305615112742, 815.2234632305143572782575424129816253112704362377725130810606973313697267499208255201025103925668128),
new_x_y(771.4624409628159379072825945978666496989079718841649507279639186457006720197002866616006998247689142, 813.1976721271518143253933722652077644397163691806671546244110066963226154169990254172329157146438879),
new_x_y(771.7667607302140212268652824060173191115543691270056400285213283961367549331531104476407717453844582, 811.1717218158365833517860636701275792002485031623378138373861441129878696912414277868208463037720374),
new_x_y(772.0671491509379163262852989836141485742837075553553100303822515512378264116018893581095007853725613, 809.1456377468535302676885346974656030626759269298039438454394905359717337228347381341545004691580378),
new_x_y(772.3635008293644450563658223874713233160211034619676647698953066333460750639074111368456170382707877, 807.119447682202690244187419349182484235618819381228914305427475773980665992171229599174163041736242),


    




new_x_y(772.0, 802.0),
new_x_y(772.2365023887155780570933348782552144741007370723834609242111438121016873891669925464882445453869424, 799.9608398807853339210573689667286631123151067364006112864653839275607544286057109959291365453693145),
new_x_y(772.4698754940178678124640668359283225450735975352530288902606673654992985588726983703459126534765085, 797.9213190914444526935286505951791405681313624460519135968033166318389073488859862511795328035308135),
new_x_y(772.6969885053502257785841253787576579786063821321851727065075662471473873763802811299460258351922454, 795.8810903751933207434834561914460232152438240979396805788093003955270971108599046232714746386112451),
new_x_y(772.9147076635167799058426916939249001784041774348960547811200646069071084960812224501732629347012679, 793.8398333138030563041846432984924247816156179400850100519591575900009685062026659283625913699842074),
new_x_y(773.1198950505973448092932487423115051793474727441222683092850189423753545537973800551774697897772114, 791.797267775497698903331208915705394096636896118729092862133482555716037827650933246857861333233419),
new_x_y(773.3094076972530991026311772933352975933096939015920020659043539195743376443541576314824255865397855, 789.753167394235565238349053477710493867139737104929044177704587856584141536673247774378074875705334),
new_x_y(773.4800971136905158532224544955461077851198415026865686981358840361119362052410336943580295876573621, 787.7073730858963905732532174709135342161783837881748774345016181204426952500219773972856272514603507),
new_x_y(773.6288093508643633205441572213276093655603878613454757985959926296123659624442135633751770614894323, 785.6598066026545689579522234855684095180724946345235065601254596355444563961083303216198195088111414),
new_x_y(773.7523856987795803587296884955095681460614984078927656008860831806351680485107399931481125681300227, 783.6104841215075489137761403531026450325659295973515770266773088421545807349414952977414058762181335),
new_x_y(773.847664128921178006161211263940198979972429485198090565172302908067458220450731982112165737950871, 781.5595298565444756226228962814366593330374442491165225876743386342612904530968922472515943055208318),
new_x_y(773.9114815878105457312276810936149089870611494736913290183314712307567963001316840079229857219609938, 779.5071896770820961228871538206567098365308850880071314925894308986602628823556285288330559076890625),
new_x_y(773.9406772483500109370097871624792819269417319769592845123120866372742476812752369391108415435110033, 777.453844705264721267932135992319284093294874020650495929434565277299535191011180334838922458347792),
new_x_y(773.9320968248545244156261978573828503715877929313596216084838243883907200352731543961306557706629334, 775.4000248571296519183586989825321277848303457107961257657632328950019451673886862096474580533619963),
new_x_y(773.8825980563443881931414932062704382250661906857012427926034488293785447957145043981968004429054459, 773.3464222804928345490627211811448499687289599780098071823570682753340785029870762270791501021011636),
new_x_y(773.7890574606359935213948839149894753458690530647090096011084190442112522960370827476450382798101302, 771.2939046313345743000189383600533531687433954670506359839525983451165217427398485439685672940272589),
new_x_y(773.6483784588546577354805656095146254432781998747300921494275014638713536112281891957343056157645194, 769.2435281176962696482942893055175374658230620740867731832957015713191428686773716626336180523257657),
new_x_y(773.4575009660277753473517756822496298149384853647927452589925395061741475530351673284203804611342266, 767.1965502264846821171693465631834951501949003246675891030970697865159400399442164729872680142039069),
new_x_y(773.2134125382085282179498594859973220354318970089317559865377319957644585957629108687060021369399307, 765.1544420340852860754741154855773477406280233259410468783824374195029077037558271590148571186753566),
new_x_y(772.9131611599306289067382708537221384326308369978116258967855591146290896376315163807002796454948132, 763.1188999863954842593091796857977657473516068287874279549829268930163217137709316496616943274601999),
new_x_y(772.5538697474945437328870362965278440685265867770109550470319506873845292590915504289054649834486934, 761.0918570179093734762025167730811748520045311831454989645748240400321309245884965066360939775402577),
new_x_y(772.132752433420447393355497835993141153480708312062883933145196584266164812861217490209387900402171, 759.0754928629516393345647446519414363363420177906040523922481434893093986883164208062305997355525719),
new_x_y(771.6471326851542823597949135670914534767633112639191207878887113898285588105410707109846108287414789, 757.0722433952314865561101351705609687269031323154282588892700368985598752368406458795539927045765696),
new_x_y(771.0944632965620244158239153753783703642140524106826192632634598701573730718327982249943857641852405, 755.0848088147630120138306121828553242189313559266873398266038430167835675797996806650659933446360304),
new_x_y(770.472348273678760665007607559821259732633137640784338056542257335219965078893251950231327612411949, 753.1161604841062571972212660717226819057216300542429484139468676640812409112405455025973489278427171),
new_x_y(769.7785666163872731676277227251330208833129809132309682743928153621175197865269130422545076337559922, 751.1695461990918535558038112816725555176001853405038184697398990942951668333452085197085059662364195),
new_x_y(769.0110979749934652496163014618572958293228348435462515893746146424969961565166791168424356495858616, 749.2484936630112428522465561922547185013324495805708087459006750081801962427135122604869395989949884),
new_x_y(768.1681501348715736724721019703244204294484498219249711192448959421285446550142525500324182558748192, 747.3568119180367749420959960969081266529624636670950130773044535637762623549820858940209646218836447),
new_x_y(767.2481882533266572041630237571875210797803297409898567743265243945982710388582129549755071181119618, 745.4985904737794825366185732617235172265548915022123555222277760033382856825033252200895033551943701),
new_x_y(766.2499657404567925067681174254905358940842817638296439016100992117720565026310486536503223942326828, 743.6781958608410935787666903053106096167110769318379104131530843687183374599676511875334799718005679),
new_x_y(765.1725566400284207860684975618467470705731971871709019094386899826071184637959851519719868226829027, 741.9002653274613699848736376022403720314898837224946036022046513465501928410449385677628566216816137),
new_x_y(764.0153893271947879642515126586082280436203659670192073163355055640179321371795825872513784660803116, 740.169697390438327708926677836799457040960464632814388124851422008623220091262333607866081590742555),
new_x_y(762.7782812973428088088428136729091404243156167063710012187828276964852370263139228183358263594246905, 738.4916389479871773733965041238100612550182783997957140892855710922452975919696674126016042282093215),
new_x_y(761.4614747745761864726130635506970705415454200312296473148330771559689347415101018197905041708682071, 736.8714686627241755297297621149395654856821273609209771827421870298739702227360467100426845933074848),
new_x_y(760.0656728195466785998194848440050925857776179083330676785945996377940947415530980881942280480808189, 735.3147763281696254755730149657536204436337912500482319071047187690541066311682200855070924047046171),
new_x_y(758.5920755648433196656468228256105901671497881175616249522871506370863104316416114250696088154760757, 733.8273379427442650344980463981183899598051487222322691453582976689946938230334705490887031527296603),
new_x_y(757.0424161523631751597964226381137009112678142873065137649174982768379441884322183732183369781518518, 732.4150862318892448748781075798810312690792914297491350222804176325524603939009381749954388385881409),
new_x_y(755.4189958915600628328225306661892001356427787319003214401383062679627948797155893638705679133495159, 731.0840763823845891243560443791532081546935705037047161789144062040506318872987971771153228479463601),
new_x_y(753.7247181008753055301348375457302591511582261639343989916583869301813371991287771119885461573095988, 729.8404467838824085995734798239857844184293760340802846182237192678245427361125120907432220680839952),
new_x_y(751.9631200378152910504149227349555095318708571213547969401962078055447490615167506892589396447527381, 728.6903746117963460876833868161183703142122302031373956549888722110729191602995071381477191774806704),
new_x_y(750.1384022670243212238701544199480181683993205742517618410409183496535187392223710948600499108639282, 727.6400261336452629513732076496810309957945728530463883296609216128517975728905683890112496264603907),
new_x_y(748.2554547614365417799231391586976950118681541959867748270279119299157267146473250742630873980916803, 726.6955016783232873363377217480089437136112647224067390888255144141773622375023329335935802313987958),






new_x_y(750.0, 730.0),
new_x_y(748.4424022168341327734714782624759374031597542855909048757614801115087508260571457687845858932803649, 728.6628401301487837349426653394510760195962285190274974089743073188726779519508776710901891039686535),
new_x_y(746.8848044336682655469429565249518748063195085711818097515229602230175016521142915375691717865607298, 727.325680260297567469885330678902152039192457038054994817948614637745355903901755342180378207937307),
new_x_y(745.3272066505023983204144347874278122094792628567727146272844403345262524781714373063537576798410948, 725.9885203904463512048279960183532280587886855570824922269229219566180338558526330132705673119059605),
new_x_y(743.7696088673365310938859130499037496126390171423636195030459204460350033042285830751383435731214597, 724.651360520595134939770661357804304078384914076109989635897229275490711807803510684360756415874614),
new_x_y(742.2120110841706638673573913123796870157987714279545243788074005575437541302857288439229294664018246, 723.3142006507439186747133266972553800979811425951374870448715365943633897597543883554509455198432674),
new_x_y(740.6544133010047966408288695748556244189585257135454292545688806690525049563428746127075153596821895, 721.9770407808927024096559920367064561175773711141649844538458439132360677117052660265411346238119209),
new_x_y(739.0968155178389294143003478373315618221182799991363341303303607805612557824000203814921012529625544, 720.6398809110414861445986573761575321371735996331924818628201512321087456636561436976313237277805744),
new_x_y(737.5392177346730621877718260998074992252780342847272390060918408920700066084571661502766871462429194, 719.3027210411902698795413227156086081567698281522199792717944585509814236156070213687215128317492279),
new_x_y(735.9816199515071949612433043622834366284377885703181438818533210035787574345143119190612730395232843, 717.9655611713390536144839880550596841763660566712474766807687658698541015675578990398117019357178814),
new_x_y(734.4240221683413277347147826247593740315975428559090487576148011150875082605714576878458589328036492, 716.6284013014878373494266533945107601959622851902749740897430731887267795195087767109018910396865349),
new_x_y(732.8664243851754605081862608872353114347572971414999536333762812265962590866286034566304448260840141, 715.2912414316366210843693187339618362155585137093024714987173805075994574714596543819920801436551884),
new_x_y(731.308826602009593281657739149711248837917051427090858509137761338105009912685749225415030719364379, 713.9540815617854048193119840734129122351547422283299689076916878264721354234105320530822692476238419),
new_x_y(729.751228818843726055129217412187186241076805712681763384899241449613760738742894994199616612644744, 712.6169216919341885542546494128639882547509707473574663166659951453448133753614097241724583515924953),
new_x_y(728.1936310356778588286006956746631236442365599982726682606607215611225115648000407629842025059251089, 711.2797618220829722891973147523150642743471992663849637256403024642174913273122873952626474555611488),
new_x_y(726.6360332525119916020721739371390610473963142838635731364222016726312623908571865317687883992054738, 709.9426019522317560241399800917661402939434277854124611346146097830901692792631650663528365595298023),
new_x_y(725.0784354693461243755436521996149984505560685694544780121836817841400132169143323005533742924858387, 708.6054420823805397590826454312172163135396563044399585435889171019628472312140427374430256634984558),
new_x_y(723.5208376861802571490151304620909358537158228550453828879451618956487640429714780693379601857662036, 707.2682822125293234940253107706682923331358848234674559525632244208355251831649204085332147674671093),
new_x_y(721.9632399030143899224866087245668732568755771406362877637066420071575148690286238381225460790465686, 705.9311223426781072289679761101193683527321133424949533615375317397082031351157980796234038714357628),


]

SEC_6_WAYPOINTS = [
    

new_x_y(87.0, 199.0),
new_x_y(85.42915097420127230652532124868456923935990699251458722347438377726692700489612710681222105832574598, 197.6784324800408668882775329738292436991171448915691566375910974805833118743929871842649297995936327),
new_x_y(83.85830194840254461305064249736913847871981398502917444694876755453385400979225421362444211665149196, 196.3568649600817337765550659476584873982342897831383132751821949611666237487859743685298595991872654),
new_x_y(82.28745292260381691957596374605370771807972097754376167042315133180078101468838132043666317497723795, 195.0352974401226006648325989214877310973514346747074699127732924417499356231789615527947893987808981),
new_x_y(80.71660389680508922610128499473827695743962797005834889389753510906770801958450842724888423330298393, 193.7137299201634675531101318953169747964685795662766265503643899223332474975719487370597191983745308),
new_x_y(79.14575487100636153262660624342284619679953496257293611737191888633463502448063553406110529162872991, 192.3921624002043344413876648691462184955857244578457831879554874029165593719649359213246489979681635),
new_x_y(77.57490584520763383915192749210741543615944195508752334084630266360156202937676264087332634995447589, 191.0705948802452013296651978429754621947028693494149398255465848834998712463579231055895787975617962),
new_x_y(76.00405681940890614567724874079198467551934894760211056432068644086848903427288974768554740828022187, 189.7490273602860682179427308168047058938200142409840964631376823640831831207509102898545085971554289),
new_x_y(74.43320779361017845220256998947655391487925594011669778779507021813541603916901685449776846660596786, 188.4274598403269351062202637906339495929371591325532531007287798446664949951438974741194383967490616),
new_x_y(72.86235876781145075872789123816112315423916293263128501126945399540234304406514396130998952493171384, 187.1058923203678019944977967644631932920543040241224097383198773252498068695368846583843681963426943),
new_x_y(71.29150974201272306525321248684569239359906992514587223474383777266927004896127106812221058325745982, 185.784324800408668882775329738292436991171448915691566375910974805833118743929871842649297995936327),
new_x_y(69.7206607162139953717785337355302616329589769176604594582182215499361970538573981749344316415832058, 184.4627572804495357710528627121216806902885938072607230135020722864164306183228590269142277955299597),
new_x_y(68.14981169041526767830385498421483087231888391017504668169260532720312405875352528174665269990895179, 183.1411897604904026593303956859509243894057386988298796510931697669997424927158462111791575951235924),
new_x_y(66.57896266461653998482917623289940011167879090268963390516698910447005106364965238855887375823469777, 181.8196222405312695476079286597801680885228835903990362886842672475830543671088333954440873947172251),
new_x_y(65.00811363881781229135449748158396935103869789520422112864137288173697806854577949537109481656044375, 180.4980547205721364358854616336094117876400284819681929262753647281663662415018205797090171943108578),
new_x_y(63.43726461301908459787981873026853859039860488771880835211575665900390507344190660218331587488618973, 179.1764872006130033241629946074386554867571733735373495638664622087496781158948077639739469939044905),
new_x_y(61.86641558722035690440513997895310782975851188023339557559014043627083207833803370899553693321193571, 177.8549196806538702124405275812678991858743182651065062014575596893329899902877949482388767934981232),
new_x_y(60.2955665614216292109304612276376770691184188727479827990645242135377590832341608158077579915376817, 176.5333521606947371007180605550971428849914631566756628390486571699163018646807821325038065930917559),
new_x_y(58.72471753562290151745578247632224630847832586526257002253890799080468608813028792261997904986342768, 175.2117846407356039889955935289263865841086080482448194766397546504996137390737693167687363926853886),
new_x_y(57.15386850982417382398110372500681554783823285777715724601329176807161309302641502943220010818917366, 173.8902171207764708772731265027556302832257529398139761142308521310829256134667565010336661922790213),
new_x_y(55.58301948402544613050642497369138478719813985029174446948767554533854009792254213624442116651491964, 172.5686496008173377655506594765848739823428978313831327518219496116662374878597436852985959918726541),
new_x_y(54.01217045822671843703174622237595402655804684280633169296205932260546710281866924305664222484066562, 171.2470820808582046538281924504141176814600427229522893894130470922495493622527308695635257914662868),
new_x_y(52.44132143242799074355706747106052326591795383532091891643644309987239410771479634986886328316641161, 169.9255145608990715421057254242433613805771876145214460270041445728328612366457180538284555910599195),
new_x_y(50.87047240662926305008238871974509250527786082783550613991082687713932111261092345668108434149215759, 168.6039470409399384303832583980726050796943325060906026645952420534161731110387052380933853906535522),
new_x_y(49.29962338083053535660770996842966174463776782035009336338521065440624811750705056349330539981790357, 167.2823795209808053186607913719018487788114773976597593021863395339994849854316924223583151902471849),
new_x_y(47.72877435503180766313303121711423098399767481286468058685959443167317512240317767030552645814364955, 165.9608120010216722069383243457310924779286222892289159397774370145827968598246796066232449898408176),
new_x_y(46.15792532923307996965835246579880022335758180537926781033397820894010212729930477711774751646939554, 164.6392444810625390952158573195603361770457671807980725773685344951661087342176667908881747894344503),
new_x_y(44.58707630343435227618367371448336946271748879789385503380836198620702913219543188392996857479514152, 163.317676961103405983493390293389579876162912072367229214959631975749420608610653975153104589028083),
new_x_y(43.0162272776356245827089949631679387020773957904084422572827457634739561370915589907421896331208875, 161.9961094411442728717709232672188235752800569639363858525507294563327324830036411594180343886217157),
new_x_y(41.44537825183689688923431621185250794143730278292302948075712954074088314198768609755441069144663348, 160.6745419211851397600484562410480672743972018555055424901418269369160443573966283436829641882153484),
new_x_y(39.87452922603816919575963746053707718079720977543761670423151331800781014688381320436663174977237946, 159.3529744012260066483259892148773109735143467470746991277329244174993562317896155279478939878089811),
new_x_y(38.30368020023944150228495870922164642015711676795220392770589709527473715177994031117885280809812545, 158.0314068812668735366035221887065546726314916386438557653240218980826681061826027122128237874026138),
new_x_y(36.73283117444071380881027995790621565951702376046679115118028087254166415667606741799107386642387143, 156.7098393613077404248810551625357983717486365302130124029151193786659799805755898964777535869962465),
new_x_y(35.16198214864198611533560120659078489887693075298137837465466464980859116157219452480329492474961741, 155.3882718413486073131585881363650420708657814217821690405062168592492918549685770807426833865898792),
new_x_y(33.59113312284325842186092245527535413823683774549596559812904842707551816646832163161551598307536339, 154.0667043213894742014361211101942857699829263133513256780973143398326037293615642650076131861835119),
new_x_y(32.02028409704453072838624370395992337759674473801055282160343220434244517136444873842773704140110937, 152.7451368014303410897136540840235294691000712049204823156884118204159156037545514492725429857771446),
new_x_y(30.44943507124580303491156495264449261695665173052514004507781598160937217626057584523995809972685536, 151.4235692814712079779911870578527731682172160964896389532795093009992274781475386335374727853707773),
new_x_y(28.87858604544707534143688620132906185631655872303972726855219975887629918115670295205217915805260134, 150.10200176151207486626872003168201686733436098805879559087060678158253935254052581780240258496441),
new_x_y(27.30773701964834764796220745001363109567646571555431449202658353614322618605283005886440021637834732, 148.7804342415529417545462530055112605664515058796279522284617042621658512269335130020673323845580427),
new_x_y(25.7368879938496199544875286986982003350363727080689017155009673134101531909489571656766212747040933, 147.4588667215938086428237859793405042655686507711971088660528017427491631013265001863322621841516754),
new_x_y(24.16603896805089226101284994738276957439627970058348893897535109067708019584508427248884233302983929, 146.1372992016346755311013189531697479646857956627662655036438992233324749757194873705971919837453081),





new_x_y(23.0, 145.0),
new_x_y(21.42915097420127230652532124868456923935990699251458722347438377726692700489612710681222105832574598, 143.6784324800408668882775329738292436991171448915691566375910974805833118743929871842649297995936327),
new_x_y(19.86126737546993689431724615727065399457781956744966961099695016713724721318412221794753972409399756, 142.3533483945726119110231072775406368670820689679242596477171318049011737071991452578258113438675485),
new_x_y(18.29933885589350578353572117362181005439334112687903140338170430120056447300189273077844764194819826, 141.021251685985266887812635068508942295001434802795192570608632451678833218281939764692690864213324),
new_x_y(16.74640328101261030482218469757471776908519910468582228493019149565737967018536095022041568140262068, 139.6786875904205747186248697310544624209843857503184641176967338032872133154275367193162264719910428),
new_x_y(15.20557024061789193944178655025784631645937155205077298641647167277644821347866041518909969362256971, 138.3222639757068969830549055555543210975531515453343225306768154464796415907694586272745343872626052),
new_x_y(13.68004383201761371518125787971019067214781461083521779471751779503014355282181460834669221248363296, 136.9486734977644781922444990674783034048309636946432587153881734233155801916365738309481706562326656),
new_x_y(12.17314445281775296172881440514644734742308510780756453854063318860854572725744436283001770200429394, 135.5547168300296233927101257419380269791285594951147453259573951965401141329148251402206955267309127),
new_x_y(10.68832932321825722699356764681927967191603066462586789372015905814171193870502050569209253696969829, 134.137327204240475555221685918383389816607436714718820445743994964848698374857285581808671657611142),
new_x_y(9.229211437159087822874276725317280033128869511443010675213289087820504290997907513059013321016050695, 132.6935964800073408286488830401975444202770042809419649091285355284155397710983638516321681915438704),
new_x_y(7.79957661779047827015686376519182296081926642927327465286288029094072983392907716961173327771842983, 131.2208029345386923349159338983427641158679433415364776089166737855631448758861905655458284325793071),
new_x_y(6.403398326248466309798108674517222462472437945318793486416872529382770043104978294953979870302270791, 129.7164409322325677024474618092356844508550977197471641901813908646856529233336352691160770810937925),
new_x_y(5.04484984426796139526317493720060115624843614220669720698439666451951535173259284415200578325341084, 128.178252596049675044399356481264613960837904228890835826024517154995594100133301444590996445827195),
new_x_y(3.728313421576435442643761966283896687138271143994689522354528384948056821880956549699691298366031414, 126.6042615581096310388881103911086642358563324639031879628876987286219351676063057214146114563433636),
new_x_y(2.458385949245985097512065991645653719203473083323074324347740487952386704132509060010789367728663693, 124.9928088152401797720491497897164203753303644593373320395431155112011582466168777931849869282373409),
new_x_y(1.239880691365917280603261953231410287496748090407489265150708587617843772475730084860036867801625809, 123.3425906557255405486713529437284151191317334141715165611839634482315280382920844668575920179213296),
new_x_y(0.07782458083116175629011230550058156418170517809826274526375084121768704533706500309233487169269985897, 121.6526985557583181343023114709593050570471615753307366729436234281888395359219588405913597402454253),
new_x_y(-1.022549437794818379974557796346751247159260046739063373998808764640755015544941761154348276534153802, 119.9226608676977563125956751993907346747133509383421487447181813921016919366858921744183585070780274),
new_x_y(-2.05581555282020400640692876517471574596361105828279881022306070183106025721711099796529634855635003, 118.152486036896801226152166775959017337197494932425139230407691636620163113511028078611208087568437),
new_x_y(-3.016374262020789446105918047971073982795095050878147294712371236297311079159827713903312673611208875, 116.3427069894700413458302008493141249870182314646707997638181242111330087388317561811343970307235368),
new_x_y(-3.898478126810866021067294778632858797721069072652388315489679093436348371132389154521629758051838271, 114.4944262300388617457158960100240691782188428870273108783388646567857671432532128219866142871137275),
new_x_y(-4.696264638842477779758686438840236796773307380501470961289944383414001241064852338999404196551002471, 112.6093610765833509194656523102383275703442748123830670780306307371593403067430253470021017120558963),
new_x_y(-5.403796709309370571759182382300720565075341168064172032154322720316963398190423276339971978453451164, 110.6898883397535899014589695023175914139781932043592688689955749000473196751192059722681683226754889),
new_x_y(-6.0151112502382433864123608041543673982358427368004740493410251720495204709938396685005873263908697, 108.7390876274337441448040149108631341411085442962999474674561742408461282353719267579350962392961962),
new_x_y(-6.524276257051368138294538210305594403548738781647597507165870814660224380990470428914299166216219111, 106.7607823235473385712946756294083466765681009529544811938721885825301922702408142023346968992290565),
new_x_y(-6.925456720066425214536824581102087829859937386072659177200765626361332436576356626689549077840099275, 104.7595771550881173625576023858723401494548231500922051609505897393679685532408180223437355104682949),
new_x_y(-7.212989586792222309784225436354105750306746264852407886605152461201135247322675330611544751535949778, 102.7408911257747537830312345495689605522024658709839118034655590936560194660713158571530654582803093),
new_x_y(-7.381467864448415824343362931303016188195221674157961308987415108115364378450811795608956994151689998, 100.7109844618005143701632022906252419342238729635211172991407165983889454251447424643610851668291767),
new_x_y(-7.42583379089137749100277746055940844597668582992462552536823248624021810764195849742622278996748993, 98.67697808879235701593293445883171338370157647939334535269688577368841002063779510174131411643766711),
# new_x_y(-7.341480810243836966643206201381313816003199055789657324038645019297103213300954914062180214531920878, 96.64686404392368550046594050079534958577147713328887485704197759515653347176562845811869070693414365),
# new_x_y(-7.124363865694315812074723560997896463902840017457323919261335171558020890477128798876607772410533729, 94.62950512847822715809145932412944810433410710571124723026090845108093510279700255930923780356705954),
# new_x_y(-6.771117265523883154130482521368336572433194890028193481877528504976636508622075434610791401621822931, 92.63462203009290640421607840667115908092234532799835462751396005691459156995027685214836392092919226),






# new_x_y(-8.0, 95.0),
# new_x_y(-7.906242498688109933117542054731158050479522453142107107076379389044292686278071914827945304924407552, 92.9493130656749440120512958084004985196838679500693601630244550840589446009310022023952981318019306),
# new_x_y(-7.813264190353248832041705106628202878485828544719091722469122634222393742311720039291706639231658143, 90.89859065886988919145988298465274259932839929026641681073098146445031794028632916218777023984143021),
# new_x_y(-7.72184430917895598828691306172846869968441091295044844984559873773598390577451515880241751506353202, 88.84779821877403890798610486382865082473260292075312234163754930011358797798776255107396783449496711),
# new_x_y(-7.632762169984000113091561292094579511216223053098560015291639434331584500089753019593093554064721014, 86.79690300799454474056326857885594188190010021237073740107679976444421759273529301987476452552400145),
# new_x_y(-7.54679720409530165317536715461355791931646940171415338419211310804059048162948055862857594334386246, 84.74587502445948659693750276314358027939292172705150637939637570227844602713755610449074101555254615),
# new_x_y(-7.464728989886641188438241675397055847552195957941189389457570903472538762807974682251432677727561294, 82.69468791354006952288619961739251653409194913106887906664328409075782246627946802874313927859385347),
# new_x_y(-7.38733727620416197322012513591986221177182392973034296780916361411129926984965316903424394135523065, 80.64331988044245640307659083320695468355564488121121210448754940607102641293556741092654604777041023),
# new_x_y(-7.315401996898987796221447365320407606136361267596847012044937496383277168767544714654435404183140859, 78.59175460290023481038661549456260959979512354229090857270475990660407293569471399620634708743486677),
# new_x_y(-7.249703274686546566343398333641646976778007838863323937086954445216610540305274108525274390267167057, 76.53998214417423551418807059719406028227965494490340207115102745637559981251075174077308727360172143),
# new_x_y(-7.191021412551500533265843281987910850925113583058107075134473929285141778922297264898539392548749261, 74.48799986633727737558856433999695608352670465325725002053988708112758649287439997963802006912729498),
# new_x_y(-7.140136870916638289238707198673025956912664791760053444469352999852152562589839509638313994901105817, 72.43581334378740665176506613655960809272486907437693277127989283127689861505414270635011035205237937),
# new_x_y(-7.097830228793800795734212039517632322546615718790299698475836838800605438669861641868183664862239961, 70.3834372768943270250392290707058911110870547278526705531030694424664488639924183357724660968896215),
new_x_y(-7.064882127135030449239708012480767490540187071218924165851565140754789548694174336134498172936644728, 68.3308964056399801762080954029813413390384887704896307805244546135097264999810667974148144259274437),
new_x_y(-7.042073192602801762638653950109206991019617011052922881834547175963090919880969385287203179452439128, 66.27822642306563756327636690692035957794083779660077529490186471124274177810942120662362503259297923),
new_x_y(-7.030183939979585560355248191239242162623459930039762942079317746591596953573974049004759084684111362, 64.22547488828440690284172575295932158747360411357285878456473087174717277218103314512624977309254121),
new_x_y(-7.02999465143930303101429028129969017652715186987974810694552309093482865835242748468823761279382209, 62.17270213875974964048774895315294453244161055587275519086201404927518282536841035725915210662566288),
new_x_y(-7.042285230906646127488168807299609289438080515093642174731655654638464323702428490495496820307930107, 60.11998220148746047507124462567945350464207737250713201583249340496575812169168800023050109585768918),
new_x_y(-7.067835031734998433920911956769825847411075927938935365204684969663038429460505498569345422301721865, 58.06740370265059377408654203754577196115594336643797596261358235263583583799963766702617124908712192),
new_x_y(-7.107422655940023524183031094778957944245225880875079526606239827703028914014723701386856194205999316, 56.01507077524405740455492746696110714985052405146613490889953245660715538425999172248385466228342406),
new_x_y(-7.161825723234151529954813240639455791308217618622057885942607538756567726294901861405680082164232721, 53.96310396408806195786767280087263188365507323263130025652776311870072655077129354783775756708364739),
new_x_y(-7.231820608117460494141036945462914436143209749023841339323719353381951817534276621765013543111807692, 51.91164112756735045765960786013448039985819071939380825974568973719937897844880612051654459036072652),
new_x_y(-7.318182143293104473758286213818216834379123245207805925811712018966530623286240527596712389304220256, 49.86083833534618746252659751575909470234668098779385302070973187757388025091774606894188585500929073),
new_x_y(-7.421683287690790071588655660440360333680085531912638077060090305582382337230865430649209772003246777, 47.81087076121751451574815629334245022711518119357702636967289941979676807418482840240776364778513955),
new_x_y(-7.543094757400166049254164070825925846721400128104484237690168668135051753538920813215008909858960409, 45.76193357014855034263823117713263070698562085734013685312874690818289915176613949093777869617989345),
new_x_y(-7.683184617837702371866746533823488967702373329726900425117001666845144845035834749899503725105608806, 43.71424279848451133742859356359594633153293303553112118902716699105034233023106346836781902930043524),
new_x_y(-7.842717835496045624985431687623653074069064371026007868026869812744579485005623247573228940077009949, 41.66803622616714751034262813519596730044668120738553802151665476936440276920828241145682383177392934),
new_x_y(-8.022455787654311252009896231653464031739861407216125133095136684427818347742923433584696978644875708, 39.62357423971554391720447075236583035608925400404696383960056165092090554117151637931530649069976088),
new_x_y(-8.223155728461688717332738507973579721580320293968137901075047616040858283827716543413340800001771343, 37.58114068460325801502362966884501879911047058178933375877270820155508679038997694046161430447740851),
new_x_y(-8.445570209845484795078455212442132360979441640926802299085507193835570293795755678971576550441527815, 35.54104370554849879912413409511256067644739820342560703214298123469290392975426816719463015358644164),
new_x_y(-8.69044645573871639900185623390214453631480349783233282414392847720311598019853621712660167396907949, 33.5036165731128741339308792364948661814584376277098176964461858280639808113664723020935175190014209),
new_x_y(-8.958525688172004564698077968084397386307333120339111964643141374022390168702418994283211135686357884, 31.46921849487943117836674731709322223748367584135809228273885070967353647064919595366116194646143409),
new_x_y(-9.250542403830239941802293264122627634860662973480781799477253884195977024129887069433197691255065434, 29.43823540935250792434206033452021089291130694870054484152383664431615517222461332976193760464251822),
new_x_y(-9.567223599736725385342293856453873315355953320190393911967549411547171758383884346685033857564175701, 27.41108076059054470230526607757070640859671080237906010381630405641283774386859018675179280498165583),
new_x_y(-9.909287946796694540813193586319926390428818727182311756766289224679666157395585027707153783428039162, 25.38819625144874338306317570347113286745860751925254759274912087327664871937875016166782235490047162),
new_x_y(-10.27744491000871022751274073568986647048377539096157138890745704971752262686319795495727185820498834, 23.37005257317160947776316957771341091378790663535869556488770183220765608098739794541572291991791293),
new_x_y(-10.672393814236915639403186307023883452661988460004920539818871778178424641474935324933926438534744, 21.3571501089362994840800102286139158849707321766237928735297137265935823249975153823397468692200514),
new_x_y(-11.09482285452990864570792683362787459666030455748539274582499122523294628317711269603127096266304099, 19.35001960880668809817452113883484928147476266562685777631894379998628243785484795112688056345145427),
new_x_y(-11.54540805007359285759502599184031649256071802336469604053361166705895406277176099072301927666383727, 17.34922283341556664938020823654082958655611848567405394446392611499903898600719473494234743841635175),
new_x_y(-12.0248121409761930311649866332037779089697546558583669820790729276775405814044919510114475191219742, 15.3553531635488226575908552319789611578779940412330252838320615092522461792337708031832930754666879),
new_x_y(-12.53368342720416221834817184420801737163955304545266638499661026481405732996856325276971857920056541, 13.36903617266130548940909214161821186669700898329458245354799383421732574386978548048667923989476817),
new_x_y(-13.07265454911841166699981636849545530718705007772780673899617361417017719679330310398417149060935653, 11.3909301592098713068074301392564628941071696674607273333146897936770881927941783322469442446175486),
new_x_y(-13.64234120920160625076876711374852720701735157483405376471294359677539761739536907970180813596862203, 9.421726635545379376499089427454689767144072168192266956111254804719852968210137831616664508283703066),
new_x_y(-14.24334083471962310754470207162999604732591181445495466134856655602125292460546929654732819334458391, 7.462150769962782424186732814437783699371088758974007004117746266612605406175012637887166499126721156),
new_x_y(-14.87623118122409791317961026645076562741142722469169218426856647064185476816168082720887555182334852, 5.512961778367564450918581308815268757474954212484795204608832832967769734050480331540719478937364661),
new_x_y(-15.54156887697868108680052006427687257696359850678894919436185199782562471123945275032860033383148256, 3.574953261878321702173600883222961125005200381831150758108197956931138066976719304647579747516467061),
new_x_y(-16.23988790857958803479090675936424998235203018610602792611184872529562745981667551348028705115423588, 1.64895348654999931484186826913652094732761969465614519551720991468821080538733720979934333505646869),
new_x_y(-16.97169804824161354153416933893403165328914998866037286583041423085010252959758862226448512350041242, -0.2641743987290227473343303614420630273695360993582744096904356595432079196716460621612383201413904946),
new_x_y(-17.73748322343432725346492476147251445521397220919478712753337603176546891156934197878186866140991619, -2.163532210239314703035357936615920608771889531285265982121744368747722056181450731348582427723254611),
new_x_y(-18.53769982977997887018956311449164341566813757848180792342588023588085398852559032543285898507783837, -4.048186647526390487674290593457530712462681850882294363073411532450156988990250874553208184478067753),
new_x_y(-19.37277498836498722224299911086356416362430903249342102127071622080234735491283668718166101544225483, -5.917169285898624878755810156570450718591386865241545553202864336808670679043222573657030383297715236),
new_x_y(-20.24310474887099175097249701118396533371945747954553686811800902432221494584554881386737279038915223, -7.769476647741953242218380020362754187119185155238856274342924875277338318888680157919412520192043918),
new_x_y(-21.14905224019949610656186875394453156962889857510820995548065239319619683985264366625411537750920221, -9.604070357164074024043614311669789390358681637867496024184674271537698396058718803066215903013326648),
new_x_y(-22.0909457705462564036934719676918977082721154920755142395566933023507704343861609130568474407209835, -11.41987738256895671704671140206182755835899532442546560180239075007109915505827580107053814974799028),
new_x_y(-23.06907687917784390746709080660626684608822996629131863067132488128779719313277633120272834633534313, -13.21579037183960928679386743839583430890949038678113914414693563617414890039207460083842482173676614),
new_x_y(-24.08369834247325391984878755581557054370630144888651251242650486243023458176907679511368842087509279, -14.9906680848720806264517964642723952756045772474648794651282042220288093108192332169187191500534012),
new_x_y(-25.13502213711799222060158252358060954737096541150945983198149276984446131782616845132016249843343169, -16.74333592825531347183498553315503774543076607331392736111023599853900949414670853991389150536890397),
new_x_y(-26.22321736367662016373587501085037295049127827379798695501282401586082042753052964465239556663181945, -18.47258659692842533445242810221885910839369502668075893650256640418196032066826356928586216594108469),
new_x_y(-27.34840813412207740581529499048220784267165177625517154505938078601947705925096084909636060320525758, -20.17718082766793591117817901301435180777740575226406359382738571861326004662828282030072885563211577),
new_x_y(-28.5106714272659443997867025682338862434205048528404120064328698519392182790296421034035443183079683, -21.85584826926098942608387284192103571175633640424992247303411230324078638837404815545424413500914029),
new_x_y(-29.71003491641276643260188619026986181026101854764580343319758724679245360425757632117765972500890037, -23.50728847420531916981636667456853668450729789315947408276883344412208517596438369466228743903127372),
new_x_y(-30.94647477395317439891519905606807052967604032695309833673847453141471724185870419639002317293254461, -25.13017201674109709531229004758890648948026680374372731638416829436757790348513810375163398537836739),
new_x_y(-32.21991345801421443923773095874866320238060291024162081890310435583148655351457393152785618415178125, -26.72314174196241393573254745412198640860132791225653969500209156073449443602652921403498662549986846),
new_x_y(-33.53021748670035416454139108115017487720006107745675697931159433501416950762389386206318209828655583, -28.28481415067541715762591145439771426238969947938615127561321257052725493025920634315997897310244566),
new_x_y(-34.87719520588425888308871154362747758432899222440838021858236526904958287849762503773531797765670994, -29.81378092456454882450120514631006176250442377785024863750102056315583034194474408942908837379607369),
new_x_y(-36.26059455694169709161873994518097217849105097704474926492563842794410154798302256724508810248680878, -31.30861059609630855494942229568929145229534649609329727259095396289467473042021640528265854869785241),

]

SEC_7_WAYPOINTS = [
    
# new_x_y(-37.0, -33.0),
# new_x_y(-38.5306323377483917671519454761669219811624428214113497755063377407277171562188066184440510017754662, -34.3679444500061538352939357139056237422459636548596895690975464504128589180812493264875496151051556),
# new_x_y(-40.05859728645511436662039197686967395261834798367855083963785752085915180040766260303026753731886079, -35.7388696765219862763922181175801207686455377338131287986370752068767421096770523608220349049659486),
# new_x_y(-41.5812160316673860253148011302545861465190384538302935254223671071703638702854291737407906536647545, -37.11574619009643154360452161713057052873370240553066206243059522063456230924139757129694845649659784),
new_x_y(-43.09578701083021483124486707215202225330586230360485223256177295723208373360320986684321003236779249, -38.50152385515816455001191645620081359684336833994418295210740713043742431651633533423652273401400086),
new_x_y(-44.59957479705413600753885140827234134852147853375312519224104276462000877629064281221728920393680644, -39.89912128238382897512614572941216856200720192977295864146038288474463571658106482197012049518783647),
new_x_y(-46.08979929510156821080141467587107972141995944381648595081939584026353087249944309784474020769499243, -41.31141488218943682633319021666985682296995867520198534523557906429054741526904283236009223791164086),
new_x_y(-47.56362535834115138228649715554108353057155584519348817718081846608031574016301742237548650037553781, -42.74122747078937087832212430345194924845028400499806730755984309575981920347667282249306997877342048),
new_x_y(-49.01815293932666708683772031565675619771916714593736976729560032760730625437403146575125656350941682, -44.19131632415059975306128234519680902585491708546593606502719104569053549579477371255104552821624753),
new_x_y(-50.45040789141293497545079498830109962315664075134428192726936500029935684108326956893193142185149146, -45.66436058015805095430658612842464684411043826793418522589609179115323194811087317169589242500675627),
new_x_y(-51.85733354433720515762977576177920607238030722194130734069372046569412206576478961885982784383856966, -47.16294789548744315976593327750517872096949133266331877656033732056631705358582587710586117932065324),
new_x_y(-53.23578318286253329092087225580047878464631493922232683346449817287769128219566585004081745165445026, -48.68956027115568679581829889392087640539550900299387107556510280771946714068811754286392855338348537),
new_x_y(-54.5825135642693521166492498658943980169092516401836626527509356626910575070734337564285040391718763, -50.24655896960057460496537880621176638535225990957635740402999848458062415104881829863633249315240774),
new_x_y(-55.89417961753983667272346019247253398781431870214978400781999175079201421232722863028724244270018888, -51.83616845655616082814815891770451934999709404079248400547989609310660632070668585058645223405487271),
new_x_y(-57.16733047432906771070532473187800884751126550047867492718630160508010175018458081529378916387564586, -53.46045931307171421107721662119597072601464039599597306742403998668308430421578028802977003889275995),
new_x_y(-58.39840698905375370975287084182548250759500922093914475388323725077900239675291027951540343648154887, -55.12133007690978309063560435717555946752937048916389504018053210034570975457329938851240654825875377),
new_x_y(-59.58374091242219652680099846578459986033567810677252101521631592725936243362846990705804435348097867, -56.82048798839433625077337689366703909800629806112447021091421186432392377276650099122398140357345194),
new_x_y(-60.71955588921827937689489800362378576143757549636559156056974363835469879449301653872717777547108833, -58.55942863370308668678995330596142558096352079715733792442160624994264089718566847191829532481854574),
new_x_y(-61.80197045684759603607310532615266887740882768374348607950258962388078368726432770802849166012583181, -60.33941449874277571509473542760087842218191257288662593630319366539827327439216504513876354753816176),
new_x_y(-62.82700322573481901953208856170936271740176926075414426909079107663999638510498594339442861646026608, -62.16145246923495794526715951624401238934643612606466104480769829024068916776619869842046358828345769),
new_x_y(-63.79058042577640091669698673313731524040815245092714402000005717093841945403106593958647800095030803, -64.0262703375782566571264255888673199122532111779057031978707708864726318738586815851550478061510727),
new_x_y(-64.68854600431931421064701759497951458833029866428249194610391463655234635224641743800716150972144119, -65.93429240452331619575201270679001244736421034481475096509264075425769516475845312443840751945360906),
new_x_y(-65.51667446014251786410842251712683858710446766972792380495638751031657312992639719352804626874124593, -67.88561429375040526026993371832868801543143414766408699450577205209489585779975675212531316746178637),
new_x_y(-66.27068659422288743090497471823099117155742418796659916174249745791783878606293554266111006326187098, -69.8799771300901677293165755462804579093653965011953316074444345007489986123841183949743318777116705),
new_x_y(-66.94626835120586603396721600525813755097347187727761017712030635238488381581947988221983174220532324, -71.91674126734199472233354958812788942821726198555620610420991821586257726903122143221982322903527593),
new_x_y(-67.53909291498615252079102272473567261537400983004378824587362771846075836214217581195987511471914627, -73.99485978933275717696896940853104893870480818493503274939351230365612481555442592424640828965252955),
new_x_y(-68.04484620713231509789435121723085746096209307463044400118168422351539488134952926272622735098859236, -76.11285204786673346928011947737820483107592212119154153442159733778901059770834583011020719860711681),
new_x_y(-68.45925591754894718235376481658208842340676535594106239519877471961822991833439684326475214524949843, -78.26877754331574070841607364938340294536140461008996203533242973744202418159456946583640475087144777),
new_x_y(-68.77812417224758659739376180161473281081703605874535225685538221257959752229132080561014890703790352, -80.46021049747144734197622008209620575756090824119784528714189820402203373935355035560948886433384966),
new_x_y(-68.99736391288908154847229675457568082515776345320479967563393441085869533346695336872464785283186231, -82.68421551351848080419583962818890526146059946827086116218912440747594793398447761979916779111085054),
new_x_y(-69.1130390263828215105295126861232893476796146930883455035965599202005503517856663288316862386557605, -84.93732476407009828545026218976614777190322487466467638661404180670437188900321050982283335981210152),


]


SEC_8_WAYPOINTS = [




new_x_y(-110.0, -907.0),
new_x_y(-110.0688761217366671569577760030856473709882731111576210937286180003478160915272407089594890715040403, -909.0216561786497577581681836948497058087279031690389132947311148787612940335073505962401465873710616),
new_x_y(-110.1383718807339634420547293396231119022475075436728412209039428980506767229388458443352883664226389, -911.0432911520669855903803843300323634181288003886353287874627509884135377772951326004337587431494767),
new_x_y(-110.2091068946759382721600814303908584905698910411369531920873120924105268673656499728712461724568256, -913.0648831474056534106335306015039201991960724633946213743270878118141382420823064661714047307585626),
new_x_y(-110.2817007412002934132734326943655366695170269013668808468174896384274915751436930897359083544242585, -915.0864092566238436296722945164591695997557447939666548274507930751384345428298999404509524627782773),
new_x_y(-110.3567729356423060336436536537433575967688316585073673873475054389503846149766927808834166361598905, -917.1078448689654985573701353240305793216191045106182782885867864256370457774733573457471227621174738),
new_x_y(-110.4349429060994622714269119155945674747799463633358128978266944335978057548722084915008144582492652, -919.1291631035431425095668648443873570746586210297547682028941102853725603241885322889489997507488075),
new_x_y(-110.5168299649240448153680126527438853326252117988330453078440520781694023495730606177484235735794595, -921.1503342420641451203167955403410397722691076448111670734244934776883943326718616051193290911644651),
new_x_y(-110.6030532757512418607693424083981801052434749605299783321181209649856966264894043314991578023778844, -923.1713251617507267605536615568149112107627955765435307067347371937472596621637873082141993279902734),
new_x_y(-110.6942318151707901657220183116233217836646470519387682658797826270418757663835610606111452881865146, -925.1920987685134482656111582602377138772436299622305168033495523513506603474387776475213420191073119),
new_x_y(-110.7909843281507588651716373480827710562313650564281492246576677550244166145838025438162141918687323, -927.2126134304493740422976986737563311921069558966272267991386977231267870060379142642430490913691279),
new_x_y(-110.8939292763228555215943786061861669037782713876272407874311894776467408104872871505787134665482681, -929.2328224117494482667368852160167187958202547190127855429104500718251196890622987473281962155680976),
new_x_y(-111.0036847782396295722649318568723685253417828241436678403726943921789918312058775325902236319455715, -931.2526733071148759157989767981568641950063257646086785470932978999212302295743681225452115109290875),
new_x_y(-111.120868540715203807101409095022378568971277201690705784023436959059145923770093493729941151775685, -933.2721074767994507335088258256909583705136545863472539509183216982190636836565795487810516155054444),
new_x_y(-111.2460977803627306304984735417335114840773771388716447939373780692809824223610944002605012059600445, -935.2910594824138169826360334665849704587723913771259539813670758048449182547248503833247646708683771),
new_x_y(-111.379989134443700307707884445105725226125723854721187460761802498544316960906736704790353456705627, -937.3094565236485860479737226831090487550538745954094936287073283306078798801516133907928224828521955),
new_x_y(-111.523158560146582470145918981379488349339657006813811158301135291484832852358244352330830590716318, -939.3272178760960465161615904011381777870986065555807142200813109406252632141244047592019716442632875),
new_x_y(-111.6762212214151244119550483404325997748616088675012686490446595162947387749107806063514057394914769, -941.3442543303748997443485233152417529570488616052964301370685767452907105503617685331290344063155437),
new_x_y(-111.8397913624500300670655131292528008101529599353580412552051235730933410346408431969546982807887042, -943.3604676327890130211803342589838132479083836899531523547274977699083171054726292215368194926697489),
new_x_y(-112.0144821670117770476220634314350890816853935645589574231596761415022741578745483244575658795423734, -945.3757499277795982593609830759279174923385437849498740091558297203993169637635371484195707431747713),
new_x_y(-112.2009056026570756370167584219837697625788104011690531996677021097529245129188536706246971816376022, -947.3899832024604827224375040254103848517728824347466360602685725249544462430123566788153301183255923),
new_x_y(-112.3996722490470195774827349245847518188135506760745823513831637852840744073337068479139910581044013, -949.4030387335582240903548359264556705862672724677044350769505200449455510281635685366729175636698226),
new_x_y(-112.6113911094714131359179644627315484956931380496864760915555513180751283687997070983997192681332701, -951.4147765371127172500631434408534520323042601267840834505469163453814056149301752584581312195697058),
new_x_y(-112.8366694047411789467524635550908460009370604904814780131044788022293027386431727554253287317057997, -953.425044821329623460817112984762060035880163818682282258642059689321096761490957092607753557027352),
new_x_y(-113.0761123486092562179908548208897788338493064626351209432470687649244588413008939399807393796856343, -955.4336794430133997780984397629904090492485208906770764433444997732425204738812188932579279787615025),
new_x_y(-113.3303229038900945815875368504579087366284444500573497232077529139941923129155639410393386427273723, -957.4405033680488899364337977586355708390284396662615025279882434376250094118060338409469506094360134),
new_x_y(-113.5999015184588455983858184316460309269294642354740470733224879516065698034374832537676583537857213, -959.445326136440325412254521029903652269984144605135358153897849835996322256264210492896078935498215),
new_x_y(-113.8854458403237656414661810269442492979835530936409823931061214795799611750137541045386098636964608, -961.4479433324591410392168189213862086130320929532692569165313649166358779303061265905936955281502803),
new_x_y(-114.1875504109792903504864400662314285946518781854919398067809219936313676352046246338989995306007826, -963.4481360604961924041125763395060436788713961030337314539856474230826243777645828209523090967152771),
new_x_y(-114.5068063362628446753626444443331019143686235487164028031733497523702590635860153015103510016429132, -965.4456704272597264709363771242045060560651480712039506812030040288101689913994267246400402499374101),
new_x_y(-114.8438009339558423853671541825471204244511340130899384250938036880044757370124640383671610149569526, -967.44029703100775085759602933191209070798185904063239816981879039144797492479804185388576142581567),
new_x_y(-115.1991173573886346635083883594414364156383639739131793747273684517184451937613158867010030980411185, -969.4317504585522137771479064598703377014934518336298485533167270019330988076342776527318965002709855),
new_x_y(-115.5733341943305256365306686514372725510809105951464741349405371905025330503127419049284549778775991, -971.4197487908225819170067879843194612831758130370967895013762108059308333099611337484591818062341017),
new_x_y(-115.9670250404695207305975348112391715216849258963526797982322922655006674973607960363414702144089281, -973.4039931178279171257996101426124926755683124199057086649640857245386944109306183703881818265613927),
new_x_y(-116.3807580468123537472852867223567943207342125538107224567138987658273996097765082404121202721896367, -975.3841670639093268808972654961499817558010514483214336743257174212067946477482979811097886218997295),
new_x_y(-116.8150954403636951351954277647816465192324359327124598314381719259332223005231054924148714580389601, -977.3599363242286126179303522084567517595094113291895944010808800044162685791459246546063568864208378),
new_x_y(-117.270593017474422871315046327807495413243628202300165223490378477558476590294247232399761330147596, -979.3309482134939707258920383839763589163286880030911211507153668818889009581020012617328521616617161),
new_x_y(-117.7477996092825895620044608897542422541785180640097528234059643994766894649756579377688563034742017, -981.2968312279796107147399335115224674551849432249494452216401812488672388702128399991104308489836738),
new_x_y(-118.247256518707393600758195136187349851521288358358320364892684563088701593622352827294402122583041, -983.2571946219530327587022030418109242986921155147250081164256546323152787158784851099518464416732518),
new_x_y(-118.7694969284962115423565480183845112593811516066757844496526518917006507697000960749371042190980114, -985.2116279996813313417843460392911774890030989913801751714221081476332685456308421633408916696888286),
new_x_y(-119.3150452798677255088981498458054731822946470260718968050517477638834933270892927227149395622520241, -987.1597009242461321688979517193567512995812296712102624742180689474811653975793429513500621605166167),
new_x_y(-119.8844166213405367095510449245564656415942911957140523554181448391545036350013368105925687442820993, -989.1009625444554842309121312662652842505316297503905792076508452188232292436628405861345518478864851),
new_x_y(-120.4781159273865458972607170178465394410882352464968671343028942881864375298414998240050814758587028, -991.0349412412000655861879146909030032354090608594760478423031953858792421871245599707781614865722776),
new_x_y(-121.0966373866019557460495356302426742987544676175833412685000041143922780421046852585218259651229198, -992.961144294660256019927103945629018289158970357962953507347187080373302995041299014108893938241432),
new_x_y(-121.7404636591461602512846289725091407105000737945128421097629723510204231319213262541861751890597615, -994.8790575738298059023154473433756587552513087844527481823032422740538503324669921027948840618359639),
new_x_y(-122.410065103260175896120821602647115158337580587179907848007918999154371947000132804340309607439787, -996.7881452498808013483596200293632269190221273322350354619770559535949566893925816098054661408937685),
new_x_y(-123.1058989707417891279666619701043421665885576190327299307859381611021044741042295597605457508713933, -998.6878495349531884092575105679482094160770218751040489725865749714083367992853198962563631678102101),
new_x_y(-123.8284085713243787750851377860733271264827370161197169541831228179369149692100538039664892728980561, -1000.577590448010061501286935871218423814138964395083675233560476089032250782179703038429119735273977),
new_x_y(-124.5780224059805632313528343875919909119928799287648445797236329454137270571343432668923321389751812, -1002.456765609457013781550718738529632359419854488490325540803216615584507973555972138229940604862957),
new_x_y(-125.3551532692505443119545486181230920302016478346831041500831436800236653083134544010119952978680395, -1004.324750066279850534229873431899167470439657663869656521399137951999953405898016486376062723210081),
new_x_y(-126.1601973207784031266856632581620762357972570617588488151192444864125666042420828556076435075361535, -1006.18089614950962283161794528023040873664734152158417811243570412055156857483939345821389182107221),
new_x_y(-126.9935331263277586478528766257746463736108339612228565966172236456963729845307424619420305766100415, -1008.024533365876979796124084975907840481884165622055913141758741331700708732757278602294004960522736),
new_x_y(-127.8555206686412382385706071550876214052171909158707393533154967817224171844617957953944574570757084, -1009.854968325568977585210156931580170371907517523118207266484940732581679808621198790681496960647078),
new_x_y(-128.746500328606229170385781504170905354395843477624369664636821121526980466047652073251616488399929, -1011.67148470805042190083011288182398610084979521166494800136431458107066307845070613704169808302617),
new_x_y(-129.666791837292464629441631954928545650431273184862227995899289311833854518993700148400698352318172, -1013.473343267958245248489335320276038310019373203885113406379639576660123925043150833461304745097345),
new_x_y(-130.6166931995352273651944149639625080408841957798398668930783782641645011511465106587763659352348508, -1015.25978188312099794933243687623574419597236619277421276508442355011140951133582802655429077178728),
new_x_y(-131.596479589851384932699829112427252190393827841643679211438427552767197734254268372083135441878077, -1017.030015646795920500397405368575367064445427300032970291971186326207628406962346006290657166519163),
new_x_y(-132.6064022215941527299572922228994493816667518906657391828912217598697617886921241065880491591995071, -1018.783237006252902525134404189605010221341708731345390273125336656600635739734292344013691797900151),
new_x_y(-133.6466871903764452066095508927714007572228061671117812652257976327491751757419097663019750128160519, -1020.518615949867544955838268270903531547209138149056208518321135294250999694155537197953539634347777),
new_x_y(-134.7175342929219321873171473460882283624829415644243953526322858943958921281507272874104648135427073, -1022.235300244914138625100120776457708004997878682094529295325471117016500626117752411398860209090369),
new_x_y(-135.8191158226374630258141245979079634582143689373635744133015936267211595402145096383830528991572832, -1023.932415728273248052359766396551948112183907362046363195399210664341187980506666086929208064333613),
new_x_y(-136.9515753433403289629671918820088908754010549640518397449417444766132922704143394932844047631927748, -1025.609066652287325366567158661563937891098285946839224316550285423589520212652117771986573808332547),
new_x_y(-138.1150264427188550504280175049597574309443921876059826738563848979943933731567451061156512720102549, -1027.264336088010943256183263481551818009864841654303857239871346080644373336069177250140353711805422),
new_x_y(-139.3095514672549724405062366228773088357140885541254248774529818576030565814797810965978918893687174, -1028.897286388109383477634083425388348786489703742857372287792538203106462170595693420012937140636272),
new_x_y(-140.5352002404926279259311397379517761731204699903860524849739167382596812905884216909005519299367582, -1030.506959711659986415145051889311823805539951012671919731919744255895783607349861994530296399380424),
new_x_y(-141.7919887666960091906357684977722937681379609252556213611352688422501004497321008570737795334067489, -1032.092378613104391352141138131176810342132938071550828196774546786894616838585315272481732940659061),
new_x_y(-143.0798979221064515782791064110611023308709208517336376215646747969838818968173597718254387653310439, -1033.652546697586092960254445364651070595033207540955488671086253475115880796740475119139151244484364),
new_x_y(-144.3988721361763612710705129038441331235394951990348190534695351185693493733653365803867872891306005, -1035.186449344886115831474375591567986495814471205476730002139808175905556003979027354642179819532299),
new_x_y(-145.7488180653323209296917312023619145508751626932502324475411361868158882793521497266656827253082485, -1036.693054504139568839686272436744877400374740431324170495268683948392289304043440502293308423410537),
new_x_y(-147.1296032619974942782941655545377629467235795214363359303309788310361699068400885241247487978482902, -1038.171313561476871658129165831032282234200835586549722271116180432515198444875763294073873897037766),
new_x_y(-148.5410548417852206509189140458163211004445412670931292152322388003233587768283589572788155288478338, -1039.620162282685041539334407408612024877204927785775427485121476564003025535406157141430167836359818),
new_x_y(-149.9829581519609756511733620892931512121993478368121839587474950016231081116550104972248201312806029, -1041.038521832926063988231865584935537890127043383052688837733337598175682351778078705761815764707412),
new_x_y(-151.4550554444583005280157078712505415411060091573375788152198505564039797116395096483515231992928473, -1042.425299875480532163206150587689464112494234894104764656445105940393335609474077921472249294193163),
new_x_y(-152.957044556925472626115663854136198012415671393861424617642251918447415942316525400770421106778365, -1043.779391751404901570526036663379083753773005086757816572911663735046663449160215772395602999897603),
new_x_y(-154.4885776054731527875428723905498372937731430573298828953852989615051227702161998938292606086769587, -1045.099681741899353701474409699019003181762508923498917904270013362798126316014593244564257801525756),
new_x_y(-156.0492596929885191674199143366245442749503797699994472627991534036947205903298044074004926111419393, -1046.385044415079873353586360660789411528540783657528399186792417947363878947119122786157184958438549),
new_x_y(-157.6386476370779415798256306989412715114809943269866704647202362198466385656718927117031263668130909, -1047.634346058732214068439233395330193014676413945269131663691542959131823632511290908567939583517808),
new_x_y(-159.2562487218974850328773910157225255521369294675655998888085268076985712491666905543199009831120782, -1048.846446200496453704130425225164977158627886578539757632524304394857075830710770765496626259774672),
new_x_y(-160.9015194783278295776305961094046363498651067240359447123653840306389928888432989694090028755333623, -1050.020199216788333639961499956218119914714121293154884021496329759083003822886686236083750256755582),
new_x_y(-162.5738644971468727810908169180132485522409203819654798745021685141174751087149861590194611044157743, -1051.154456031607060122886685858427520470833033905422172826445517956020093380632585225904137452988548),
new_x_y(-164.2726352800486115247112484086107587000351215824930080824291279599351844072323252902882987793052133, -1052.248065906208268065004140369996005564309843341758573589694070310494198591992463367715365218727609),
new_x_y(-165.9971291335501034141385495649436657894232697978926881185694315623379452328491856469114982801331098, -1053.299878320434965529573040999827705425810967730762448251153714295586965970988614056023710638163313),
new_x_y(-167.7465881110185473493843236475642897508888783437230722709255232484375491623338263579815523224018972, -1054.308744946298084893051398966641114795807492442359007936625780970956072915662918582714298032789712),
new_x_y(-169.5201980082369111816580616766508580690384061217367751828940084665195251051900703136830484209024285, -1055.273521714181382061686271918388584804843811698908965964382276210485095608987958149993938124497645),
new_x_y(-171.3170874181081345786312536365632422784546912103918303663904488744639838702650760181010690885204516, -1056.193070971812493002745630720671400646540333897878177740217517165526228796350673798708075828320042),
new_x_y(-173.1363268502737490170433688635578595680210357242769719189893096355510803232297717185272667824675412, -1057.066263735892670405241772419402822278418539223132415021254893521104099257766256653941058199683178),
new_x_y(-174.9769279215917387418108827606344517881870216574173667758117550923493910395846689650674717614994336, -1057.89198203601180658722560781125273144744596130194803189738160492878927737459355840297580859649114),
new_x_y(-176.837842623579514048482606256521439430437342080149500332502387065076446009777539526183915856319478, -1058.669121350192577063334995786002119517039233407146595454282730170714397410620438315725575177114487),
new_x_y(-178.7179626730798281294125549220833931991482808180861417186506899151485587735101253866089691890404622, -1059.396593131107733312719046207252308720544146458432929581192019564032431856721159753339516110220091),
new_x_y(-180.6161189525491306947070813040985802981738070099814840601747642355489770307945905233421666035970709, -1060.073327421697620753677302546756917791587078706737153589866866649858008955287275623374070532507566),
new_x_y(-182.5310810464979639468039185895983697634353937836005921945768794677454821406957384621448151432788068, -1060.698275558580823815833302083440852092022613899159443499903149802058318329922975117913286453902309),
new_x_y(-184.4615568807302557703172610654894562869615462333354386413269534309740814936434302666538300081549293, -1061.270412961299462866313071573613930426595679656899013785987625957870758075771375749806636856282275),
new_x_y(-186.4061924711313999405200882844171432621554697892675697067815570890798034828143952824898694961748432, -1061.788742005072154264206803217404382035380191279362517645163341608207354777314156872576632750062614),
new_x_y(-188.3635717888424319851761031931063516883496664434750551762576029047745788523807406036631763412079113, -1062.252294974342148815210476365962483577598591378905067801418455979674046753597728261179068297372938),
new_x_y(-190.3322167487279688483463421219142484352910120213421733652906569746778199623653860178190648741379945, -1062.660137094005917855339472303695338155659437667677315025513288990126488400463991391843305125614724),
new_x_y(-192.310587328097398245566411484523931520326908377572608095184388998553565990806335657331542866658766, -1063.011369634788776486590450040205907663858251747942493127644411871424464644209947855840978445801462),
new_x_y(-194.2970818226705584910389607819665260364963603579203965778786714944085935049255899824879467047057645, -1063.305133088799441018789122681437086514839328492274876185800651765727077994912922227662613355649018),
new_x_y(-196.2900372467892933741211549270165473532852523447414286955988295860115296465825911328511053890137507, -1063.540610410845207865859223409410243381847297419103427070308672516155369018249238648211825813154381),
new_x_y(-198.2877298848632173458952641226178995281824664616232524163763928744400742472965535072596365577815113, -1063.717030320624339124556553641500509418690485205902855725681265022262922858806828270219965366926312),
new_x_y(-200.2883760010001830114277258129204515750560605650204500651832240766625042301774999835073766596981301, -1063.833670660432971886856551729224970570007954837361183808299792364684101435370535756402030741072671),
new_x_y(-202.290132713707688945552820488139388142050845521752130522065880228929004617829409530074780330024126, -1063.889861802531250091803602455165545891227416999736326901218838631108057972223847395792743102052151),
new_x_y(-204.291099042459168363037585529571669001491961557642080151423544754984443379186637790602093418233657, -1063.884990099808397322719194059094452526827466460672381571894367183682545860058527445001886406796869),
new_x_y(-206.2893171327971272130846741464864800735483288762818626925306690684676200953998658519339568580323155, -1063.81850137287016161521581381466353650212064332286582689164340908066394002068662519141975596991706),
new_x_y(-208.2827736664918219294170671914782676794470208889142955959182195640234198439284499654556662540894506, -1063.689904426145703202813049471740644345775957382530809139959148748015929433422542035674760648164365),
new_x_y(-210.269401463087963236947090439881153914876702759390866611875539770449261898444496270706438941343809, -1063.498774585075892104019171636577715211583041777308043462732121113316447824780822086377252426213066),
new_x_y(-212.2470812789512028993492798024974029162469343018867294490695227455072443172233239919310351885851931, -1063.244757245902636959156481827953368172555099965637322567430678200736243377217033459335122157214603),
new_x_y(-214.2136438096693345026261045493763144947050489549692748370205012310138570007242382985817331168351815, -1062.927571429030904449108596225883471118019518494279272771485822225105848375687330360232397399129813),
new_x_y(-216.1668719013686826727024919584524833767869228428465722565670657232517292303826748098738211270137909, -1062.547013326383279240986860811599881252283875627198618259265807734964119878231469541653914867270329),
new_x_y(-218.1045029761725864161424297172362735533993218295846149530548848760267070023908921591232803485720981, -1062.102959832613207001790415614617353081755639852801604714966177395985737526529694550695800831730722),
new_x_y(-220.0242316766547749058388071481403621663556724051425794215134227112021468350651344311921096196191442, -1061.595372049489518146481229047467490801704978034799734345723256556485501172278394221213388604725802),
new_x_y(-221.9237127337244356004840023803638574458955109570111469996241209744209503665744334932206202483859513, -1061.024298752213694394881369913352534559377564163846891854246830794130519355139250685949495106063269),
new_x_y(-223.8005640619206171430941615640073798676753892630674560805619022489429851495205695181516097060933752, -1060.389879805885011594775083383305192258707502589648141874628690870763639532876680541851216367295658),
new_x_y(-225.6523700855901191501579978237715142746106142208576699899535885363484310919362240369599849345187111, -1059.692349519789718037263367934651952158848127571552930483808712178219761599495581962031277795643442),
new_x_y(-227.4766852988741355010882851562339497235541589289507564578812871797590128885094753194805789245449172, -1058.93203992666150202183315744477063078291574438602468303337999062012363987956678568057523408670361),
# new_x_y(-229.2710380618336965412442199898581843331502125114693603181112062010263171641312041835528318526205742, -1058.109383973544531658871808042810224518987367835438786977517861889597805510282653387535077668004459),
# new_x_y(-231.0329346344016011050284667936824968952009137700051821604933864442446360882314423833220958651385984, -1057.224918610390347324406650897803394006324087217996045419693569490411999453326280145200370559866665),
# new_x_y(-232.7598634491583890902700026903956642319908345276954336081365827899522731744182947318418411509353742, -1056.279287762039020997955419188770875640552351825208731249488564935824960025880791838185253143018106),
# new_x_y(-234.449299623191499549614270785632387969430978694941057455849909854664010989244943526402658411889843, -1055.273245168776599469308284917027779342863954304673817004387903677771739842346276173520031047977602),
# new_x_y(-236.0987097085098022393402125877505031275208252869129028698130987453978141589626862423395314492067353, -1054.207657080228411490942472089548924385691169738188848041408241855393899329966255140065280117028853),
# new_x_y(-237.7055566796500672052561957330215468044605517079300684985883686551257619207477697905056727402557996, -1053.083504786944923268882553212782182896633569260846848632193909053334597205212804691053148900922248),
# new_x_y(-239.2673051562277901661649272740105808056543265881407916396153589078487325442171257060039456312350942, -1051.901886973667259902117684119385897134097829669663416799153288326670931191033478946401813786758964),



]

SEC_12_WAYPOINTS = [
    


new_x_y(-341.0, 208.0),
new_x_y(-340.9266881203616154451998445941891994256363478651103623585716149563450848798717418133395469854154934, 210.0515196172514735605823821435869182910005987417918560053148648002742311389538227287960079542367733),
new_x_y(-340.8532962918122812625483208537967397497088500597452800490329982409846341144786538108071927269349288, 212.1030363758953988306396422618434522684078520755754352966683601790619600058777356245161464437616729),
new_x_y(-340.7797445657844005253446068759816083942988388334350621112072609138040115903291572250401728835083899, 214.1545474077303696477961131864980024423099718831260645345990676155235124728439313187525203999885292),
new_x_y(-340.7059529943990004775962458679321535362739664443112725597831947214081037375620901128529166146853275, 216.2060498253673328660592861853602555933477408206715923534816253524719302741857311836785679056451049),
new_x_y(-340.6318416308148415209439249952135709432908920883737411162934692344066527004437008290072765754013909, 218.2575407126359373014701705579158610460775475341296915215027395626906373673865797215788037174744082),
new_x_y(-340.5573305295832824296098614847642812159449329894089980623023068447778935611656843398662118903126679, 220.3090171149910911070203274333381721035729970450817670707251312758862254110388822646386074362498577),
new_x_y(-340.4823397470108204443276747407445540432247901178343895652843602856938198061808729419418614531525037, 222.3604760299197995603810078464341977673749682378821314643256829892257101825937059991420342085595573),
new_x_y(-340.4067893415312248185052195661942471506087683091017202083080264901465923852793311340877777652995385, 224.4119143973483573969848890004047257063051840658785361906835322889879713583639496299861030574142573),
new_x_y(-340.3305993740891822891043612556580646659036242217660063516442780043746407401869324076417392890278798, 226.4633290900499725071133148072768188609922519858235268844372418957569770863476475960098255919952913),
new_x_y(-340.2536899085373728219712335246757126960427716493950576423567050391372477545459599356554235027635263, 228.5147169040529010389079878088626298808530046953121206580508769632564995308433976960698580776061235),
new_x_y(-340.1759810120488938341160421027370524611809385000939047190241784290784653566951629213031194483837332, 230.5660745490491777095501935821146531379466215916319659067438454542655486571125797087280780209429115),
new_x_y(-340.0973927555469509298991106642443802545027220869392227723708151371310774651023192218802331184024953, 232.6173986388040294244634889452828170890064618276094368899668090010482505078680890855262248180543296),
new_x_y(-340.0178452141537329826390026261496322455666753922589340101379641114098462349075541536379858055976015, 234.6686856815660651384812842484626830844988215126615384692442139376395067035995684779172795829176055),
new_x_y(-339.9372584676603891792319558896664257688793216759835696399823598612641255501326890736293302223033869, 236.7199320704783402646548657470902890489024052812802931771956646947084479447680449186590792562897661),
new_x_y(-339.8555526010200253815891200561258739603962388272606309364455711403513336222860693212881172241300885, 238.7711340739903998441873015365392647208516033832689549130209763315150751681352379993915303723248023),
new_x_y(-339.7726477048656368771686703539376909718512062376740956729982215980175074360135473529848936563343653, 240.8222878262714111360649399824727099873245379321711560855290304013168788223781286019835871654503104),
new_x_y(-339.6884638760548942645822111045216033965205923616927018489195104543101588524118662111307884383598642, 242.8733893176245032662651061952327556371537144547741638395797002719031531974032687419181879176557525),
new_x_y(-339.6029212182436988709430294239351843976567857751324395048367177570631494781941177198353851926973299, 244.9244343849024390947844010135146245325168207929584510338484858411663946368054096657791837920329297),
new_x_y(-339.5159398424904236913914066909949526331861507070877839365102457638056039458228576119121010921672759, 246.9754187019247525127744214432740636634417817154545477839708505636394749972624833517134292735396959),
new_x_y(-339.4274398678927554055137651692089563536144824538067172010806231009922778601028901993818660435570132, 249.0263377698964929728473694926185097372873686801434150092391456238125743369027332490338747652354804),
new_x_y(-339.3373414222590525388755380420597552737912479203666150140797315280135004257626375913237846177871983, 251.0771869078287281822314783675923262871598972624822975209707248755685925883977186880602006601218227),
new_x_y(-339.2455646428161343033941299814218873181590713023672796921270562802273216859004255506562061392078276, 253.1279612429609655509675903951227851334635673303980884606372959525296461272574656586790516554654371),
new_x_y(-339.1520296769554140634437548248783986505980378769636084085619631447099845475689660739650064268522056, 255.1786557011856631855017820857284790185449144396995628000057022338349055836293875188649432851564377),
new_x_y(-339.0566566830192907317378876619768559395639071117870658300928322244848403066881368439510442216436618, 257.2292649974750119516154916226170878946527600478483313931649087151270585819405215207248973733416043),
new_x_y(-338.9593658311297107032455728448847896370718511919102896580969773614173566426035999873819385645428278, 259.2797836263101813997228913614856395067076071538695764579064657135981787219904082452186099852367086),
new_x_y(-338.8600773040608121271611794771817752263108562825747848770971366041977744798886809521853494476050236, 261.3302058521132341472371106598063642114483163101262005990126745731157875432842356553544070282164319),
new_x_y(-338.7587112981575625895509910268794911576017264140179807384718253036569275248660104387707900123425946, 263.3805256996819256560509941525163675651619100466912267501156787363056496896118805949704802410755523),
new_x_y(-338.6551880243023002159053164730911076880175015562217443789508522321917827075453676256627953388356554, 265.4307369446276192082696297232526661433802790438048817085126058208774043619773793314916963114321696),
new_x_y(-338.5494277089310873532864172020765775837200722708967840301547793931474394039270716333482963454740104, 267.4808331038165592961391590502866401717080089722161278174481354333236947075626930707541452780006217),
new_x_y(-338.4413505951017848660776788080427336190218492681598280266658023737206192432716897752930569081052888, 269.5308074258147605792690032821009967542728098581855859766051207677303354357924585058811698001881615),
new_x_y(-338.3308769436157539185566515170447046471468633267578143172046824788223393473031349189851645365429039, 271.5806528813367840347861292582586872532852891793676666779050464114585241046388688789459821689149569),
new_x_y(-338.2179270341950908817613836137845094165467043099263486448238143071885990619152085793304810885051209, 273.6303621536986869320925638882534781797374269949645732059796874734071856020630869080576319132688684),
new_x_y(-338.1024211667172996963792237784632271531983513999285484095634408951606641419024507632195285019924308, 275.6799276292754488036964543367779006165194411441181745031703918252868098634767264484946643456398357),
new_x_y(-337.9842796625093044896532161252730456129174552839204284462900390612423594008556443361905153376070087, 277.7293413879631916480763262986133114757528180168196196649850108001258954580837089508020549967687966),
new_x_y(-337.8634228657027037955019189479747071864393438942463636244468014162842152884631232518331459267043618, 279.7785951936465292066425943663679897046107799338127207816436582999230443521460668541613580424585009),
new_x_y(-337.7397711446521659449811888945121651346586320114557383569768547564388031840448320952463390580707337, 281.8276804846713972811256194857520785471675179750926924855739897791998557703325104071548931286984917),
new_x_y(-337.613244893418863512642377654120340042351567719765579461496957693265480370267960823250568326861159, 283.876588364323734724245531701436937736593575111520720256867596574120766161181482800226969036229479),
new_x_y(-337.4837645333208426804820220483909143874808633176889042063747732800121542291641628170988242072508253, 285.9253095913144029194714112668575641190152894698930377131765930981355489412785663743815852288417826),
new_x_y(-337.3512505145522213964142082088116472853597694543399990884129667604446970713270401933384525274609707, 287.9738345702707502868656005458316976364774155164348581035548984873615145173872790894683939495741642),
new_x_y(-337.2156233178731080093393270146154726399961380030031935046709393374403233147637078742998927361615841, 290.0221533422352475966238090247391739577622825262429233335233346269948794931306441747815818555934626),
new_x_y(-337.0768034563721296579932166197107106936244068579930681171118138033919817446748654264685799609339671, 292.0702555751716396386303235728366216515561939476833314551388488697517939645285655784708023534839847),
new_x_y(-336.9347114773034572487647760063966391997298130275657566932326173445959507225854448711056658033503663, 294.1181305544790790959038841414660995942654103070970600329809631403451150139562363311052733937451462),
new_x_y(-336.7892679640002112327412460966995344223537306941922756036426075262066862424559848442050059479805881, 296.1657671735147292929017065371289413729552340783758389332558020455416897610698073433686887877179153),
new_x_y(-336.6403935378661294315887994208158639027352202522979561558465256548638019097662295266622702385888548, 298.2131539241253438242152320088588795808438049016056279284485671823249391796836329027594534227420577),
new_x_y(-336.4880088604473752939351367790292233294393311883667888250538233979434643338238624841023243131430445, 300.2602788871883529443622989275220635405094111959417380066468681278809224997113258032080488966001978),
new_x_y(-336.3320346355863616893358500946746516800680536204539457678098001445742290670677207310773257743707725, 302.3071297231630089801234380096890018140898566589781872783439246768918335079475178854113157897584188),
new_x_y(-336.1723916116594620452635763343579442036211952894126537017881380517953920501258941262530594111729649, 304.3536936626521659390280030269140308746614958090267534300151082659479379529070862373772335571899513),
# new_x_y(-336.009000583900476908121249499592689051550111438110382660154046762942561498886019756545941765119184, 306.399957496975291901057439122547102375640615477150119208948652827770494296062737557955189484569658),
# new_x_y(-335.8417823968117203656563225077771373955808112682752091269611526722536044400955084274041530600673431, 308.4459075687533367326300014066915322817350954858113268990087045558065981848761450038507226097635134),
# new_x_y(-335.6706579466645865879241433299147804579637282022540755047676423066314430872580395634991470474359232, 310.4915297625061021041181315125069038212710351306545974974613157173407698926289014141536287983104178),






new_x_y(-336.0, 311.0),
new_x_y(-335.2488059564214508170089663068031163708240008035849618599370002670557327497989568418202688324357431, 312.9104488688671713072381613831449594193478182012580807749743128110522572378233653710366960031291199),
new_x_y(-334.4808902242912552998597074703800507510556488368989088040147084642593524670294845651899559060140601, 314.8142356851236493631354652484358454948620829971336331549768002787563456908276956152871892728408665),
new_x_y(-333.6797143692576300396326350200675774794744681561928179256216297577679151696673342902465657261137752, 316.7042482778874342935271413120499517119618307160507543549281457130876747122387627171843436403567101),
new_x_y(-332.8291266324484962616303877771588964100019368788417785179394145421471364302685496736847248518003018, 318.5724826981680243941800913365046057533202953769849255627855677272805754777773707443001354632856487),
new_x_y(-331.9136051359526541927147067325933221951670031648111022656630021613950880750757970154379624529999374, 320.4096197390552519557980850350822432860788445247606542417985507588731373048975047937002015511996429),
new_x_y(-330.9185592881468052383237749207798172179875490460241294753097386516409914588666853146287873787243447, 322.2046318376163072856861323290889352951571338724322268242118345402469373729920401785182755220673082),
new_x_y(-329.8307057531697052217890656872824595344969493704493959766123805687945387036002216560770475070688406, 323.9444361547927799980189959557630558440900157643191836418605646912842672492392832968137916862658597),
new_x_y(-328.6385321631929032089990652442439843824695590289078828549108910981855909789565878146589483896832131, 325.6136141607777746736865036463886940722686676930997616228814128971123218332571001788806520650783633),
new_x_y(-327.3328570804813449049207874301587476564387072183958479304683871294235586024183270028040913725304967, 327.1942232402548068792098399992378823874232143290958499888059112633041543024868441553222229866038519),
new_x_y(-325.9074881697171024979097616770683341041005183389440949169249230792479486189686910983559802874830014, 328.6657312576067030537870733135873493922675669255994415728485022737469537339804140458885730222082744),
new_x_y(-324.3599717386376035732189512119598677651287854437469276731230505005603177998899172607676520966854963, 330.0051100988282685389782774219673738076029956373046331820265308583359582006714770042565951384626491),
new_x_y(-322.6924154370215392398874800678900259144810202986226456332359920792037731829007520192147064319951068, 331.1871281441900503737049426700164218934898495857950846582675202652604712272060452970858061018754205),
new_x_y(-320.9123518169663329906628622280401228966074762229915418408692096391291639924159217263684457037364737, 332.1848834117210982515802741859463238936385404664217205070282064195607128390354103046458863885783983),





new_x_y(-336.0, 311.0),
new_x_y(-335.473079725544098041079802262436430746106015102145844815000033130798041271749249310075726316622282, 311.911484595507275980916077806029091554901103887421398115897887907445846827633109367340007870845639),
new_x_y(-334.9442208172323560874046022548866032286518325726085568750685872011998329405736364784173093533218926, 312.8218470073710664655705731052208850052765792243745119908392983494525210044937837651415174334338532),
new_x_y(-334.4114885047454499009714360279957377872608307832400133501750536479378465922319649537788592358118011, 313.7299584145022501304260405871832984358187675178913453632219310461520605565912792415701286718942186),
new_x_y(-333.8729558191815648240404326277187615121277283178489109718886884682419274207680013439411646632897805, 314.63467676418211412810075563738881403970310548120725486793482164149740074806262433653175979040849),
new_x_y(-333.3267076792766213953643835454437945269491451553638631111885399438742106969342229160504529824044012, 315.534840264994404399875459049784526293464699243694434002416766011399151521702069624427470920931126),
new_x_y(-333.7708451992470744441159227892927581861244236695663925717423130105956868981328684976499108615089061, 318.4292610119059766466051042159794282415630633872174348531545228658338208793532880958192122337957089),
new_x_y(-333.2034902904416908918249940388719132742360653284155815461292814044665330643128266478724857474087048, 319.3167187902783395594089424166806077072287686984001878767724127092532375422793485399546182957810655),
new_x_y(-332.6227906274707248246854186780948404215813227990576368651539232724334369296105800916424107816688582, 320.1959551078918438279183674200553817495989431318238944477991327254733171281859404003125112474342598),
new_x_y(-331.0269250474949440036760386226326542434471553270058363402735762304808458009012248369022015301348897, 321.065667506883927964066864683475722374746240935921353494039209139236995200999248475180017659717249),
new_x_y(-330.4141094488450941047894411994114096922151722272504542448053019507696003886787816354006563861567618, 321.924504210804638456415027359529433516403701798924937610409747797194483739436369050860123128426344),
new_x_y(-329.7826032520362276207103256946053753604147964030747742855628833700094480796181516832365899536958788, 322.7710591657293795126734559680521808173202593288713777224436065351912199243518267288607437427283944),
# new_x_y(-329.130716482462713089224496208988524361387056156554246781350018492455831916933424479056075154605939, 323.603867538483326221654997067615470602905268848347776789051445232748908019433252091308144620189133),
new_x_y(-328.4568175295216134846491464538532831523797186664409175791941252906194373434676175813659670991202529, 324.4214017394561003338366761316492091301732302210780613103284552144161473324851211308357456667025126),
new_x_y(-327.7593416315195858912237826853793038699846629688594340338438597064459373866592429602721812368896718, 324.2220680421392821446203596399927527596288530073248738557032851769458914360553622321598675711191375),
new_x_y(-327.0368001293700951017066226034537691029454810760944488145132360890775439826420219088683041912128176, 325.0042038763104140602148927157256396727912909565898763368061528998079351567229454788944229711031876),
new_x_y(-326.2877905246771966368010441227575913584338764104126719814826410848892171918241228823956678171674438, 325.7660758766088047863058767294923832866044236167546451491141495810537532804153014500895726403933014),
new_x_y(-325.5110073692200185055542188853673356953034703177879030620890006183776977403303231474751901326686821, 326.5058787729793147570892148902787689029982465968572593798322493283075437515475535678105022397785388),
new_x_y(-324.7052540029881022060408024088826262554215615633944914751552477972222279307448847092100403621690236, 327.2217352139633064237831276033476589494638779485437316580084431780480880959935612726970103558184686),
new_x_y(-323.8694551466634492863684189318882551568228395958801345571434612435616738907824505282141119717039469, 327.911696617937461848550818069713473451361834726425827290222647476828169230940543975896434692678503),
new_x_y(-323.0026703416967010326144871450138065798418785142718902669170133618968955052038498156981143787176427, 328.5737451509704366799846138860749090275467257241283861012995188515875779896521359171161893095787687),
new_x_y(-322.1041082167867618197051491117664928654916073645665721292290912579606423121064215856553845449301164, 329.205796932796021279577899123887457567806161182359896600787368924129885113885306132254067967098854),
new_x_y(-321.173141543561816913751388705967938110031912621266991836415090178311038909122247407690725791275727, 329.8057065742836738711369553594053238128336272197035001993397317249604682551695735325704222651840227),
new_x_y(-319.2093230265079484539863876634728755181371189676389945144258996103302342010208463138201809938826497, 329.3712731504996916807257928764271551155935581499822730420440995287556239270107310002899471681177838),
new_x_y(-318.2124017526535060627535623152983031748888361633683703645420519196088758490108139524338012239341238, 329.9002477127550105163373852717504393043368373604708412364171686140978731374659835225103447382559955),
new_x_y(-317.1823402051736727755540645661414662017159459523376774857126779072810255205327696708235092267864204, 330.3903424406734819631854959645558402732923998867436355333435280926374538243655571594583365856644702),
new_x_y(-316.1193317219432194609939471221791465417203188333410886481790561647942341012317602373628994752209709, 330.8392415310188910582509515821542960655470880709773657799512314404547375570879373958750870609750606),
new_x_y(-315.0238182551876972534225501501335613391469813421967132315994763069910977790990401873046197885252236, 332.2446139135106280093630018188491400169745339965759838302442182812286458902887194074297737169397376),
new_x_y(-313.8965082618607696372109207239911423932608801245961670925347478358317306211590174124734818687574323, 332.604127874850228365050172853738680545954504210711007834319228795575460369580297572661368984323317),
new_x_y(-312.7383945263564186806476191304357837261643845691086534843942774106399316899214828596565295544919803, 332.9154676603844619192034726218293664120207939111185035391535891280107229830579631334567994161544885),
new_x_y(-311.5507716878566886457608193201126176127921024397959255143020396780954072243407093706204540527236693, 333.1763521079582902471236467681062334891381602522637584469839799851628634720092776076034501727554089),
new_x_y(-310.3352532142917702046075708628771703743594927516414841044349554806915456083741178156611029698589327, 333.3845553502847869993774338062427223981587596491299633087996989613591012151833476232086475563197681),
new_x_y(-309.0937875338958468336007551667138729728202726681283834161418919641309123401107596382849695506253186, 333.5379296003175422635683545647115860381167337209135065552303875300136334741431802288245837041650161),
new_x_y(-307.8286730041050819904218803418674415887286628357812579167272099922381269540805643265379922339792491, 333.6344300084180569315337978168313381459520915308119816487368865139358994672092982976795098073755985),
new_x_y(-306.542571366574934477585224878467985227679627850263618067824435055592151896825437716740566352876761, 333.6721415503654993082238216993951362402109247917396101880755840484543907332848222466415747543043391),
new_x_y(-305.2385193069950420802010928822220104931543782981037527660195135193679975676060532349856701705644497, 333.6493078713049638533278898256341798843189943869604818415485728166642650456927430930099141453120142),
new_x_y(-303.9199377098485490162204211432040855033226554894203225708484594143335709417643301424354622141726724, 333.5643619724782460319184100085998121837683370088064559853608886651170912028905372315913008922849454),
new_x_y(-302.5906381720937853942182896668133707987262615888750569981819498593871164148724006313283522020974358, 332.4159585850060887548208266988947638235373273665728726694567867040435055451621328655076828775594708),
new_x_y(-301.254826316832559848852423532070399428576113137309776221394836328654255257410758427022326433511996, 332.2030080281581973848691065178763040369482729471826243336896114112865168944739192406479003234683894),
new_x_y(-299.917101429360297413021454717533101122782739535079134905302764773605131986334202322947667100472428, 331.9247112986251437660798751853033843198260717802007641851048126702647516597612569431549190577138538),
new_x_y(-298.5824519246499906009910445213593247058607666413852395386873399470469520728388805757159926210797983, 331.5805960825813974573698867993342653677965225873257136826838218828837521896376585475147274183449187),
new_x_y(-297.2562461484696115881674722258828086536175158679154193175547037198428911509981212696746392263371237, 331.1705533242228507737761845566717585071261111270888099071502102289898701871422081191319922686258971),
new_x_y(-295.9442180152088892180364247059204469829495570521105440866928471989949989744152469429200681297298332, 330.6948739235480308796996178578727182675171799927098151125105501990008631229847515788166026253181649),
new_x_y(-294.6524469953904304256988431153044823206111904691139719835712284827035703293512946157431753905455972, 330.1542850731687288238826612512013368273930902130357061871403419542555348572245916951843159700284224),





#   new_x_y(-337.3687309999671, 297.6956472447914),
#   new_x_y(-337.15758115312997, 299.81011987990416),
#   new_x_y(-336.91836908806914, 301.9215914330887),
#   new_x_y(-336.6418076495959, 304.0284823421357),
#   new_x_y(-336.318683504231, 306.12871865486255),
#   new_x_y(-335.9398960929732, 308.21961161765273),
#   new_x_y(-335.4965100939307, 310.29774031792573),
#   new_x_y(-334.9798238937089, 312.3588389125448),
#   new_x_y(-334.38145639508144, 314.39769046277337),
#   new_x_y(-333.69345423906987, 316.40802995065553),
#   new_x_y(-332.90842117070656, 318.38245965990507),
#   new_x_y(-332.01967080638593, 320.3123807501063),
#   new_x_y(-331.02140344302677, 322.18794551404585),
#   new_x_y(-329.9089067619942, 323.99803545528476),
#   new_x_y(-328.67877930271305, 325.7302709198313),
#   new_x_y(-327.3291743951622, 327.37105851663097),


#   new_x_y(-325.8600608366352, 328.90568291218955),
#   new_x_y(-324.27349497551563, 330.3184497216768),
#   new_x_y(-322.57389703549154, 331.5928860707134),
#   new_x_y(-320.7683225063256, 332.7120048904),
#   new_x_y(-318.86671729112027, 333.6586380505436),
#   new_x_y(-320.55279541015625, 332.7275085449219),
#   new_x_y(-318.68414306640625, 333.4375305175781),
#   new_x_y(-316.74951171875, 333.9408569335937),
#   new_x_y(-314.7717590332031, 334.2315979003906),
#   new_x_y(-312.7741394042969, 334.30633544921875),
#   new_x_y(-310.7801818847656, 334.16412353515625),
#   new_x_y(-308.80279541015625, 333.8643798828125),
#   new_x_y(-307.814697265625, 333.7107238769531),
#   new_x_y(-306.8265686035156, 333.5570678710937),
#   new_x_y(-305.83843994140625, 333.4034423828125),
#   new_x_y(-304.8503112792969, 333.2497863769531),
#   new_x_y(-303.8621826171875, 333.0961303710937),



new_x_y(-305.0, 332.0),
new_x_y(-303.0519228429490050351184927307710268582707193522851168444165350436888015106301205381326780578550155, 331.8638495720849239866131331906456673745836101269849558176241123259255751277175439026817572701912716),
new_x_y(-301.1014723066907676880051936527513607937564176078740768920289802063869834841036732302542834615352922, 331.7636202184595163430651440730133111396548257680826708383050332618895038742243593940261140782065598),
new_x_y(-299.1470951325810546237402207054923488965462559417766962776078348289491343799257968301099721959103482, 331.7352695462665182917854756175805080362202528538442791342799405497328663353743755808164590397337771),
new_x_y(-297.1888822481404815204107523200798237353694144156465537182478147786102365788119003180635253634202901, 331.8147505953563309436539388981220164997047953417956635764756226297183691412656045182817774371579175),
new_x_y(-295.2293965795273433505189026338090713331999973208636827348396698857198447471465859001220112212911753, 332.0378154966976806484813891663764506604405310309004357498011357260449509843273601368005222345716288),
new_x_y(-293.2744961084154086129566313248276843515135376995104292391784978023869563315309477524135670901617532, 332.4395870339810474320779458874334762382415296220447765418179224931869818541505083004184479923561056),
new_x_y(-291.3341312288061457060030717679974609119875133473191453004191321538074979577593384110285442850689952, 333.0538241378060737415521457482605780521237262426394200847998674704413341720942180708928931796215826),
new_x_y(-289.4230790624105697939017290265345523926485314530234226424936352884152980368927065122686534949699774, 333.9118144365179114214152174691989651371468203418697422359512911977596658406938719991592479709965509),
new_x_y(-287.561557556912385953689660147556603133333589066302569935786985215592693388344368945056469553879516, 335.0408409737899864871633743260987769513637694039335731267743590004234802421207957149087128801514782),
new_x_y(-285.7756399969914853113694847699213749239583726166857590125086391782792132577164458122518623739330782, 336.4621942037080361181850311130899452115421710964704568459320221511728748885590937527832175159567258),
new_x_y(-284.0973678989226995504810707576875468956371980757707353818993284962709050199942283832002272483122856, 338.188737652019938471040197980364483221123534952062751345875636005084153571388042394657872854918373),
new_x_y(-282.5644401082118705729971706648261315104402182460083299147640029257363459093437950991994694083209964, 340.2220890904146617564112387016734095584509140945247063498047223651266352905594952769613990632514481),
new_x_y(-281.2193425298703096674152084403271095131870283796825659766868481854577364115192359138529312968071201, 342.5495505093448120604521199137090725716978077345851873794573483583272429055875457507181370945516363),
new_x_y(-280.1077818969998065006359092656480819037396385316703416307440890664389639832540540496546588760732958, 345.1410092812285122609581766833876801188387222745337908373222192997244815938997356378098656382236042),
new_x_y(-279.2763051054592622680812977971117832975900231403883106839469949401387844303349989177856406660547864, 347.9461360053742348723653388354635108067555180818797538459046441292128239661454155832427329247286004),
new_x_y(-278.7690303167804121049535488257814490971923140665590016091060822116589540282486731738638259028662534, 350.8923132052619447222934046157739810753874227063223476135418577909358761750981535337159549112412364),




# new_x_y(-276.0, 350.0),
# new_x_y(-276.3948377239018490706248587560068369596897154859943720476653483625135012434987383047557117915888485, 352.0145000728338911533056629996974358792083255924017885711853651324748734237634038834999318657855715),
# new_x_y(-276.7896754478036981412497175120136739193794309719887440953306967250270024869974766095114235831776969, 354.029000145667782306611325999394871758416651184803577142370730264949746847526807766999863731571143),
# new_x_y(-277.1845131717055472118745762680205108790691464579831161429960450875405037304962149142671353747665454, 356.0435002185016734599169889990923076376249767772053657135560953974246202712902116504997955973567145),
# new_x_y(-277.5793508956073962824994350240273478387588619439774881906613934500540049739949532190228471663553939, 358.0580002913355646132226519987897435168333023696071542847414605298994936950536155339997274631422861),
# new_x_y(-277.9741886195092453531242937800341847984485774299718602383267418125675062174936915237785589579442424, 360.0725003641694557665283149984871793960416279620089428559268256623743671188170194174996593289278576),
# new_x_y(-278.3690263434110944237491525360410217581382929159662322859920901750810074609924298285342707495330908, 362.0870004370033469198339779981846152752499535544107314271121907948492405425804233009995911947134291),
# new_x_y(-278.7638640673129434943740112920478587178280084019606043336574385375945087044911681332899825411219393, 364.1015005098372380731396409978820511544582791468125199982975559273241139663438271844995230604990006),
new_x_y(-279.1587017912147925649988700480546956775177238879549763813227869001080099479899064380456943327107878, 366.1160005826711292264453039975794870336666047392143085694829210597989873901072310679994549262845721),
new_x_y(-279.5535395151166416356237288040615326372074393739493484289881352626215111914886447428014061242996363, 368.1305006555050203797509669972769229128749303316160971406682861922738608138706349514993867920701436),
new_x_y(-279.9483772390184907062485875600683695968971548599437204766534836251350124349873830475571179158884847, 370.1450007283389115330566299969743587920832559240178857118536513247487342376340388349993186578557152),
new_x_y(-280.3432149629203397768734463160752065565868703459380925243188319876485136784861213523128297074773332, 372.1595008011728026863622929966717946712915815164196742830390164572236076613974427184992505236412867),
new_x_y(-280.7380526868221888474983050720820435162765858319324645719841803501620149219848596570685414990661817, 374.1740008740066938396679559963692305504999071088214628542243815896984810851608466019991823894268582),
new_x_y(-281.1328904107240379181231638280888804759663013179268366196495287126755161654835979618242532906550302, 376.1885009468405849929736189960666664297082327012232514254097467221733545089242504854991142552124297),
new_x_y(-281.5277281346258869887480225840957174356560168039212086673148770751890174089823362665799650822438786, 378.2030010196744761462792819957641023089165582936250399965951118546482279326876543689990461209980012),
new_x_y(-281.9225658585277360593728813401025543953457322899155807149802254377025186524810745713356768738327271, 380.2175010925083672995849449954615381881248838860268285677804769871231013564510582524989779867835727),
new_x_y(-282.3174035824295851299977400961093913550354477759099527626455738002160198959798128760913886654215756, 382.2320011653422584528906079951589740673332094784286171389658421195979747802144621359989098525691443),
new_x_y(-282.7122413063314342006225988521162283147251632619043248103109221627295211394785511808471004570104241, 384.2465012381761496061962709948564099465415350708304057101512072520728482039778660194988417183547158),
new_x_y(-283.1070790302332832712474576081230652744148787478986968579762705252430223829772894856028122485992725, 386.2610013110100407595019339945538458257498606632321942813365723845477216277412699029987735841402873),
new_x_y(-283.501916754135132341872316364129902234104594233893068905641618887756523626476027790358524040188121, 388.2755013838439319128075969942512817049581862556339828525219375170225950515046737864987054499258588),
new_x_y(-283.8967544780369814124971751201367391937943097198874409533069672502700248699747660951142358317769695, 390.2900014566778230661132599939487175841665118480357714237073026494974684752680776699986373157114303),
new_x_y(-284.2915922019388304831220338761435761534840252058818130009723156127835261134735043998699476233658179, 392.3045015295117142194189229936461534633748374404375599948926677819723418990314815534985691814970018),
new_x_y(-284.6864299258406795537468926321504131131737406918761850486376639752970273569722427046256594149546664, 394.3190016023456053727245859933435893425831630328393485660780329144472153227948854369985010472825733),
new_x_y(-285.0812676497425286243717513881572500728634561778705570963030123378105286004709810093813712065435149, 396.3335016751794965260302489930410252217914886252411371372633980469220887465582893204984329130681449),

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