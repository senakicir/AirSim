from PythonClient import *

class State(object):
    def __init__(self, positions_):
        self.positions = positions_
        #shoulder_vector = positions_[R_SHOULDER_IND, :] - positions_[L_SHOULDER_IND, :]
        #self.human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])
        #self.human_rotation_speed = 0
        self.human_pos = positions_[HUMAN_POS_IND,:]
        self.human_vel = 0
        self.human_speed = 0
        self.drone_pos = np.array([0,0,0])
        self.current_polar_pos = np.array([0,0,0])
        self.current_degree = 0
        self.drone_orientation = np.array([0,0,0])
        projected_distance_vect = positions_[HUMAN_POS_IND, :]

        global radius
        radius =  np.linalg.norm(projected_distance_vect[0:2,]) #to do

        drone_polar_pos = - positions_[HUMAN_POS_IND, :] #find the drone initial angle (needed for trackbar)
        global some_angle
        some_angle = RangeAngle(np.arctan2(drone_polar_pos[1], drone_polar_pos[0]), 360, True)
    
    def updateState(self, positions_):
        self.positions = positions_
        
        #get human position, delta human position, human drone_speedcity
        prev_human_pos = self.human_pos
        self.human_pos = self.positions[HUMAN_POS_IND,:]
        delta_human_pos = self.human_pos - prev_human_pos #how much the human moved in one iteration
        self.human_vel = delta_human_pos/DELTA_T #the velocity of the human (vector)
        self.human_speed = np.linalg.norm(self.human_vel) #the speed of the human (scalar)
        
        #what angle and polar position is the drone at currently
        self.drone_pos = positions_[DRONE_POS_IND, :] #airsim gives us the drone coordinates with initial drone loc. as origin
        self.drone_orientation = positions_[DRONE_ORIENTATION_IND, :]
        self.current_polar_pos = (self.drone_pos - self.human_pos)     #subtrack the human_pos in order to find the current polar position vector.
        self.current_degree = np.arctan2(self.current_polar_pos[1], self.current_polar_pos[0]) #NOT relative to initial human angle, not using currently
        print(self.current_degree)
        #calculate human orientation
        #shoulder_vector = positions_[R_SHOULDER_IND, :] - positions_[L_SHOULDER_IND, :]
        #prev_human_orientation = self.human_orientation
        #a filter to eliminate noisy data (smoother movement)
        #self.human_orientation = np.arctan2(-shoulder_vector[0], shoulder_vector[1])*BETA + prev_human_orientation*(1-BETA)
        #self.human_rotation_speed = (self.human_orientation-prev_human_orientation)/DELTA_T

    def getDesiredPosAndAngle(state):
        desired_polar_angle = self.current_degree + INCREMENT_DEGREE_AMOUNT*(np.linalg.norm(self.human_vel)/MAX_HUMAN_SPEED)
        desired_polar_pos = np.array([cos(desired_polar_angle) * radius, sin(desired_polar_angle) * radius, 0])
        desired_pos = desired_polar_pos + self.human_pos + TIME_HORIZON*self.human_vel - np.array([0,0,z_pos])
        desired_yaw = desired_polar_angle - pi
        return desired_pos, desired_yaw

    def getDesiredPosAndAngleTrackbar(state):
        #calculate new polar coordinates according to circular motion (the circular offset required to rotate around human)
        input_rad = radians(cv2.getTrackbarPos('Angle', 'Angle Control')) #according to what degree we want the drone to be at
        #input_rad_unreal_orient = input_rad + INITIAL_HUMAN_ORIENTATION #we don't use this at all currently
        #desired_polar_angle = state.human_orientation + input_rad + state.human_rotation_speed*TIME_HORIZON
        desired_polar_angle = input_rad

        desired_polar_pos = np.array([cos(desired_polar_angle) * radius, sin(desired_polar_angle) * radius, 0])
        desired_pos = desired_polar_pos + self.human_pos + TIME_HORIZON*self.human_vel - np.array([0,0,z_pos])
        desired_yaw = desired_polar_angle - pi
        return desired_pos, desired_yaw


def SaveBonePositions2(index, bones, f_output):
    bones = [ v for v in bones.values() ]
    line = str(index)
    print(len(bones), 'lol')
    for i in range(0, len(bones)):
        line = line+'\t'+str(bones[i][b'x_val'])+'\t'+str(bones[i][b'y_val'])+'\t'+str(bones[i][b'z_val'])
    line = line+'\n'
    f_output.write(line)

def doNothing(x):
    pass

def RangeAngle(angle, limit=360, is_radians = True):
    if is_radians == True:
        angle = degrees(angle)
    if angle > limit:
        angle = angle - 360
    elif angle < limit-360:
        angle = angle + 360
    if is_radians == True:
        angle = radians(angle)
    return angle

def TakePhoto(client, index):
    response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene)])
    response = response[0]
    bone_pos = response.bones
    loc = 'temp_main/img_' + str(index) + '.png'
    AirSimClient.write_file(os.path.normpath(loc), response.image_data_uint8)
    SaveBonePositions2(index, bone_pos, f_bones)
    return response.image_data_uint8