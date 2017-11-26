from PythonClient import *
import time
import cv2

def takePhoto(index):
    response = client.simGetImage(0, AirSimImageType.Scene)
    rawImage = np.fromstring(result, np.uint8)
    loc = 'temp/py' + str(index) + '.png'
    AirSimClient.write_file(os.path.normpath(loc), rawImage)
    return response

numOfPhotosTaken = 0

# connect to the AirSim simulator 
client = AirSimClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

#take off and fly somewhere
print('Taking off')
client.takeoff()
print('Flying to pos (0,0,-1) in 5m/s')
client.moveToPosition(0, 0, -1, 5)

while(True):
    
    result = client.simGetImage(0, AirSimImageType.Scene)
    rawImage = np.fromstring(result, np.int8)
    png_img = cv2.imdecode(rawImage, cv2.IMREAD_UNCHANGED)
    cv2.imshow('png', png_img)
    
    #takePhoto(numOfPhotosTaken)
    #numOfPhotosTaken += 1
    key = cv2.waitKey(1) & 0xFF;
    if (key == 27 or key == ord('q') or key == ord('x')):
        break;
    elif (key == ord('p')):
        takePhoto(numOfPhotosTaken)
        numOfPhotosTaken += 1

    loc = client.getPosition()
    print('%.2f,' % loc.x_val, ' %.2f,' % loc.y_val, ' %.2f' % loc.z_val)

print('Landing')
client.land()


