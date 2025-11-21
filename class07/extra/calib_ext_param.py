
import cv2
import numpy as np
from matplotlib import pyplot as plt




### Setting printing options
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.set_printoptions(precision=3,suppress=True)

##### Helping functions ######

def draw_camera (R,T,axis,length=200,camId=1):

    cam = np.eye(4)
    G = np.hstack((R,T))
    G = np.vstack((G,np.array([0,0,0,1])))
    G_inv = np.linalg.inv(G)
    # Transform the camera frame from the origin to the positon defined from
    # the extrinsic parameters
    cam = G_inv @ cam
    axis = draw_arrows(cam[:,-1],cam[:,0:3],axis,length)
    axis.text(G_inv[0,-1]+10, G_inv[1,-1]+10, G_inv[2,-1]+10, "Camera " + str(camId), color='black')
    return axis 



# Complementary functions for ploting points and vectors with Y-axis swapped with Z-axis
def set_plot(ax=None,figure = None,figsize=(9,8),limx=[-2,2],limy=[-2,2],limz=[-2,2]):
    if figure ==None:
        figure = plt.figure(figsize=(9,8))
    if ax==None:
        ax = plt.axes(projection='3d')
        
    ax.set_xlim(limx)
    ax.set_xlabel("x axis")
    ax.set_ylim(limy)
    ax.set_ylabel("y axis")
    ax.set_zlim(limz)
    ax.set_zlabel("z axis")
    return ax

#adding quivers to the plot
def draw_arrows(point,base,axis,length=1.5):
    # Plot vector of x-axis
    axis.quiver(point[0],point[1],point[2],base[0,0],base[1,0],base[2,0],color='red',pivot='tail',  length=length)
    # Plot vector of y-axis
    axis.quiver(point[0],point[1],point[2],base[0,1],base[1,1],base[2,1],color='green',pivot='tail',  length=length)
    # Plot vector of z-axis
    axis.quiver(point[0],point[1],point[2],base[0,2],base[1,2],base[2,2],color='blue',pivot='tail',  length=length)

    return axis
##################################


##### Main program #########



# Flag to define for which camera will be estimated the extrinsic parameters

CAM = 1


# Load the intrinsic parameters and image for the selected camera
if CAM ==1: 
    calibration_data = 'cam_data/calibration_data_cam01.npz'
    bev_data = 'cam_data/bev_warp01.npz'
    image_file = 'cam_data/img_ext_param01.jpg'
    param_output_file = 'cam01_param.npz'
else:
    calibration_data = 'cam_data/calibration_data_cam02.npz'
    bev_data = 'cam_data/bev_warp02.npz'
    image_file = 'cam_data/img_ext_param02.jpg'
    param_output_file = 'cam02_param.npz'
    

# Load calibration data
data = np.load(calibration_data)
mtx = data["mtx"]
dist = data["dist"]
rvecs = data["rvecs"]
tvecs = data["tvecs"]
newcameramtx = data["newcameramtx"]


# Load Birds Eye View Mapping data
data = np.load(bev_data)
H = data["H"]

# ID, size and dictionary of the aruco associated to the world frame
ID_MARKER = 14
MARKER_LENGTH = 170
DICTIONARY = cv2.aruco.DICT_4X4_50

# Load images for calculating the extrinsic parameters
img = cv2.imread(image_file) 


#Load the dictionary that was used to generate the markers.
#Initialize the detector parameters using default values
parameters =  cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
arucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)


# Detect the markers in both images
corners, ids, rejectedImgPoints = arucoDetector.detectMarkers(img)
        
# Check if any marker was detected and insert the image 
if corners != ():
    # Check if the ID_MARKER was detected in the image
    indices = np.where(ids == ID_MARKER)[0]
    print(ids[indices[0]][0])
    print(indices[0])
    print(corners[indices[0]])
    
    if indices.size > 0:
        print("ID_MARKER found")
        print(ids[indices[0]][0])
        print(indices[0])
        print(corners[indices[0]])
        
        # Draw the marker
        img_corners = cv2.aruco.drawDetectedMarkers(img, [corners[indices[0]]], ids[indices[0]])
        #cv2.namedWindow("Aruco Marker",cv2.WINDOW_NORMAL)
        #cv2.imshow("Aruco Marker", img_corners)
        
        # Estimate the pose of aruco marker (world frame)
        rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[indices[0]], MARKER_LENGTH, mtx, dist)
        R,_ = cv2.Rodrigues(rvec)
        T = tvec
        M = np.hstack((R,T[0].T))
        M = np.vstack((M,np.array([0,0,0,1])))
        print ("Extrinsic parameters (from aruco to camera): \n", M[0:3,0:3], "\n", M[0:3,-1])
        M_inv = np.linalg.inv(M)      
        print ("Extrinsic parameters (from camera to aruco): \n", M_inv[0:3,0:3], "\n", M_inv[0:3,-1])
        
        # Show the Axis of the aruco marker
        img_axis = cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, MARKER_LENGTH * 1.5, 3)
        cv2.namedWindow("Aruco Axis",cv2.WINDOW_NORMAL)
        cv2.imshow("Aruco Axis", img_axis)
        
        # Save calibration results
        np.savez(param_output_file, mtx=mtx, dist=dist, newcameramtx=newcameramtx, R=R, T=T[0].T, Hbev=H)

  
    else:
        print("ID_MARKER not found")

maze_w = 4700
maze_h = 4200        
world = np.eye(4)
axis1 = set_plot(limx=[-maze_w/2,maze_w/2],limy=[-maze_h/2,maze_h/2+300],limz=[-20,4200])
axis1.set_title('World and Camera Frames')
axis1 = draw_camera(M[0:3,0:3],M[0:3,-1].reshape(3,1),axis1,300,CAM)
axis1 = draw_arrows(world[:,-1],world[:,0:3],axis1,500)    
axis1.text(10, 10, 10, "World Frame", color='black')
axis1.plot(np.array([-maze_w,maze_w,maze_w,-maze_w,-maze_w])/2,
           np.array([maze_h,maze_h,-maze_h,-maze_h,maze_h])/2,
           np.zeros((1,5)),linewidth=3)
axis1.set_xlabel('X')
axis1.set_ylabel('Y')
axis1.set_zlabel('Z')
axis1.set_aspect('equal')

        
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.show()
