import numpy as np
import cv2 
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


### Setting printing options
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
np.set_printoptions(precision=3,suppress=True)


###########  Helper functions ##################

# Draw camera reference frame according to its extrinsic parameters R and T

def draw_camera (R,T,axis,length=200,camId=1):
    # Create a camera frame
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

#Triangulation Functions 

def triangulate_point(ptsl,ptsr,Pl,Pr):
    """ Point pair triangulation from
    least squares solution. """
    M = np.zeros((6,6))
    M[:3,:4] = Pl
    M[3:,:4] = Pr
    M[:3,4] = -ptsl
    M[3:,5] = -ptsr
    U,S,V = np.linalg.svd(M)
    X = V[-1,:4]
    Y = V[-1]
    
    return X / X[3]	
	
def triangulate(ptsl,ptsr,Pl,Pr):
    """ Two-view triangulation of points in
    x1,x2 (3*n homog. coordinates). """
    
    n = ptsl.shape[1]
    if ptsr.shape[1] != n:
      raise ValueError("Number of points don't match.")
    
    X = [ triangulate_point(ptsl[:,i],ptsr[:,i],Pl,Pr) for i in range(n)]
    
    return np.array(X).T

# Function to sort the aruco markers that were detected on the frame
def sort_markers (corners, ids): 

    if ids is not None:
        # Combine IDs and corners into a list of tuples
        markers = list(zip(ids.flatten(), corners))

        # Sort by ID
        markers.sort(key=lambda x: x[0])

        # Unzip back into sorted arrays
        sorted_ids, sorted_corners = zip(*markers)

        # Optionally convert back to numpy arrays
        sorted_ids = np.array(sorted_ids).reshape(-1, 1)
        sorted_corners = list(sorted_corners)
       
        sorted_centers = []
        for corners in sorted_corners:
            center = np.mean(corners[0],axis=0)
            sorted_centers.append(center)
           
        sorted_centers = np.array(sorted_centers)  
        
    return sorted_ids, sorted_centers, sorted_corners  

###################################################

####### Read the cameras parameters

# Camera 01 paramters
calibration_data = 'cam_data/cam01_param.npz'

# Load calibration data
data = np.load(calibration_data)
mtx1 = data["mtx"]
dist1 = data["dist"]
R1 = data["R"]
T1 = data["T"]
newcameramtx1 = data["newcameramtx"]

M1 = newcameramtx1 @ np.hstack((R1,T1)) 

print ("Camera 01 Parameters")
print ("Original Intrinsic Matrix\n", mtx1)
print ("Intrinsic Matrix adjusted to undistorted images\n", newcameramtx1)
print ("Extrinsic Matrix\n", np.hstack((R1,T1)))

print("Pinhole Projection Matrix - CAM01\n", M1)


# Camera 02 paramters
calibration_data = 'cam_data/cam02_param.npz'

# Load calibration data
data = np.load(calibration_data)
mtx2 = data["mtx"]
dist2 = data["dist"]
R2 = data["R"]
T2 = data["T"]
newcameramtx2 = data["newcameramtx"]

M2 = newcameramtx2 @ np.hstack((R2,T2)) 

print ("Camera 02 Parameters")
print ("Original Intrinsic Matrix\n", mtx2)
print ("Intrinsic Matrix adjusted to undistorted images\n", newcameramtx2)
print ("Extrinsic Matrix\n", np.hstack((R2,T2)))

print("Pinhole Projection Matrix - CAM02\n", M2)

#########  Load Aruco Dictionary #########
    
#Load the dictionary that was used to generate the markers.
#Initialize the detector parameters using default values

parameters =  cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)
   
   

    
######## Read videos from camera01 and camera02  ##########

vid_list = [cv2.VideoCapture("cam_data/cam01_sync.mp4"),
            cv2.VideoCapture("cam_data/cam02_sync.mp4")
            ]
# Organize cameras data            
mtx_list = [mtx1,mtx2]
dist_list = [dist1,dist2]
newcameramtx_list = [newcameramtx1,newcameramtx2]

# Set 3D plot to show the reconstruction
maze_w = 4700
maze_h = 4200
axis01 = set_plot(limx=[-maze_w/2,maze_w/2],limy=[-maze_h/2,maze_h/2+300],limz=[-20,4200])
axis01.set_title('3D Reconstruction')

# Create camera frame and world frame
world = np.eye(4)
    

# Define the aruco marker of the robot
ROBOT_ID = 14


# Main loop

pts_robot = []  
count = 0

while True:
    
    # Read the frames from both video files
    _, frame1 = vid_list[0].read()
    _, frame2 = vid_list[1].read()
    
    count = count +1
    
    if frame1 is None:
        print("Empty Frame, done!")
        break
    elif frame2 is None:
        print("Empty Frame, done!")
        break
    
    
    # Only perform the reconstruction and show the images 
    # and result every 5 frames to avoid slowdowns
    if count == 5:
    
        count = 0
	
        # undistort images
        img1 = cv2.undistort(frame1, mtx_list[0], dist_list[0], None, newcameramtx_list[0])
        img2 = cv2.undistort(frame2, mtx_list[1], dist_list[1], None, newcameramtx_list[1])

        # Detect the markers in both images
        corners1, ids1, _ = arucoDetector.detectMarkers(img1)
        corners2, ids2, _ = arucoDetector.detectMarkers(img2)
        
        # Sort the markers
        sorted_ids1, sorted_centers1, sorted_corners1 = sort_markers (corners1, ids1)
        sorted_ids2, sorted_centers2, sorted_corners2 = sort_markers (corners2, ids2)
        
        # Draw detected markers and their centers
        cv2.aruco.drawDetectedMarkers(img1, sorted_corners1, sorted_ids1)
        cv2.aruco.drawDetectedMarkers(img2, sorted_corners2, sorted_ids2)
           
        for c in sorted_centers1:
            cv2.circle(img1, tuple(c.astype(int)), 5, (0, 0, 255), -1) # filled green circles
        for c in sorted_centers2:
            cv2.circle(img2, tuple(c.astype(int)), 5, (0, 0, 255), -1) # filled green circles
        
        # Show both frames with the detections    
        current_frame = np.vstack((img1, img2))
        cv2.imshow('Both Cameras', cv2.resize(current_frame, (480, 540)) )
	
	# Match the IDs of the markers found on both images
	# and also check if the robot was detected
        ids1 = sorted_ids1.flatten()
        ids2 = sorted_ids2.flatten()
    
        pts1 = []
        pts2 = []
        
        rob_index = None
        
        # Use the smaller vector as reference to look for the matches 
        # among the aruco detections 
        if (ids1.size < ids2.size):
            
            for i in range(ids1.size):
                if ids1[i] in ids2:
                    pts1.append(sorted_centers1[i])
                    idx = np.where(ids2 == ids1[i])[0][0]
                    pts2.append(sorted_centers2[idx])
                if ids1[i]==ROBOT_ID:
                    rob_index = i
        else:
        
            for i in range(ids2.size):
                if ids2[i] in ids1:
                    pts2.append(sorted_centers2[i])
                    idx = np.where(ids1 == ids2[i])[0][0]
                    pts1.append(sorted_centers1[idx])
                if ids2[i]==ROBOT_ID:
                    rob_index = i
	
	# Prepare the points select on the images 
	# from cam01 and cam02 to perform triangulation    
	# Write the points on homogeneous coordinates   
        pts1 = np.array(pts1)
        pts1 = np.vstack((pts1.T,np.ones((1,pts1.shape[0]))))
        pts2 = np.array(pts2)
        pts2 = np.vstack((pts2.T,np.ones((1,pts2.shape[0]))))
        
        # Perform the reconstruction through triangulation
        P3D = triangulate(pts1,pts2,M1,M2)
        
        print('Reconstructed points\n', P3D)
        
        # if the robot is seen, store its position to draw its path 
        if rob_index is not None:
            pts_robot.append(P3D[:,rob_index])
            pts_robot_array = np.array(pts_robot)
    
        
        # Clear previous plot
        axis01.cla()
        
        # Plot the reconstructed points
        axis01.scatter(P3D[0], P3D[1], P3D[2], c='red', s=100, label='Current Position')
        # Plot robot's path
        axis01.plot(pts_robot_array[:, 0], pts_robot_array[:, 1], pts_robot_array[:, 2], 'b-')
        axis01.text(pts_robot_array[-1, 0]+35, pts_robot_array[-1, 1]+35,
                   pts_robot_array[-1, 2]+35, "Robot", color='blue') 
        # Plot the cameras and world frame
        axis01 = draw_camera(R1,T1,axis01,300,camId=1)
        axis01 = draw_camera(R2,T2,axis01,300,camId=2)
        axis01 = draw_arrows(world[:,-1],world[:,0:3],axis01,500) 
        axis01.text(10, 10, 10, "World Frame", color='black')   
        
        # Plot the borders of the maze
        axis01.plot(np.array([-maze_w,maze_w,maze_w,-maze_w,-maze_w])/2,
                    np.array([maze_h,maze_h,-maze_h,-maze_h,maze_h])/2,
                    np.zeros((1,5)),linewidth=3)
    
        axis01.set_xlabel('X')
        axis01.set_ylabel('Y')
        axis01.set_zlabel('Z')
        axis01.set_title('3D Reconstruction')
    
        plt.pause(0.001)
        
        
        if cv2.waitKey(1) == ord('q'):
    	    break
    
cv2.destroyAllWindows() 
plt.show()


