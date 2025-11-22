import numpy as np
import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d 
import pathlib
import matplotlib
matplotlib.use("TkAgg")



def calibrate_charuco(dirpath, image_format, x_squares,y_squares, square_length, marker_length,SHOW=True):
    '''Apply camera calibration using aruco.
    The dimensions are in cm.
    '''
    criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, .001)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_100)
    board = aruco.CharucoBoard((x_squares, y_squares), square_length, marker_length, aruco_dict)

    #aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    #board = aruco.CharucoBoard_create(x_squares, y_squares, square_length, marker_length, aruco_dict)
    arucoParams = aruco.DetectorParameters()
    arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR 

    counter, corners_list, id_list = [], [], []
    img_dir = pathlib.Path(dirpath)
    first = 0
    # Find the ArUco markers inside each image
    for img in img_dir.glob(f'*{image_format}'):
        print(f'using image {img}')
        image = cv2.imread(str(img))
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imsize = img_gray.shape
        
        
        # Verify if the image has landscape format. If not, rotate image
        if (imsize[1] < imsize[0]):
            img_gray = cv2.rotate(img_gray,cv2.ROTATE_90_CLOCKWISE)
            image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
            imsize = img_gray.shape
           
        print('Image size: ', imsize[1],' x ',imsize[0])
        corners, ids, rejected = aruco.detectMarkers(
            img_gray, 
            aruco_dict, 
            parameters=arucoParams
        )
       
        print(len(corners))
        
        image_with_markers = aruco.drawDetectedMarkers(image, corners, ids)
                
        for corner in corners:
           cv2.cornerSubPix(img_gray, corner, (3, 3), (-1, -1), criteria)
           
        #print(len(corners))  
        
        
        resp, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img_gray,
            board=board
        )
        
        print (resp)
        
        #print(charuco_corners.shape)
        # If a Charuco board was found, let's collect image/corner points
        # Requiring at least 20 squares
        
        if resp > 0 and charuco_corners is not None:
            
            image_with_markers = aruco.drawDetectedCornersCharuco(image_with_markers, charuco_corners, charuco_ids, (0,0,255))
            if SHOW:
            	cv2.imshow('Markers and corners',image_with_markers)
            	cv2.waitKey(100)
            # Add these corners and ids to our calibration arrays
            print (charuco_corners.shape)
            corners_list.append(charuco_corners)
            id_list.append(charuco_ids)

    # Actual calibration
    
    #distCoeffsInit = np.zeros((5,1))
    #flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    
    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
         charucoCorners=corners_list, 
         charucoIds=id_list, 
         board=board, 
         imageSize=img_gray.shape, 
         cameraMatrix=None, 
         distCoeffs=None)
   
    #cv2.destroyAllWindows()
    return [ret, mtx, dist, rvecs, tvecs]

########################################################
# Complementary functions for ploting points and vectors with Y-axis swapped with Z-axis
def set_plot(ax=None,figure = None,figsize=(9,8),limx=[-2,2],limy=[-2,2],limz=[-2,2]):
    if figure ==None:
        figure = plt.figure(figsize=(9,8))
    if ax==None:
        ax = plt.axes(projection='3d')
    
    ax.set_title("Camera Calibration")
    ax.set_xlim(limx)
    ax.set_xlabel("x axis")
    ax.set_ylim(limy)
    ax.set_ylabel("y axis")
    ax.set_zlim(limz) 
    ax.set_zlabel("z axis")
    return ax
  
#adding quivers to the plot
def draw_arrows(point,base,axis,length=0.5):
    # Plot vector of x-axis
    axis.quiver(point[0],point[1],point[2],base[0,0],base[1,0],base[2,0],color='red',pivot='tail',  length=length)
    # Plot vector of y-axis
    axis.quiver(point[0],point[1],point[2],base[0,1],base[1,1],base[2,1],color='green',pivot='tail',  length=length)
    # Plot vector of z-axis
    axis.quiver(point[0],point[1],point[2],base[0,2],base[1,2],base[2,2],color='blue',pivot='tail',  length=length)
    
    return axis
#########################################################

###############################
# MAIN 
###############################
# Parameters
IMAGES_DIR =  './canon_charuco/' #  './imagens_calibracao_charuco/resized/' 
IMAGES_FORMAT = 'JPG'
#Number os chessboard squares
X_SQUARES = 9
Y_SQUARES = 7
# Dimensions in cm
MARKER_LENGTH = 3.3
SQUARE_LENGTH = 4.5
# Showing option
SHOW = True


# Calibrate 
ret, mtx, dist, rvecs, tvecs = calibrate_charuco(
    IMAGES_DIR, 
    IMAGES_FORMAT,
    X_SQUARES,
    Y_SQUARES,
    SQUARE_LENGTH,
    MARKER_LENGTH,
    SHOW
)

# Print the number of images where the corners could be detected
print('Number of images where the corners were detected', len(rvecs))

print('Calibration Matrix \n', mtx)
print('Radil distortion coeficients \n', dist)
print('Ret \n', ret)

# Organize the extrinsic parameters (rotation and translation) in 3xN arrays, where N is the number of images 
transl = np.hstack(tvecs)
rot = np.hstack(rvecs)

if (0):
    # Load coefficients
    original = cv2.imread(IMAGES_DIR + 'IMG_1580.' + IMAGES_FORMAT)

    dst = cv2.undistort(original, mtx, dist, None, newcameramtx)
    #dst = cv2.undistort(original, mtx, dist, None, mtx)

    cv2.imshow('original img',original)
    cv2.imshow('undistorted img',dst)
    

   # When using Colab, we can not use cv2.imshow. So we are showing images 
   # with matplotlib.pyplot       
   #fig = plt.figure(figsize=(7,7))
   #ax = fig.add_subplot(1, 1, 1)
   #ax.set_title("Original Image")
   #plt.imshow(original) 
   #fig = plt.figure(figsize=(7,7))
   #ax = fig.add_subplot(1, 1, 1)
   #ax.set_title("Undistorted Image")
   #plt.imshow(dst) 


cv2.waitKey(0)
cv2.destroyAllWindows()



"""# Show Extrinsic Parameters
## Considering the calibration pattern fixed at the origin and moving the camera.
"""

# Initialize figure
axis0 = set_plot(limx=[-80,80],limy=[-80,80],limz=[-200,20])
axis0.set_title('Camera Calibration - Fixed Board / Moving Camera')
# Create base vector values
e1 = np.array([[1],[0],[0],[0]]) # X
e2 = np.array([[0],[1],[0],[0]]) # Y
e3 = np.array([[0],[0],[1],[0]]) # Z
base = np.hstack((e1,e2,e3))
#origin point
origin =np.array([[0],[0],[0],[1]])
# Create camera frame
cam  = np.hstack([base,origin])

# Prepare 3D object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((Y_SQUARES*X_SQUARES,3), np.float32)
objp[:,:2] = np.mgrid[0:X_SQUARES*SQUARE_LENGTH:SQUARE_LENGTH,0:Y_SQUARES*SQUARE_LENGTH:SQUARE_LENGTH].T.reshape(-1,2)

# Plot camera frames considering that the calibration patern is on the XY-plane and Z=0
for i in range(rot.shape[1]):  #

    R,_ = cv2.Rodrigues(rot[:,i])
    t = transl[:,i]
    Rt = np.eye(4)
    Rt[0:3,0:3] = R
    Rt[0:3,-1] = t
    # To get the camera's rotation and translation we have to invert the
    # Extrinsic Parameters
    M = np.linalg.inv(Rt)
    # Transform the camera frame from the origin to the positon defined from
    # the extrinsic parameters
    new_cam = M@cam
    axis0 = draw_arrows(new_cam[:,-1],new_cam[:,0:3],axis0,10)

# Plot the calibration pattern as it was fixed and all the
# relative positions of the camera
X,Y = np.meshgrid(objp[:,0],objp[:,1])
Z = np.zeros(X.shape)


axis0.plot_wireframe(X,Y,Z)
axis0.view_init(elev=-60,azim=-111,roll=23)
axis0.set_aspect('equal')



"""##Considering the camera fixed at the origin and moving the calibration pattern"""

# Initialize figure
axis1 = set_plot(limx=[-80,80],limy=[-80,80],limz=[-20,200])
axis1.set_title('Camera Calibration - Fixed Camera / Moving Board')
# Create base vector values
e1 = np.array([[1],[0],[0],[0]]) # X
e2 = np.array([[0],[1],[0],[0]]) # Y
e3 = np.array([[0],[0],[1],[0]]) # Z
base = np.hstack((e1,e2,e3))
#origin point
origin =np.array([[0],[0],[0],[1]])
# Create camera frame
cam  = np.hstack([base,origin])
axis1 = draw_arrows(cam[:,-1],cam[:,0:3],axis1,10)


calib_points = objp.T
#add a vector of ones to the chessboard points to represent them in homogeneous coordinates
calib_points = np.vstack([calib_points, np.ones(np.size(calib_points,1))])



# Plot calibration patter considering the camera fixed at (0,0,0)
for i in range(rot.shape[1]):

    # Set up the transformation matrix
    R,_ = cv2.Rodrigues(rot[:,i])
    t = transl[:,i]
    Rt = np.eye(4)
    Rt[0:3,0:3] = R
    Rt[0:3,-1] = t

    chessboard = Rt@calib_points
    axis1.scatter(chessboard[0,:],chessboard[1,:],chessboard[2,:])


axis1.view_init(elev=-60,azim=-142,roll=53)
axis0.set_aspect('equal')

plt.show()





