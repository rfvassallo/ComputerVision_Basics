#Modified from: https://github.com/cjcarver/OpenGL-OpenCV-AR

import cv2
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from threading import Thread
import numpy as np
from objloader import *

print('glutInit' in dir())


class ARRenderer:
    def __init__(self):
        self.INVERSE_MATRIX = np.array([[1.0, 1.0, 1.0, 1.0],
                                        [-1.0, -1.0, -1.0, -1.0],
                                        [-1.0, -1.0, -1.0, -1.0],
                                        [1.0, 1.0, 1.0, 1.0]])
        self.texture_id = 0
        self.thread_quit = False
        self.cap = cv2.VideoCapture(2)
        self.new_frame = self.cap.read()[1]

        # Camera matrix - you may change the calibration matrix according to the camera being used
        self.mtx = np.array([[626.43971938, 0, 325.56339794],
                             [0, 624.36927812, 248.45980208],
                             [0, 0, 1]])
        self.dist = np.array([-0.14875744, 0.54064546, -0.00069566, 0.00267234, -0.54526408])
        self.obj = None


    def init(self):
        video_thread = Thread(target=self.update, args=())
        video_thread.start()


    def init_gl(self, width, height):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(33.7, 1.3, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

        self.load_objects()


    def load_objects(self):
        #Load object
        self.obj = OBJ('models/car/F05ZJJ7S6KBZPJVBC1JSU901Z.obj', swapyz=False)
        glEnable(GL_TEXTURE_2D)
        self.texture_id = glGenTextures(1)


    def track(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            for i in range(len(ids)):
                               
                # Estimate the marker pose               
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 1, self.mtx, self.dist)
                
                # Build the RT matrix
                rmtx = cv2.Rodrigues(rvec)[0]
                tmtx = tvec[0]
                RT = np.hstack((rmtx,tmtx.T))
                RT = np.vstack((RT,np.array([0,0,0,1])))                                      
                view_matrix = RT     
                                   
                # --------------------------------------------------------------------------------------------------------------------------
                # If you want to change the object, you can create transformations and 
                # multiply by the view_matrix





                # --------------------------------------------------------------------------------------------------------------------------
                
                # Prepare the view_matrix to be used by OpenGL
                view_matrix = view_matrix * self.INVERSE_MATRIX
                view_matrix = np.transpose(view_matrix)
                
                
                glPushMatrix()
                glLoadMatrixd(view_matrix)
                glRotate(90, 1, 0, 0)
                #glRotate(90, 0, 1, 0)
                glTranslate(0.5, 1.3, 0.2)
                #glScalef(0.5, 0.5, 0.5)
                self.obj.render()
                glPopMatrix()

        

    def update(self):
        while True:
            self.new_frame = self.cap.read()[1]
            if self.thread_quit:
                break
        #self.cap.release()
        glutDestroyWindow('My and Cube')
        


    def draw_gl_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        glLoadIdentity()
        frame = self.new_frame
        glDisable(GL_DEPTH_TEST)

        self.convert_image_to_texture(frame)
        self.draw_background()
        self.draw_object()

        glEnable(GL_DEPTH_TEST)
        self.track(frame)

        glutSwapBuffers()


    def convert_image_to_texture(self, frame):
        tx_image = cv2.flip(frame, 0)
        tx_image = Image.fromarray(tx_image)
        ix, iy = tx_image.size[0], tx_image.size[1]
        tx_image = tx_image.tobytes('raw', 'BGRX', 0, -1)

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, tx_image)


    def draw_background(self):
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glPushMatrix()
        glTranslatef(0.0, 0.0, -16.0)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-8.0, -6.0, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(8.0, -6.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(8.0, 6.0, 0.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-8.0, 6.0, 0.0)
        glEnd()
        glPopMatrix()


    def draw_object(self):
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glPushMatrix()
        glTranslatef(0.0, 0.0, -16.0)
        glPopMatrix()


    def run(self):
        glutInit(sys.argv)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(1280, 960)   #(1280, 960)(640, 480)
        glutInitWindowPosition(800, 400)
        window = glutCreateWindow('My and Cube')
        glutDisplayFunc(self.draw_gl_scene)
        glutIdleFunc(self.draw_gl_scene)
        glutKeyboardFunc(self.key_pressed)
        self.init_gl (1280, 960) #(1280, 960)(640, 480)
        glutMainLoop()


    def key_pressed(self, key, x, y):
        key = key.decode("utf-8") 
        if key == "q":
            sys.exit(0)
            self.thread_quit = True
            #sys.exit(0)


if __name__ == "__main__":
    ar_renderer = ARRenderer()
    ar_renderer.init()
    ar_renderer.run()
