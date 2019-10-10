from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2,pickle
import code,os,struct
import ctypes,_ctypes
import pygame
import numpy as np
import open3d as o3d
from PIL import Image


class DepthRuntime(object):
    def __init__(self):
        pygame.init()
        self.save_flag_color = True
        self.save_flag_depth = True
        self.save_flag_points = True
        self.stream_screen_size = (456,342)
        self.cnt =0 

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect_depth = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

        # back buffer surface for getting Kinect depth frames, 8bit grey, width and height equal to the Kinect color frame size
        self._frame_surface_depth = pygame.Surface((self._kinect_depth.depth_frame_desc.Width, self._kinect_depth.depth_frame_desc.Height), 0, 24)
        # here we will store skeleton data 
        self._bodies = None
        
        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen_depth = pygame.display.set_mode((self._kinect_depth.depth_frame_desc.Width, self._kinect_depth.depth_frame_desc.Height), 
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Point_cloud")

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen_color = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1), 
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect_color = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)


        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface_color = pygame.Surface((self._kinect_color.color_frame_desc.Width, self._kinect_color.color_frame_desc.Height), 0, 32)
        # here we will store skeleton data 




    def draw_color_frame(self, frame, out_path,frame_cnt):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        a = np.reshape(frame,(1080,1920,4))
        a[:,:,0],a[:,:,1],a[:,:,2] = a[:,:,2].copy(),a[:,:,1].copy(),a[:,:,0].copy()
        im =  Image.fromarray(a).convert("RGB")
        im.save(out_path+"color_"+str(frame_cnt) + ".jpg")           

        

    def draw_depth_frame(self, frame, out_path,frame_cnt):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        f8=np.uint8(frame.clip(1,4000)/16.)#clip을 이용해서 최소값을 0 최대 값을 250으로 바꾼다. 
        frame8bit=np.dstack((f8,f8,f8))# RGB값 전부 같은 값으로 바꾼다. 

        a = np.reshape(frame,(424,512))
        b = np.reshape(frame8bit,(424,512,3))

        im =  Image.fromarray(b).convert("RGB")
        im.save(out_path+"/"+"depth_"+str(frame_cnt) + ".jpg")

        with open(out_path+'depth_raw_'+str(frame_cnt)+".pck", 'wb') as f:
            pickle.dump(a, f)

    def stream_color_frame(self, frame, target_surface,screen_size = (304, 228)):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        target_surface.lock()
        address = self._kinect_color.surface_as_array(target_surface.get_buffer())

        a = np.reshape(frame,(1080,1920,4))
        a = cv2.resize(a, dsize=screen_size, interpolation=cv2.INTER_AREA)
        cv2.imshow("color",a)
        cv2.waitKey(1)

    def stream_depth_frame(self, frame, target_surface,screen_size = (304, 228)):
        if frame is None:  # some usb hub do not provide the depth image. it works with Kinect studio though
            return
        target_surface.lock()
        

        f8 = np.uint8((frame / frame.max())*255) 
        #f8=np.uint8(frame.clip(1,4000)/16.)#another depth visualizing option

        frame8bit=np.dstack((f8,f8,f8))     
        depth_raw = np.reshape(frame8bit,(424,512,3))   

        depth = cv2.resize(depth_raw, dsize=screen_size, interpolation=cv2.INTER_AREA)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

        #Drawing black cross line of depth camera focusing area
        x_half,y_half,x_max,y_max = int(screen_size[0]/2),int(screen_size[1]/2),int(screen_size[0]),int(screen_size[1])

        cv2.line(depth,(x_half,0),(x_half,y_max),(0,0,0),1)
        cv2.line(depth,(0,y_half),(x_max,y_half),(0,0,0),1)
        cv2.imshow("depth",depth)
        cv2.waitKey(1)

    def rgb2float( self,r, g, b, a = 0 ):
        return struct.unpack('f', struct.pack('i',r << 16 | g << 8 | b))[0]

    def point_cloud(self,depth_frame,color_frame,Float_RGB_type):
        #I referred some codes from https://github.com/Kinect/PyKinect2/issues/72

        # size must be what the kinect offers as depth frame
        L = depth_frame.size
        # create C-style Type
        TYPE_CameraSpacePoint_Array = PyKinectV2._CameraSpacePoint * L

        # instance of that type
        csps = TYPE_CameraSpacePoint_Array()
        # cstyle pointer to our 1d depth_frame data
        ptr_depth = np.ctypeslib.as_ctypes(depth_frame.flatten())
        # calculate cameraspace coordinates
        error_state = self._kinect_depth._mapper.MapDepthFrameToCameraSpace(L, ptr_depth,L, csps)
        # 0 means everythings ok, otherwise failed!
        if error_state:
            raise "Could not map depth frame to camera space! "
            + str(error_state)

        # convert back from ctype to numpy.ndarray
        pf_csps = ctypes.cast(csps, ctypes.POINTER(ctypes.c_float))
        position_data = np.copy(np.ctypeslib.as_array(pf_csps, shape=(L,3)))
        color_frame = np.reshape(color_frame,(1080,1920,4))


        # Reserve valid points(remove out of range data such as infinites ) 
        color_data = np.zeros(shape=(L,3), dtype=np.int)
        color_position = np.zeros(shape=(L,2), dtype=np.float)

        null_cnt = [] 

        for index in range(L):
            a = self._kinect_depth._mapper.MapCameraPointToColorSpace(csps[index]) 
            color_position[index][0] = a.y;color_position[index][1] = a.x

        x_range = np.logical_and(color_position[:,1] >= 0, color_position[:,1] <=1799.4)
        y_range = np.logical_and(color_position[:,0] >= 0, color_position[:,0] <=1079.4)
        color_pos_range = np.logical_and(x_range,y_range)

        position_data = position_data[color_pos_range]
        color_mapper =np.rint(color_position[color_pos_range]).astype(int)


        #Float_RGB_type=True is for point cloud library(PCL)
        if Float_RGB_type:
            #*coloring with float casting takes too much time >1.7 sec
            color_data = np.asarray([self.rgb2float(color_frame[y][x][2],color_frame[y][x][1],color_frame[y][x][0]) for y,x in color_mapper])
        else:
            color_data = np.asarray([color_frame[y][x][:3] for y,x in color_mapper])

        del pf_csps, csps, ptr_depth, TYPE_CameraSpacePoint_Array
        return position_data,color_data


    def run(self):
        out_path = "temp"
        frame_cnt =0 
        depth_frame = None
        color_frame = None

        # -------- For open3d streaming initialize --------
        #you need to load and show a point cloud before streaming
        pcd = o3d.io.read_point_cloud("temp.ply")
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=960,height=540)
        vis.add_geometry(pcd)

        a  = vis.get_view_control()
        a.rotate(1050,0)

        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'], 
                                                pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)
                elif event.type == pygame.KEYDOWN:
                    self.save_flag_color = True              
                    self.save_flag_depth = True              
                    self.save_flag_points = True
            
            # --- Getting frames and drawing  
            self.cnt = self.cnt + 1;
            if self._kinect_depth.has_new_depth_frame():
                depth_frame = self._kinect_depth.get_last_depth_frame()
                self.stream_depth_frame(depth_frame, self._frame_surface_depth,self.stream_screen_size)
                #depth_frame = None

            if self._kinect_color.has_new_color_frame():
                color_frame = self._kinect_color.get_last_color_frame()
                self.stream_color_frame(color_frame, self._frame_surface_color,self.stream_screen_size)


            if depth_frame is not None and color_frame is not None:
                       
                position_data,color_data = self.point_cloud(depth_frame,color_frame,False)
                pcd.points = o3d.utility.Vector3dVector(position_data)
                pcd.colors = o3d.utility.Vector3dVector(np.flip(color_data.astype(np.float)/255,axis=1))

                object_out_path = out_path + "/"
                ply_out_path = object_out_path + "point_cloud_"+str(frame_cnt) + ".pcd"
                os.makedirs(object_out_path,exist_ok=True)

                #o3d for point streaming
                vis.update_geometry()
                vis.poll_events()
                vis.update_renderer()                
                
                #capturing color depth point_cloud
                if (self.save_flag_points == True):
                    o3d.io.write_point_cloud(ply_out_path, pcd,write_ascii=True)
                if (self.save_flag_depth ==True):
                    self.draw_depth_frame(depth_frame,object_out_path,frame_cnt)
                if (self.save_flag_color == True):
                    vis.capture_screen_image(object_out_path+"point_cloud_%d.jpg" % frame_cnt)
                    self.draw_color_frame(color_frame ,object_out_path,frame_cnt)

                frame_cnt += 1
        
            depth_frame = None
            color_frame = None

                
            # --- Limit to 60 frames per second
            self._clock.tick(60)
            pygame.display.update()
        # Close our Kinect sensor, close the window and quit.
        pygame.quit()


__main__ = "Kinect v2 Point_cloud"
game = DepthRuntime();
game.run();