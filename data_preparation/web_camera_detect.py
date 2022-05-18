import cv2
import torch
from time import time
import numpy as np
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", type=str, required=True,
                help="path to the video file")
ap.add_argument("-f", "--folder", type=str, default="detection",
                help="name of folder to save images")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
               help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

class FindBirds:
    """
    Finds birds from video using YOLOv5s and saves images
    """
    def __init__(self, source, out_dir = None,
                conf_lvl = 0.6, birds_label = 14):
        """
        """
        assert conf_lvl<1, f'Confidence level {conf_lvl} should be less than 1'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.source = source
        self.out_dir = out_dir
        self.conf_lvl = conf_lvl
        self.model = self.load_model()
        self.birds_label = birds_label
        
    def check_directory(self):
        """
        Check existance of folder, and create it if it is't exist
        :returns: void
        """
        directory = os.path.isdir(self.out_dir)
        if not directory:
            os.makedirs(self.out_dir)
            print("created folder : ", self.out_dir)
        else:
            print(directory, "folder already exists.")
            
    def get_video_from_source(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :returns: opencv2 video capture object
        """
        try:
            player = cv2.VideoCapture(self.source)
            assert player.isOpened()
        except AssertionError:
            print("Camera is not available")
            quit()
        return player #cv2.VideoCapture(self.source)
    
    def load_model(self):
        """
        load YOLOv5s model
               
        :returns: Pytorch model with weights.
        """
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.to(self.device)
        return model
    
    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame .
        :param frame: input frame in numpy/list/tuple format.
        :returns: Labels and  boxes of objects detected by model in the frame.
        """
                
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        with torch.no_grad():
            preds = self.model(frame)
        torch.cuda.empty_cache()
        
        return preds
    
    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :returns: void
        """
        self.check_directory()
        
        player = self.get_video_from_source()
        
        
        i = 0
        while True:
            
            start_time = time()
            try:
                ret, frame = player.read()
                assert ret
            except AssertionError:
                player = self.get_video_from_source()
            
            if i%1==0:
                results = self.score_frame(frame)
                
                detections = [detect[-2]>self.conf_lvl and detect[-1]==self.birds_label for detect in results.xyxy[0]] 
                if any(detections):
                    cv2.imwrite(os.path.join(self.out_dir, str(round(time())) +"_frame%d.jpg" % i), frame)
                
                end_time = time()
                fps = 1/np.round(end_time - start_time, 3)
                #if i%1000==0:
                    #print(f"Frames Per Second : {fps}")
            
            i+=1
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                player.release()
                cv2.destroyAllWindows()
                break
       
if __name__ == "__main__":
    searcher = FindBirds(source=args['source'],
                     out_dir=args['folder'],
                     conf_lvl=args['confidence'])
    searcher()     
    
        
