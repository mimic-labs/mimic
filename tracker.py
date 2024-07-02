import sys
base = ""
sys.path.append("/nethome/abati7/flash/Work/mimicopy/mimic/third_party/detectron2/")
sys.path.append("/nethome/abati7/flash/Work/mimicopy/mimic/third_party/Detic/")
sys.path.insert(0, '/nethome/abati7/flash/Work/mimicopy/mimic/third_party/Detic/third_party/CenterNet2/')
sys.path.append("/nethome/abati7/flash/Work/mimicopy/mimic/third_party/Segment-and-Track-Anything/")
sys.path.append("/nethome/abati7/flash/Work/mimicopy/mimic/third_party/Segment-and-Track-Anything/aot/")
sys.path.insert(0, '/nethome/abati7/flash/Work/mimicopy/mimic/third_party/Detic/third_party/Deformable-DETR')

from SegTracker import SegTracker
from model_args import aot_args,sam_args,segtracker_args
from tqdm import tqdm
import os
import cv2
from PIL import Image
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from centernet.config import add_centernet_config
from detic.config import add_detic_config
import mediapipe as mp
# from detic.modeling.meta_arch.d2_deformable_detr import DeformableDetr


class Tracker:
    def __init__(self) -> None:
        sam_args['generator_args'] = {
            'points_per_side': 30,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.9,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 200,
        }
        self.segtracker_args = {
            'sam_gap': 100, # the interval to run sam to segment new objects
            'min_area': 200, # minimal mask area to add a new mask as a new object
            'max_obj_num': 255, # maximal object number to track in a video
            'min_new_obj_iou': 0.8, # the area of a new object in the background should > 80% 
        }
        self.segtracker = SegTracker(self.segtracker_args,sam_args,aot_args)
        self.segtracker.restart_tracker()

        self.frame_masks = None
        self.specificObjects = []
        self.fps = 30
        cfg = self.setup_detic_cfg()
        self.predictor = DefaultPredictor(cfg)

    def setup_detic_cfg(self):
        cfg = get_cfg()
        cfg.MODEL.DEVICE="cuda:0"
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file("/nethome/abati7/flash/Work/mimicopy/mimic/third_party/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
        cfg.merge_from_list(["MODEL.WEIGHTS","/nethome/abati7/flash/Work/mimicopy/mimic/third_party/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"])
        # cfg.merge_from_file("/nethome/abati7/flash/Work/mimic/Detic/configs/BoxSup-DeformDETR_L_R50_4x.yaml")
        # cfg.merge_from_list(["MODEL.WEIGHTS","/nethome/abati7/flash/Work/mimic/Detic/models/BoxSup-DeformDETR_L_R50_4x.pth"])
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.55
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.55
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        cfg.freeze()
        return cfg
    
    def detic_sam_init(self, frame, setBackgroundValue=100):
        predictions = self.predictor(frame) #TODO: define the detic predictor
        boxes = predictions['instances'].pred_boxes.tensor.type(torch.int).tolist()
        for rect in boxes:
            x,y,x1,y1 = rect 
            pred_mask, _ = self.segtracker.seg_acc_bbox(frame, [[x, y], [x1, y1]])
        if 0 in pred_mask:
            pred_mask[pred_mask == 0] = setBackgroundValue #set everything not set to an id
        return pred_mask

    def process(self, video_path, anySpecificObject=True, save=True):
        """
        - should take in video and run tracker
        - anySpecificObject = if we want to track specific objects in the video or not (use sam.segment_with_click for segtracker.add_reference_frame)
        - stores each frame in the self.frame_masks
        - make sure to show/tell the user what the "first frame" is because this is where the ids are initialized
        """
        print("starting processing")
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        self.fps = fps
        self.frame_masks = np.empty((length, height, width))
        sam_gap = self.segtracker_args['sam_gap']

        with torch.cuda.amp.autocast():
            for frame_idx in tqdm(range(length)):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                if frame_idx == 0:
                    if anySpecificObject:
                        self.specificObjects = anySpecificObject
                        pred_mask = self.detic_sam_init(frame)
                    else:
                        pred_mask = self.segtracker.seg(frame)
                    torch.cuda.empty_cache()
                    gc.collect()
                    self.segtracker.add_reference(frame, pred_mask)
                elif (frame_idx % sam_gap) == 0:
                    seg_mask = self.segtracker.seg(frame)
                    torch.cuda.empty_cache()
                    gc.collect()
                    track_mask = self.segtracker.track(frame)
                    # find new objects, and update tracker with new objects
                    new_obj_mask = self.segtracker.find_new_objs(track_mask,seg_mask)
                    pred_mask = track_mask + new_obj_mask

                    self.segtracker.add_reference(frame, pred_mask)
                else:
                    pred_mask = self.segtracker.track(frame,update_memory=True)
                torch.cuda.empty_cache()
                gc.collect()
                
                self.frame_masks[frame_idx] = pred_mask
                
                frame_idx += 1
            cap.release()
            if save:
                with open('frame_masks.npy', 'wb') as f:
                    np.save(f, self.frame_masks)
            print('\nfinished processing')
            
    def getRectangleGivenID(self, mask, id):
        """
        - get rectangle coords given mask (h x w) and id
        """
        currmask = np.argwhere(mask == id)
        start_point = (np.min(currmask[:,1]), np.min(currmask[:,0]))
        end_point = (np.max(currmask[:,1]), np.max(currmask[:,0]))
        return start_point, end_point
        
    
    def getRectangles(self, mask):
        """
        - get rectangle coords for every object given mask
        """
        rects = []
        for id in np.unique(mask):
            rects.append((self.getRectangleGivenID(mask, id), id))
        return rects
    
    def getMaskImg(self, mask):
        """
        - 1 frame of video processing 
        - make sure to draw rectangles on screen based on self.specificObjects
        """
        save_mask = Image.fromarray(mask.astype(np.uint8))
        save_mask = save_mask.convert(mode='P')
        save_mask.putpalette(_palette)
        save_mask = save_mask.convert("RGB")
        
        image = cv2.cvtColor(np.array(save_mask),cv2.COLOR_RGB2BGR)
        for rectstartend, id in self.getRectangles(mask):
            start, end = rectstartend
            color = (255, 0, 0)
            thickness = 10
            image = cv2.rectangle(image, start, end, color, thickness) 
            cv2.putText(image, f'ID: {int(id)}', (start[0], start[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 4)
        return image


    def curateVideo(self, output_path):
        """
        - get video of masks processed with rectangles drawn based on self.specificObjects
        """
        print("curating video of processing")
        height, width = self.frame_masks[0].shape
        fps = self.fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for x in tqdm(range(len(self.frame_masks))):
            image = self.getMaskImg(self.frame_masks[x])
            out.write(image)
        out.release()
        print("finished video")

    def get_binary_masks(self, id_mask):
        """
        get separate binary masks per id in the given mask
        """
        binary_masks = []
        for id in np.unique(id_mask):
            id = int(id)
            binary_masks.append((id_mask == id)*id)
        return np.array(binary_masks)

    def get_frame_point(self, frame, important_hand):
        # Initialize MediaPipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False,
                            max_num_hands=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

        results = hands.process(frame)

        if results.multi_hand_world_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get a constant index for the detected hand (0 or 1). If only 1 hand is detected, default to index = 0.
                hand_idx = hand_handedness.classification[0].index
                hand_label = hand_handedness.classification[0].label

                if important_hand == "R" and hand_label != "Right":
                    continue
                elif important_hand == "L" and hand_label != "Left":
                    continue
                
                palm_points = np.asarray([
                        [hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z],
                        [hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z]])

                # both contact points
                finger_tips = np.copy(palm_points)
                finger_tips[:, 0] *= frame.shape[1]
                finger_tips[:, 1] *= frame.shape[0]
            
                wrist_center_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                wrist_center_y = int(hand_landmarks.landmark[0].y * frame.shape[0])

                return finger_tips, (wrist_center_x, wrist_center_y)

    def get_contour_points(self, mask):
        mask_np = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = np.vstack(contours).squeeze()
        return points

    def find_closest_mask(self, masks, point1, point2):
        min_distance = float('inf')
        min_idx = -1
        closest_mask = None
        
        for i, mask in enumerate(masks):
            id_checker = np.unique(mask)
            if len(id_checker) == 1 and id_checker[0] == 0:
                continue
            contour_points = self.get_contour_points(mask)
            distances_point1 = np.linalg.norm(contour_points - np.array(point1), axis=1)
            distances_point2 = np.linalg.norm(contour_points - np.array(point2), axis=1)
            closest_distances_point1 = np.sort(distances_point1)   
            closest_distances_point2 = np.sort(distances_point2)  
            total_distance = np.mean(closest_distances_point1) + np.mean(closest_distances_point2)
            
            if total_distance < min_distance:
                min_distance = total_distance
                min_idx = i
                closest_mask = mask
                
        return closest_mask, min_idx
        
    def filter_arms_mask_by_points(self, masks, wrist_point):
        idx = -1
        area = float('-inf')
        for i in range(masks.shape[0]):
            if masks[i, wrist_point[0], wrist_point[1]] and np.sum(masks[i] > 0) > area:
                area = np.sum(masks[i] > 0)
                idx = i

        if idx == -1:
            print("No mask that contains wrist point.")
            return masks

        mask = np.ones(masks.shape[0])
        mask[idx] = 0
        filtered_masks = masks[mask == 1]
        return filtered_masks

    def getTouchedObjectIDGivenHand(self, frameNum, frame, label, backtrack=30):
        """
        - locationOfTouch - x,y point?
        - given a frame number (this should correspond to some mask in self.frame_masks) and touch location (x,y)
        find which mask it corresponds to (this should be done using arsh's function of contours or whtvr)
        - get object id from the mask

        Returns:
            first ish frame (this can be modified to whatever frame we need), id of the object we're looking for
        """
        id_mask = self.frame_masks[frameNum,:,:]
        finger_tips, wrist_point = self.get_frame_point(frame, label)
        masks = self.get_binary_masks(id_mask)
        filtered_masks = self.filter_arms_mask_by_points(masks, wrist_point[::-1])
        
        # find closest mask
        thumb_pt = [int(finger_tips[0, 0]), int(finger_tips[0, 1])]
        index_pt = [int(finger_tips[1, 0]), int(finger_tips[1, 1])]

        closest_mask, closest_idx = self.find_closest_mask(filtered_masks, thumb_pt, index_pt)
        unique_numbers = np.unique(closest_mask)
        id = unique_numbers[unique_numbers != 0][0]

        return self.frame_masks[0+backtrack, :, :], id



tracker = Tracker()
video_name = 'IMG_3288'

# CHANGE DIRECTORIES
io_args = {
    'input_video': f'{video_name}.MOV',
    'output_mask_dir': f'./assets/{video_name}_masks_vith', # save pred masks
    'output_mask_plain_dir':f'./assets/{video_name}_plain_masks_vith',
    'output_video': f'{video_name}_seg_vith.mp4', # mask+frame vizualization, mp4 or avi, else the same as input video
    'output_gif': f'./assets/{video_name}_seg_vith.gif', # mask visualization
}
frames = [(110, "R"), (166, "R"), (206, "L"), (270, "L"), (510, "R")]
frameNum, label = frames[2]
cap = cv2.VideoCapture(io_args['input_video'])
cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
ret, frame = cap.read()
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

tracker.process(io_args['input_video'], save=False)

# In tracker.process, if save is true, it would've saved a 3d np array of all 
# the calculated frame masks and stored in frame_masks.npy
# with open('frame_masks.npy', 'rb') as f:
#     tracker.frame_masks = np.load(f)

tracker.curateVideo(io_args['output_video'])
print(tracker.getTouchedObjectIDGivenHand(frameNum, frame, label))






