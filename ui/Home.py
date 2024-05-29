import streamlit as st
import pandas as pd
from io import StringIO
from PIL import Image
import numpy as np
import cv2
import sys
from fastsam import FastSAM, FastSAMPrompt 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

def postprocess_results():
    all_masks = st.session_state['everything_results'][0].masks.data
    all_areas = [torch.sum(x).item() for x in all_masks]
    bad_indices = set()
    for i in range(len(all_masks)):
        curr_area = all_areas[i]
        for j in range(i+1, len(all_masks)):
            check_area = all_areas[j]
            intersection = all_masks[i] * all_masks[j]
            proportion = torch.sum(intersection)/curr_area
            if check_area > curr_area and proportion > 0.7:
                bad_indices.add(j)
    keep_mask = torch.ones(all_masks.size(0), dtype=torch.bool)
    keep_mask[list(bad_indices)] = False
    st.session_state['everything_results'][0].masks.data = all_masks[keep_mask]

if 'model' not in st.session_state:
    st.session_state['model'] = FastSAM("/Users/ajaybati/Documents/mimic/mimic/FastSAM-x.pt")

vid_file = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov', 'mkv'])

if vid_file is not None:
    # To read file as bytes:
    # bytes_data = vid.getvalue()
    # st.write(bytes_data)

    # # To convert to a string based IO:
    # stringio = StringIO(vid.getvalue().decode("utf-8"))
    # st.write(stringio)

    # # To read file as string:
    # string_data = stringio.read()
    # st.write(string_data)

    # # Can be used wherever a "file-like" object is accepted:
    # dataframe = pd.read_csv(vid)
    # st.write(dataframe)
    vid = st.video(vid_file)
    
img_file = st.file_uploader("Choose an image", type=['png','jpg','jpeg'])
if img_file is not None:
    image = Image.open(img_file)
    image = image.convert("RGB")
    st.session_state['query_img'] = image #changes as you add more masks
    st.session_state['og_img'] = image #static, does not change
    model = st.session_state.model
    everything_results = model( #generate all masks first
        image,
        device='cpu',
        retina_masks=True,
        imgsz=1024,
        conf=0.4,
        iou=0.9    
    )
    st.session_state['everything_results'] = everything_results
    postprocess_results()
    # st.session_state['query_img'] = img_file
st.write(st.session_state)
