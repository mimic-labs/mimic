import streamlit as st
from streamlit.elements.image import WidthBehaviour
import pandas as pd
from io import StringIO
from PIL import Image
import numpy as np
import cv2
from streamlit_image_coordinates import streamlit_image_coordinates
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


#get mask given point
def getPointPromptAnn(points):
    prompt_process = FastSAMPrompt(st.session_state.query_img, st.session_state.everything_results, device='cpu')
    ann = prompt_process.point_prompt(
            points=points, pointlabel=[1]*len(points)
        )
    return ann

#using mask, generate image with red mask on top
def convertImageGivenAnnotation(annotation):
    msak_sum = 1
    color = np.ones((msak_sum, 1, 1, 3)) * np.array([255,0,0])
    visual = color
    mask_image = np.expand_dims(annotation, -1) * visual
    mask_image = mask_image.squeeze().astype(int)
    if 'mask_image' not in st.session_state:
        st.session_state['mask_image'] = mask_image
    elif 'mask_image' in st.session_state and mask_image.shape != st.session_state.mask_image.shape:
        st.session_state['mask_image'] = mask_image
    else:
        st.session_state.mask_image += mask_image
    end_result = 0.4*st.session_state.mask_image + 0.6*np.asarray(st.session_state.og_img)
    end_result = end_result.astype(np.uint8)
    return end_result

if 'query_img' in st.session_state:
    image = st.session_state['query_img']
    try:
        width, height = image.size
    except:
        height, width, _ = image.shape
    new_width = 690 #WidthBehaviour.AUTO
    ratio = new_width/width
    new_height = int(1.0 * height * ratio)
    st.write(new_width, new_height)
    # imge = st.image(image, use_column_width=True)
    value = streamlit_image_coordinates(image, width=new_width, height = new_height)
    if value is not None:
        point = value["x"], value["y"]
        orig_pointX, origPointY = int(value["x"]*1/ratio), int(value["y"]*1/ratio) #converting to og size coords
        pointPrompt = [[orig_pointX, origPointY]]
        annotation = getPointPromptAnn(pointPrompt)
        end_result = convertImageGivenAnnotation(annotation)
        if 'point' not in st.session_state:
            st.session_state.query_img = end_result #Image.open("/Users/ajaybati/Desktop/ss/Screenshot 2024-05-22 at 8.42.35 PM copy.png")
            st.session_state['point'] = value
            st.rerun()
        #TODO: ADD ANOTHER CONDITION THAT DOES NOT RERENDER IF IT IS THE SAME MASK
        elif value != st.session_state.point:
            st.session_state.query_img = end_result #Image.open("/Users/ajaybati/Desktop/ss/Screenshot 2024-05-22 at 8.42.35 PM copy 2.png")
            st.session_state['point'] = value
            st.rerun()
    st.write(st.session_state)
    img_array = np.array(image)
