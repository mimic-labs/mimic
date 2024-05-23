import streamlit as st
import pandas as pd
from io import StringIO

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
    img = st.image(img_file)