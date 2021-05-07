# Clone the repo and open set up the model
import streamlit as st
import pandas as pd
import os

'''
# Facial Editing Detector
'''
from subprocess import call
import Model_Evaluator as gcm

st.markdown(' Welcome to our CheapFake Detector interface! The CheapFake Detection Team has configured ' +
            'multiple\n machine learning models to detect whether a facial photo is a CheapFake. Our models currently \n detect facial feature manipulations, ' +
            'skin smoothing, or skintone changes. For a \n model that aims to detect all three, see our Comprehensive model. ')

st.markdown('Here are some examples of each type of CheapFake:')
st.image('facewarp.png')
st.image("skin_smoothing.png")
st.image("skin_tone.png")

st.markdown('To use the interface, simply select a CheapFake characteristic from the drop down selector below' +
            ' and upload an image into the file uploader box. The image will load and a CheapFake ' +
            'probability will be output.')

select = st.selectbox('Select Type of CheapFake', [
                      "Comprehensive", "Facial Feature Editing", "Skin smoothing", "Skintone Changing"])

image = st.file_uploader("Select an image", type=["png", "jpg", "jpeg"])

comprehensive = gcm.load_classifier(
    model_path='models/combined.pth.tar', gpu_id=0)
facewarp = gcm.load_classifier(model_path='models/facewarp.pth.tar', gpu_id=0)
smooth = gcm.load_classifier(model_path='models/smooth.pth.tar', gpu_id=0)
skintone = gcm.load_classifier(model_path='models/skintone.pth.tar', gpu_id=0)


if image is not None:
    st.image(image)

    # try:

    if select == "Comprehensive":
        prob = gcm.classify_fake(image, model=comprehensive)
        edit = "facial editing"
    elif select == "Facial Feature Editing":
        prob = gcm.classify_fake(image, model=facewarp)
        edit = "facial feature editing"
    elif select == "Skin smoothing":
        prob = gcm.classify_fake(image, model=smooth)
        edit = "skin smoothing"
    elif select == "Skintone Changing":
        edit = "skintone editing"
        prob = gcm.classify_fake(image, model=skintone)

    st.write("The probability of " + edit + " is {:.2f}%".format(prob*100))

    # except:
    # st.write("Face not detected. Please upload a photo containing a face.")


# python global_classifier.py edited.jpg --model_path weights/global.pth
