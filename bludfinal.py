import streamlit as st
import os
import detectron2
import numpy as np
from PIL import Image
from torch import nn
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

def main():
    st.title("Car Damage Detection")

    # Model file uploader
    st.sidebar.title("Upload Model File")
    model_uploaded_file = st.sidebar.file_uploader("Upload the model file", type=["pth"])

    # File uploader for image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if model_uploaded_file is not None and uploaded_file is not None:
        # Display the uploaded model file
        st.sidebar.write("Model file uploaded:", model_uploaded_file.name)

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Perform detection with the uploaded files
        perform_detection(model_uploaded_file, uploaded_file)

def perform_detection(model_uploaded_file, image_uploaded_file):
    # Perform detection using the uploaded model and image files
    # This function contains the detection logic
    
    # Load the model
    model = load_model(model_uploaded_file)

    # Process the image file
    image = process_image(image_uploaded_file)

    # Run detection
    output_image = inference(model, image)

    # Display the output image
    st.image(output_image, caption="Detected Damage", use_column_width=True)

def load_model(model_uploaded_file):
    # Save the model file to a temporary location
    model_path = os.path.join("temp", model_uploaded_file.name)
    with open(model_path, "wb") as f:
        f.write(model_uploaded_file.getvalue())
    
    # Load the model from the temporary location
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_path

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE='cpu'

    predictor = DefaultPredictor(cfg)
    return predictor

def process_image(image_uploaded_file):
    # Open the image file
    image = Image.open(image_uploaded_file)

    # Convert the image to numpy array
    image = np.array(image)

    return image

def inference(model, image):
    # Run inference
    outputs = model(image)
    out_dict = outputs["instances"].to("cpu").get_fields()
    new_inst = detectron2.structures.Instances((1024,1024))
    new_inst.set('pred_masks',merge_segment(out_dict['pred_masks']))
    v = Visualizer(image[:, :, ::-1],
                   metadata=MetadataCatalog.get("car_dataset_val"), 
                   scale=0.5, 
                   instance_mode=ColorMode.SEGMENTATION
    )
    out_image = v.draw_instance_predictions(new_inst).get_image()[:, :, ::-1]
    return out_image

def merge_segment(pred_segm):
    merge_dict = {}
    for i in range(len(pred_segm)):
        merge_dict[i] = []
        for j in range(i+1,len(pred_segm)):
            if torch.sum(pred_segm[i]*pred_segm[j])>0:
                merge_dict[i].append(j)
    
    to_delete = []
    for key in merge_dict:
        for element in merge_dict[key]:
            to_delete.append(element)
    
    for element in to_delete:
        merge_dict.pop(element,None)
        
    empty_delete = []
    for key in merge_dict:
        if merge_dict[key] == []:
            empty_delete.append(key)
    
    for element in empty_delete:
        merge_dict.pop(element,None)
        
    for key in merge_dict:
        for element in merge_dict[key]:
            pred_segm[key]+=pred_segm[element]
            
    except_elem = list(set(to_delete))
    
    new_indexes = list(range(len(pred_segm)))
    for elem in except_elem:
        new_indexes.remove(elem)
        
    return pred_segm[new_indexes]

if __name__ == "__main__":
    main()
