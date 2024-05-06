import io
import os
import sys
import argparse
import numpy as np

import torch
import hashlib
import pypdfium2
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

import unimernet.tasks as tasks
from unimernet.common.config import Config
from unimernet.processors import load_processor


MAX_WIDTH = 872
MAX_HEIGHT = 1024


class ImageProcessor:
    """ImageProcessor class handles the loading of the model and processing of images."""
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.vis_processor = self.load_model_and_processor()

    def load_model_and_processor(self):
        # Load the model and visual processor from the configuration
        args = argparse.Namespace(cfg_path=self.cfg_path, options=None)
        cfg = Config(args)
        task = tasks.setup_task(cfg)
        model = task.build_model(cfg).to(self.device)
        vis_processor = load_processor(
            "formula_image_eval",
            cfg.config.datasets.formula_rec_eval.vis_processor.eval,
        )
        return model, vis_processor

    def process_single_image(self, pil_image):
        # Process an image and return the LaTeX string
        image = self.vis_processor(pil_image).unsqueeze(0).to(self.device)
        output = self.model.generate({"image": image})
        pred = output["pred_str"][0]
        return pred


@st.cache_data(show_spinner=False)
def read_markdown(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    return data


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=300):
    # Extract an image from a PDF page
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image


@st.cache_data()
def get_uploaded_image(in_file):
    # Load an uploaded image file
    return Image.open(in_file).convert("RGB")


def resize_image(pil_image):
    # Resize an image to fit within the MAX_WIDTH and MAX_HEIGHT
    if pil_image is None:
        return
    pil_image.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)


def display_image_cropped(pil_image, bbox):
    # Display a cropped portion of an image
    cropped_image = pil_image.crop(bbox)
    st.image(cropped_image, use_column_width=True)


@st.cache_data()
def page_count_fn(pdf_file):
    # Return the number of pages in a PDF
    doc = open_pdf(pdf_file)
    return len(doc)


def get_canvas_hash(pil_image):
    return hashlib.md5(pil_image.tobytes()).hexdigest()


@st.cache_data()
def get_image_size(pil_image):
    if pil_image is None:
        return MAX_HEIGHT, MAX_WIDTH
    height, width = pil_image.height, pil_image.width
    return height, width


@st.cache_data(hash_funcs={ImageProcessor: id})
def infer_image(processor, pil_image, bbox):
    # Perform inference on a cropped image
    input_img = pil_image.crop(bbox)
    pred = processor.process_single_image(input_img)
    return pred


@st.cache_resource()
def load_image_processor(cfg_path):
    processor = ImageProcessor(cfg_path)
    return processor


def run_mode1():
    """Direct Recognition mode: recognize formulas directly from an image
    """
    col1, col2 = st.columns([0.5, 0.5])
    in_file = st.sidebar.file_uploader(
        "Input Image:", type=["png", "jpg", "jpeg", "gif", "webp"]
    )
    if in_file is None:
        st.stop()

    filetype = in_file.type
    pil_image = get_uploaded_image(in_file)
    resize_image(pil_image)

    with col1:
        st.image(pil_image, use_column_width=True)
        st.markdown(
            "<h4 style='text-align: center; color: black;'>[Input: Image] </h4>",
            unsafe_allow_html=True,
        )
        bbox_list = [(0, 0, pil_image.width, pil_image.height)]

        with col2:
            inferences = [infer_image(processor, pil_image, bbox) for bbox in bbox_list]
            for idx, (bbox, inference) in enumerate(
                zip(reversed(bbox_list), reversed(inferences))
            ):
                st.latex(inference)
                st.markdown(
                    "<h4 style='text-align: center; color: black;'>[Prediction: Rendered Image]</h4>",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.code(inference)
    st.markdown(
        "<h4 style='text-align: center; color: black;'>[Prediction: LaTeX Code]</h4>",
        unsafe_allow_html=True,
    )


def run_mode2():
    """Manual Selection mode: allows users to select formulas in an image or PDF for recognition.
    """
    col1, col2 = st.columns([0.7, 0.3])
    in_file = st.sidebar.file_uploader(
        "PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"]
    )
    if in_file is None:
        st.stop()

    # Determine if the uploaded file is a PDF or an image
    whole_image = False
    if in_file.type == "application/pdf":
        page_count = page_count_fn(in_file)
        page_number = st.sidebar.number_input(
            "Page number:",
            min_value=1,
            value=1,
            max_value=page_count,
        )
        pil_image = get_page_image(in_file, page_number)
    else:
        pil_image = get_uploaded_image(in_file)
        whole_image = st.sidebar.button("Formula Recognition")
        
    resize_image(pil_image)
    canvas_hash = get_canvas_hash(pil_image) if pil_image else "canvas"

    with col1:
        # Create a canvas component where users can draw rectangles to select formulas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.1)",  # Fixed fill color with some opacity
            stroke_width=1,
            stroke_color="#FFAA00",
            background_color="#FFF",
            background_image=pil_image,
            update_streamlit=True,
            height=get_image_size(pil_image)[0],
            width=get_image_size(pil_image)[1],
            drawing_mode="rect",
            point_display_radius=0,
            key=canvas_hash,
        )

    # Process the drawn rectangles or the whole image if 'whole_image' is True
    if canvas_result.json_data is not None or whole_image:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        bbox_list = []
        if objects.shape[0] > 0:
            boxes = objects[objects["type"] == "rect"][
                ["left", "top", "width", "height"]
            ]
            boxes["right"] = boxes["left"] + boxes["width"]
            boxes["bottom"] = boxes["top"] + boxes["height"]
            bbox_list = boxes[["left", "top", "right", "bottom"]].values.tolist()
        if whole_image:
            bbox_list = [(0, 0, pil_image.width, pil_image.height)]

        if bbox_list:
            with col2:
                # Perform inference on each selected area and display results
                inferences = [infer_image(processor, pil_image, bbox) for bbox in bbox_list]
                for idx, (bbox, inference) in enumerate(zip(reversed(bbox_list), reversed(inferences))):
                    st.markdown(f"### Result {len(inferences) - idx}")
                    st.markdown(
                        "<h6 style='text-align: left; color: black;'>[Input: Image] </h6>",
                        unsafe_allow_html=True,
                    )
                    display_image_cropped(pil_image, bbox) 
                    st.markdown(
                        "<h6 style='text-align: left; color: black;'>[Prediction: Rendered Image] </h6>",
                        unsafe_allow_html=True,
                    )
                    st.latex(inference) 
                    st.markdown(
                        "<h6 style='text-align: left; color: black;'>[Prediction: LaTeX Code] </h6>",
                        unsafe_allow_html=True,
                    )
                    st.code(inference) 
                    st.divider()

    with col2:
        tips = """
        ### Usage tips
        - Draw a box around the equation to get the prediction."""
        st.markdown(tips)


def run_mode3():
    st.markdown("Coming Soon!")


if __name__ == "__main__":

    st.set_page_config(layout="wide")
    html_code = """
    <div style='text-align: center; color: black;'>
        <h2>UniMERNet Online Demo</h2>
        <h5 style='text-align: left; padding-left: 20px; list-style-position: inside;'
        >This App is based on <a href="https://github.com/opendatalab/UniMERNet">UniMERNet</a>. There are three optional modes for mathematical expression recognition:</h5>
        <ul style='text-align: left; padding-left: 20px; list-style-position: inside;'>
            <li><span style="font-weight: bold;">① Direct Recognition:</span> Input an image containing formulas and output the recognition results.</li>
            <li><span style="font-weight: bold;">② Manual Selection:</span> Input a document or webpage screenshot, detect all formulas, then recognize each one.</li>
            <li><span style="font-weight: bold;">③ Auto Detection:</span> Input an image or document, and the model automatically detects and recognizes all formulas.</li>
        </ul>
    </div>
    """
    readme_text = st.markdown(html_code, unsafe_allow_html=True)
    root_path = os.path.abspath(os.getcwd())
    config_path = os.path.join(root_path, "configs/demo.yaml")
    processor = load_image_processor(config_path)

    app_mode = st.sidebar.selectbox(
        "Switch Mode:", ["Direct Recognition", "Manual Selection", "Auto Detection"]
    )

    # Direct Recognition: Input an image containing formulas and output the recognition results.
    if app_mode == "Direct Recognition":
        st.markdown("---")
        st.markdown(
            "<h3 style='text-align: center; color: red;'> Direct Recognition </h3>",
            unsafe_allow_html=True,
        )
        run_mode1()

    # Manual Selection: Input a document or webpage screenshot, detect all formulas, then recognize each one.
    elif app_mode == "Manual Selection":
        st.markdown("---")
        st.markdown(
            "<h3 style='text-align: center; color: red;'> Manual Selection and Recognition </h3>",
            unsafe_allow_html=True,
        )
        run_mode2()
    # Auto Detection: Input an image or document, and the model automatically detects and recognizes all formulas.     
    elif app_mode == "Auto Detection":
        st.markdown("---")
        st.markdown(
            "<h3 style='text-align: center; color: red;'> Auto Detection and Recognition (Coming Soon) </h3>",
            unsafe_allow_html=True,
        )
        run_mode3()