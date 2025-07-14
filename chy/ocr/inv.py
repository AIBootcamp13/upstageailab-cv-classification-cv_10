import os
import random
import io
import numpy as np

import ipywidgets as widgets
from ipywidgets import VBox, HBox, Button, Output

from IPython.display import display, clear_output, Image as IImage
from PIL import Image, ImageOps


def resize_image_with_ratio(image_path, ratio=1.0):
    image = Image.open(image_path)
    width, height = image.size

    new_width = int(width * ratio)
    new_height = int(height * ratio)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image


def write_list(file_path, items):
    with open(file_path, 'w') as f:
        for item in items:
            f.write(str(item) + '\n')

def pil_to_ipywidget(img):
    if type(img) is np.ndarray:
        img = Image.fromarray(img)
    with io.BytesIO() as output:
        img.save(output, format="PNG")
        return widgets.Image(value=output.getvalue(), format='png')

class ImageNavigator:
    def __init__(self, image_paths, ocr, play_ms=1000):
        self.image_paths = image_paths
        self.play_ms = play_ms
        self.ocr = ocr

        self.index = 0
        self.resize_ratio = 1.0
        
        self.output = widgets.Output()
        self._setup_widgets()        
        
    def _setup_widgets(self):
        self.prev_button = widgets.Button(description='PREV')
        self.next_button = widgets.Button(description='NEXT')
        self.prev_button.on_click(self.on_prev_clicked)
        self.next_button.on_click(self.on_next_clicked)

        self.slider = widgets.IntSlider(value=0, min=0, max=len(self.image_paths)-1,
            step=1, description='Index:', continuous_update=False)
        self.slider.observe(self.on_slider_changed, names='value')

        self.resizer = widgets.FloatSlider(value=1.0, min=0.6, max=2.0,
            step=0.01, description='resize ratio:', continuous_update=False)
        self.resizer.observe(self.on_resizer_changed, names='value')
        
        self.play = widgets.Play(value=0, min=0, max=len(self.image_paths)-1, step=1,
            interval=self.play_ms, description="Auto", disabled=False)
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))

        self.vbox = widgets.VBox([
            widgets.HBox(
                [self.prev_button, self.next_button, self.play]),
                self.slider,
                self.resizer,
                self.output])

    def on_slider_changed(self, change):
        self.index = change['new']
        self.show_image(self.index)
    
    def on_resizer_changed(self, change):
        self.resize_ratio = change['new']
        self.show_image(self.index)
        
    def on_prev_clicked(self, b):
        if self.index > 0:
            self.index -= 1
        self.show_image(self.index)

    def on_next_clicked(self, b):
        if self.index < len(self.image_paths) - 1:
            self.index += 1
        self.show_image(self.index)

    @property
    def cursor(self):
        return self.image_paths[self.index]

    def run(self):
        display(self.vbox)
        self.show_image(self.index)

    def show_image(self, idx):
        curr_path = self.cursor
        filename = os.path.basename(curr_path)
        
        if self.resize_ratio != 1.0:
            rint = random.randint(1000, 10000)
            tmp_img_path = f'/tmp/{rint}.jpg'
            resized = resize_image_with_ratio(curr_path, self.resize_ratio)
            resized.save(tmp_img_path)
            curr_path = tmp_img_path
        
        with self.output:
            self.output.clear_output(wait=True)
            print(f"Image {idx+1} of {len(self.image_paths)}: {filename}")
            
            # image = Image.open(curr_path)
            painted = self.ocr.draw_ocr(curr_path, show_img=False)
            img = Image.fromarray(painted)
            display(img)
















