import os
import random
import io
import numpy as np

import ipywidgets as widgets
from ipywidgets import VBox, HBox, Button, Output

from IPython.display import display, clear_output, Image as IImage
from PIL import Image, ImageOps


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

class TrainImageNavigator:
    def __init__(self, image_paths, target_path, id2kor, play_ms=1000):
        self.index = 0
        self.image_paths = image_paths
        self.play_ms = play_ms
        self.id2kor = id2kor
        
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
        
        self.play = widgets.Play(value=0, min=0, max=len(self.image_paths)-1, step=1,
            interval=self.play_ms, description="Auto", disabled=False)
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))

        self.vbox = widgets.VBox([
            widgets.HBox(
                [self.prev_button, self.next_button, self.play]),
                self.slider,
                self.output])
        display(self.vbox)

    def run(self):
        self.show_image(self.index)

    def on_slider_changed(self, change):
        self.index = change['new']
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

    def show_image(self, idx):
        curr_path = self.cursor
        filename = os.path.basename(curr_path)
        
        with self.output:
            self.output.clear_output(wait=True)
            print(f"Image {idx+1} of {len(self.image_paths)}: {filename}")
            image = Image.open(curr_path)
            display(image)
   

class ImageNavigator:
    def __init__(self, image_paths, hists, id2kor, play_ms=1000):
        self.image_paths = image_paths
        self.play_ms = play_ms
        self.hists = hists
        self.id2kor = id2kor

        self.index = 0
        
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
        
        self.play = widgets.Play(value=0, min=0, max=len(self.image_paths)-1, step=1,
            interval=self.play_ms, description="Auto", disabled=False)
        widgets.jslink((self.play, 'value'), (self.slider, 'value'))

        self.vbox = widgets.VBox([
            widgets.HBox(
                [self.prev_button, self.next_button, self.play]),
                self.slider,
                self.output])
        display(self.vbox)

    def run(self):
        self.show_image(self.index)

    def on_slider_changed(self, change):
        self.index = change['new']
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

    def show_image(self, idx):
        curr_path = self.cursor
        filename = os.path.basename(curr_path)

        hists = self.hists
        df = hists[0]
        df_idx = df.index[df['ID'] == filename][0]
        
        preds = []
        for hist in hists:
            label_id = hist.iloc[df_idx]['target'] 
            preds.append(self.id2kor[label_id])
        
        with self.output:
            self.output.clear_output(wait=True)

            print(f"Image {idx+1} of {len(self.image_paths)}: {filename}")
            print(f"예측 결과 {preds}")
            # print(f"예측 결과 {preds}  conf: {df.iloc[df_idx]['prob']}")
            
            image = Image.open(curr_path)
            display(image)


class DualImageNavigator(ImageNavigator):
    def __init__(self, image_paths, process_func, play_ms=1000):
        super().__init__(image_paths, play_ms)
        self.process_func = process_func

    def to_widget_image(self, img1, img2):  # for horizontal display
        wimg1 = pil_to_ipywidget(img1)
        wimg2 = pil_to_ipywidget(img2)
        return wimg1, wimg2

    def show_image(self, idx):
        with self.output:
            self.output.clear_output(wait=True)
            curr_path = self.cursor
            filename = os.path.basename(curr_path)
    
            print(f"Image {idx+1} of {len(self.image_paths)}: {filename}")
            origin, rotated = self.process_func(curr_path)
            wa, wb = self.to_widget_image(origin, rotated)
            display(widgets.HBox([wa, wb]))
        

class ImageSelectorNavigator(ImageNavigator):
    def __init__(self, image_paths, func, play_ms=1000, auto_save=False):
        self.auto_save = auto_save
        super().__init__(image_paths, play_ms)
        self.func = func
        self.selected_paths = []
        self.show_image(self.index)

    def on_image_click(self, path):
        def handler(btn):
            if path not in self.selected_paths:
                self.selected_paths.append(path)
            self.on_next_clicked(None)
        return handler

    def show_image(self, idx):
        image_buttons = []
        curr_path = self.cursor
        paths = self.func(curr_path)
        filename = os.path.basename(curr_path)

        if self.auto_save and idx % 100 == 0:
            rnd = random.randint(0, 10000)
            write_list(f'./{self.index}-{rnd}.txt', self.selected_paths)
        
        self.output.clear_output(wait=True)
        
        for path in paths:
            out = Output()
            with out:
                display(IImage(filename=path))
            btn = Button(description='선택', layout={'width': '200px', 'height': '70px'})
            btn.on_click(self.on_image_click(path))
            image_buttons.append(VBox([out, btn]))
            
        grid = []
        for i in range(0, 9, 3):
            grid.append(HBox(image_buttons[i:i+3]))
        
        with self.output:
            print(f"Image {idx+1} of {len(self.image_paths)}: {filename}")
            widget_box = VBox(grid)
            display(widget_box)
   
