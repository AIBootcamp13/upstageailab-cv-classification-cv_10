import cv2
import numpy as np
import random
import albumentations as A

from albumentations.pytorch import ToTensorV2

lin = cv2.INTER_LINEAR
near = cv2.INTER_NEAREST
ipp = {'downscale':near, 'upscale': near}

class Transforms:
    def __init__(self, target_size):
        self.target_size = target_size
        
    def affine(self):
        tp = rand_ratio(-0.3, 0.3)
        scl, sch = rand_n_ratio(2, 0.5, 1.5)
        ssl, ssh = rand_n_int(2, -30, 30)
        return A.Affine(p=0.7, translate_percent=tp, scale=(scl, sch), rotate=(-90, 90), shear=(ssl, ssh))
        
    def resize_and_crops(self): 
        tfs = [
            A.CoarseDropout(p=0.6, num_holes_range=(2, 6), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), fill="random_uniform"),
            A.RandomResizedCrop(p=0.6, size=(self.target_size, self.target_size), scale=(0.2, 0.9), ratio=(0.75, 1.33), interpolation=lin, mask_interpolation=near, area_for_downscale="image"),
        ]
        return rand_select(tfs)
    
    def jitters(self, p=0.4):
        tfs = [
            A.Superpixels(p=p, p_replace=(0.1, 0.4), n_segments=(10, 40), max_size=256, interpolation=near),
            A.Downscale(p=p, scale_range=(0.1, 0.45), interpolation_pair=ipp),
            A.GaussNoise(p=p, std_range=(0.1, 0.4)),
            A.MotionBlur(p=p, blur_limit=7),
            A.GridDistortion(p=p, num_steps=5, distort_limit=0.7),
        ]
        return rand_select(tfs)
    
    def colors(self, p=0.4):
        tfs = [
            A.ToGray(p=p),
            A.HueSaturationValue(p=p, hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30), 
            A.RandomBrightnessContrast(p=p, brightness_limit=0.3, contrast_limit=0.3),
            A.RandomShadow(p=p,shadow_roi=[0,0,1,1]),
            A.PlasmaShadow(p=p, shadow_intensity_range=(0.1, 0.7), roughness=0.7),
            A.RGBShift(p=p, r_shift_limit=30, g_shift_limit=30, b_shift_limit=30),
            A.RingingOvershoot(p=p, blur_limit=(9, 17), cutoff=(np.pi/6, np.pi/3)),
            A.RandomGamma(gamma_limit=(30, 150), p=p)
        ]
        return rand_select(tfs)
    
    def rotators(self):
        tfs = [ 
            A.Sequential([
                A.Rotate(p=1.0, limit=(-10, 10)),
                A.SafeRotate(p=1.0, limit=(-90, 90)),
            ]),
            A.Sequential([
                A.Rotate(p=1.0, limit=(-90, 90)),
                A.SafeRotate(p=1.0, limit=(-90, 90), fill=255),
            ])
        ]
        return rand_select(tfs)
     
    
    def make_suite(self):
        tfs = [
            A.LongestMaxSize(self.target_size+30),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),        
            self.affine(),
        ]
        
        tfs += self.colors()
        tfs += self.jitters()
        tfs += self.resize_and_crops()
        tfs += self.rotators()
        random.shuffle(tfs)
        
        return tfs

    def make(self, num_suite=50):
        tfs = [A.Sequential(self.make_suite()) for _ in range(num_suite)]
        return A.Compose(A.OneOf(tfs, p=1.0))
        

        









'''
 sugars
'''
def make_transforms(num, generator):
    transforms = []
    for idx in range(num):
        t = generator(idx)
        transforms.append(t)
    return transforms        

def rand_flag():
    return random.random() > 0.5

def rand_n_ratio(n, low, high):
    return [random.uniform(low, high) for _ in range(n)]

def rand_ratio(low, high):
    return random.uniform(low, high)

def rand_n_int(n, low, high):
    return [random.randint(low, high) for _ in range(n)]

def rand_select(lst, k=1):
    return random.choices(lst, k=k)

