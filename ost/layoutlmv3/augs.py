import cv2
import numpy as np
import random
import albumentations as A

from albumentations.pytorch import ToTensorV2

lin = cv2.INTER_LINEAR
near = cv2.INTER_NEAREST
boad =  cv2.BORDER_CONSTANT
ipp = {'downscale':near, 'upscale': near}

class Transforms:
    def __init__(self, target_size):
        self.target_size = target_size
        
    def affine(self):
        tp = rand_ratio(-0.1, 0.1)
        scl, sch = rand_n_ratio(2, 0.8, 1.2)
        ssl, ssh = rand_n_int(2, -5, 5)
        return A.Affine(p=0.5, translate_percent=tp, scale=(scl, sch), rotate=(-5, 5), shear=(ssl, ssh))
    

        
    def resize(self): 
        tfs = [
                A.Perspective(
                    scale=(0.05, 0.08),       # 살짝만 테두리를 왜곡
                    keep_size=False,           # 원래 크기로 리사이즈 (중요)
                    fit_output=True,         # 이미지 영역 확장 없이 고정된 평면에 투영
                    border_mode=cv2.BORDER_CONSTANT,  # 패딩 방식: 검정색 여백
                    fill=0,                   # 여백 색상
                    p=1.0                     # 항상 적용
                )
        ]
            #A.CoarseDropout(p=0.5, num_holes_range=(3, 8), hole_height_range=(0.05, 0.15), hole_width_range=(0.1, 0.3), fill=0),  # 검은 마스킹 증가
            #ARandomResizedCrop(p=0.7, size=(self.ta.rget_size, self.target_size), scale=(0.3, 0.8), ratio=(0.6, 1.5), interpolation=lin, mask_interpolation=near),  # crop 강화

        return rand_select(tfs)
    def crop(self): 
        tfs = [
            #A.Perspective(scale=(0.1, 0.3), keep_size=False,fit_output=True, p=1),  # 원래 (0.15, 0.25) → 줄임
            #A.CoarseDropout(p=0.5, num_holes_range=(3, 8), hole_height_range=(0.05, 0.15), hole_width_range=(0.1, 0.3), fill=0),  # 검은 마스킹 증가
            A.RandomResizedCrop(p=0.7, size=(self.target_size, self.target_size), scale=(0.3, 0.8), ratio=(0.6, 1.5), interpolation=lin, mask_interpolation=near),  # crop 강화
        ]
        return rand_select(tfs)

    def jitters(self, p=0.4):
        tfs = [
            A.Superpixels(
                p=p,
                p_replace=(0.1, 0.3),           
                n_segments=(10, 40),            
                max_size=256,
                interpolation=near
            ),
            A.Downscale(
                p=p,
                scale_range=(0.3, 0.6),          
                interpolation_pair=ipp
            ),
            A.ISONoise(p=p),
            A.GaussNoise(
                std_range=(0.02, 0.15),         
                p=p
            ),
             A.GaussNoise(
                std_range=(0.1, 0.3),          
                p=p
            ),
            A.MotionBlur(
                blur_limit=(3, 15),              
                p=p
            ),
            A.GridDistortion(
                p=p,
                num_steps=5,
                distort_limit=0.7               
            ),
            A.Blur(blur_limit=3, p=0.2)
        ]
        return rand_select(tfs)
    
    def colors(self, p=0.4):
        tfs = [
            A.ToGray(p=p),  # 유지: 색상 제거 통한 강건성 확보
            A.Solarize(threshold_range=(0.4, 0.6), p=0.2),

            A.ImageCompression(
                quality_range=(60, 90),  # 문서 스캔/압축 품질 시뮬레이션
                compression_type="jpeg",  # 기본값이지만 명시적으로
                p=p
            ),
            A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=p
            ),
            A.HueSaturationValue(
                p=p,
                hue_shift_limit=10,      # 기존 30 → **10으로 감소** (색조 변화가 작음)
                sat_shift_limit=20,      # 기존 30 → **20으로 약화**
                val_shift_limit=20       # 기존 30 → **20으로 약화**
            ),
            A.RandomBrightnessContrast(
                p=p,
                brightness_limit=0.2,    # 기존 0.3 → **0.2로 소폭 낮춤 (밝은 이미지 비중 많음)**
                contrast_limit=0.2       # 기존 0.3 → **0.2로 소폭 낮춤**
            ),
            A.RandomShadow(
                p=p,
                shadow_roi=[0, 0, 1, 1]  # 유지
            ),
            A.RingingOvershoot(
                p=p,
                blur_limit=(7, 13),      # 기존 (9,17) → **(7,13)로 소폭 약화**
                cutoff=(np.pi/8, np.pi/3) # 기존 (π/6, π/3) → **조금 더 약한 시작점**
            ),
            A.RandomGamma(
                gamma_limit=(70, 140),   # 기존 (30, 150) → **(70, 140)** (너무 어두운/밝은 변형 줄임)
                p=p
            )
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
            #A.LongestMaxSize(self.target_size+30),
            #A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.5),
            #A.Transpose(p=0.5),        
            #self.affine(),
        ]
        
        tfs += self.colors()
        tfs += self.jitters()
        tfs += self.resize()
        tfs += self.crop()
        
        random.shuffle(tfs)
        
        return tfs


    def make(self, num_suite=50):
        tfs = [A.Sequential(self.make_suite()) for _ in range(num_suite)]
        return A.Compose([
        A.OneOf(tfs, p=1.0),
        # A.Normalize(normalization="image", p=0.9),
        ToTensorV2()
        ], 
        bbox_params=A.BboxParams(
            format='pascal_voc',     # pascal_voc가 x0, y0, x1, y1 포맷 
            label_fields=['words'],  # bbox와 대응되어 양항을 받는 속성들
            min_area=0,                    
            min_visibility=0.0,            
            check_each_transform=True,     
            clip=True                      
        ))
        

        









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

