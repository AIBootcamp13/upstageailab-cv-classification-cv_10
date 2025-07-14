import os
import csv
import json
import pandas as pd
import ipywidgets as widgets

from PIL import Image
from IPython.display import display, clear_output


# 기본 캘래스 번역
label_trans = {
    "account_number":"계좌번호",
    "application_for_payment_of_pregnancy_medical_expenses": "임신/출산 신청서",
    "car_dashboard": "자동차 계기판",
    "confirmation_of_admission_and_discharge": "입/퇴원 확인서",
    "diagnosis": "진단서",
    "driver_lisence": "운전면허증",
    "medical_bill_receipts": "진료/의료비 영수증",
    "medical_outpatient_certificate": "(외래)진료(통원/치료) 확인서",
    "national_id_card": "주민등록증",
    "passport": "여권",
    "payment_confirmation": "(진료비/약제비) 납입 확인서",
    "pharmaceutical_receipt": "약국/영수증",
    "prescription": "처방전",
    "resume": "이력서",
    "statement_of_opinion": "소견서",
    "vehicle_registration_certificate": "자동차 등록증",
    "vehicle_registration_plate": "자동차 번호판"
}



def load_json(path):
  with open(path) as f:
    return json.load(f)

def make_doc_class_mapper(json_path):
    dc = load_json(json_path)
    classes = dc['document_classes']
    label2id = {v: k for k, v in enumerate(classes)}
    id2label = {k: v for k, v in enumerate(classes)}
    return label2id, id2label

    
class Inspector:
    def __init__(self, ds_dir_path, csv_path, classes_json_path, trans=None):
        self.all_items = []
        self.label2id = {}
        self.id2label = {}
        self.label_trans = label_trans if trans is None else trans
        self.ds_dir_path = ds_dir_path

        self.current_index = 0
        self.task_idx = 0
        self.task_items = []
        self.status_hard = False
        self.etc_notes = {}
        self.checked_items = []
        
        self.buttons = []
        self.button_hard = None
        self.next_button = widgets.Button(description='NEXT')
        self.prev_button = widgets.Button(description='PREV')
        self.selected_label_widget = widgets.HTML()

        self.csv_path = csv_path
        self._load_all_items(csv_path, classes_json_path)
        
        self.out = widgets.Output()
        self.slider = None
        self._setup_slider()
        self._setup_buttons()

    @property
    def curr_image_path(self):
        return os.path.join(self.ds_dir_path, self.filename)
    
    @property
    def filename(self):
        return self.task_items[self.current_index][0]
        
    @property
    def last_idx(self):
        return len(self.task_items) - 1

    @property 
    def progress(self):
        return f"[{self.filename}] {self.current_index+1}/{len(self.task_items)}"

    @property
    def item_cursor(self):
        return self.task_items[self.current_index]
    
    def _load_all_items(self, csv_path, classes_json_path):
        items = []
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['ID']
                target = int(row['target'])
                items.append((filename, target))
        items = sorted(items, key=lambda x: x[0])
        
        self.all_items = items
        self.task_items = items[:] # copy
        label2id, id2label = make_doc_class_mapper(classes_json_path)
        self.label2id = label2id
        self.id2label = id2label

    def on_slider_change(self, change):
        self.current_index = change['new']
        self._show()

    def _setup_slider(self):
        num_items = len(self.all_items)
        self.slider = widgets.IntSlider(value=0, min=0, max=num_items-1, step=1,
            description='인덱스(0기준)',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.slider.observe(self.on_slider_change, names='value')

    def _setup_buttons(self):
        self.buttons = []
        # 선택지 추가
        for label, label_id in self.label2id.items():
            display_text = self.label_trans.get(label, label)
            button = widgets.Button(
                description=f"{display_text} ({label_id})",
                layout=widgets.Layout(width='auto', min_width='80px', max_width='300px', margin='3px 3px 3px 3px'),
                style={'button_color': '#f0f0f0', 'font_size': '14px'}
            )
            button.on_click(self._make_label_handler(label_id))
            self.buttons.append(button)

        # 모름 버튼 추가
        button = widgets.Button(description='모름/어려움 (-1)',
           layout=widgets.Layout(width='auto', min_width='80px', max_width='300px', margin='3px 3px 3px 3px'),
            style={'button_color': '#f0f0f0', 'font_size': '15px'}
        )
        button.on_click(self._make_label_handler(-1))
        self.buttons.append(button)

          # 고난이도 버튼 추가
        self.button_hard = widgets.Button(description='고난이도 체크',
           layout=widgets.Layout(width='auto', min_width='80px', max_width='300px', margin='0 10px 0 130px'),
            style={'button_color': '#f0f0f0', 'font_size': '15px'}
        )
        self.button_hard.on_click(self._make_label_handler(-2))

        self.next_button.layout = widgets.Layout(width='80px', margin='0 0 10px 10px')
        self.prev_button.layout = widgets.Layout(width='80px', margin='0 10px 10px 0')
        self.next_button.on_click(self._on_next)
        self.prev_button.on_click(self._on_prev)
    
    def get_check_item(self, label_id):
        filename = self.filename
        curr_index = self.current_index
        
        prev_label_id = self.all_items[self.current_index][1]
        prev_label_kor = self.get_kor_label_by_id(prev_label_id)
        
        next_label_id = label_id
        next_label_kor = self.get_kor_label_by_id(next_label_id)

        return {
            'filename': filename,
            'file_idx': curr_index,
            'prev_label_id': prev_label_id,
            'prev_label_kor': prev_label_kor,
            'next_label_id': label_id,
            'next_label_kor' : next_label_kor
        }

    def get_kor_label_by_id(self, label_id):
        if int(label_id) == -1:
            kor_label = "모름 (어려움)"
        else:
            label_name = self._get_label_name(label_id)
            kor_label = self.label_trans.get(label_name, label_name)
        return kor_label

    def _make_label_handler(self, label_id):
        def handler(b):
            item = (self.filename, label_id)
            self.task_items[self.current_index] = item

            check_item = self.get_check_item(label_id)
            self.checked_items.append(check_item)
            kor_label = self.get_kor_label_by_id(label_id)
                
            self.selected_label_widget.value = f"<b style='font-size: 24px;'>선택된 레이블: {kor_label} ({label_id})</b>"
            with self.out:
                clear_output(wait=True)
                print(f"Image {self.progress} 레이블: '{kor_label}' ({label_id})")

        def etc_handler(b):
            self.etc_notes[self.filename] = f'[{self.current_index}] 고난이도'

        return handler if label_id >= -1 else etc_handler

    def _get_label_name(self, label_id):
        for k, v in self.label2id.items():
            if v == label_id:
                return k
        return str(label_id)

    def _display_image(self):
        with self.out:
            clear_output(wait=True)
            print(f"Image {self.progress}")
            path = self.curr_image_path
            if os.path.exists(path):
                img = Image.open(path)
                display(img)
            else:
                print(f"Image file {path} does not exist.")
            
        item = self.item_cursor
        if item is not None:
            # 하위 호환성 체크 및 상위 구조 변환
            if type(item) is not tuple:
                label_id = item
                self.task_items[self.current_index] = (self.curr_image_path, label_id)
            else:
                label_id = item[1]

            kor_label = self.get_kor_label_by_id(label_id)
            self.selected_label_widget.value = f"<b style='font-size: 24px;'>선택된 레이블: {kor_label} ({label_id})</b>"
        else:
            self.selected_label_widget.value = "<b>선택된 레이블:</b> 없음"

    def _on_next(self, b):
        if self.current_index < self.last_idx - 1:
            self.current_index += 1
            self.slider.value = self.current_index
            self._show()
        else:
            with self.out:
                print("This is the last image.")

    def _on_prev(self, b):
        if self.current_index > 0:
            self.current_index -= 1
            self.slider.value = self.current_index
            self._show()
        else:
            with self.out:
                print("This is the first image.")

    def _show(self):
        self._display_image()
            
        display(self.selected_label_widget)  # 최상단에 선택 레이블 표시
        display(self.slider)
        display(widgets.HBox([self.prev_button, self.next_button, self.button_hard]))

        # 버튼
        rows = []
        num_in_rows = [(0, 6), (6, 11), (11, len(self.buttons))]
        for rn in num_in_rows:
            a, b = rn
            rows.append(widgets.HBox(self.buttons[a:b]))
        for row in rows:
            display(row)
        display(self.out)

    def inspect(self):
        self._show()

    def overwrite_csv(self):
        df = pd.read_csv(self.csv_path)
        for idx, item in enumerate(self.task_items):
            filename = os.path.basename(item[0])
            file_id = df.loc[idx, 'ID']
            if filename != file_id: 
                print(f"에러: 불일치 행[{idx}] {filename} != {file_id}")
                continue
            df.loc[idx, 'target'] = item[1]
            
        df.to_csv(self.csv_path, index=False, encoding='utf-8')

        print(">>> 지금까지 직접 체크했던 요소들 >>>")
        print(self.checked_items)
        self.checked_items = []


















'''
legacy
'''
def get_my_turn(csv_path, idx=0, chunk_size=100, encoding='utf-8'):
    items = []
    with open(csv_path, mode='r', encoding=encoding) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row['ID']
            label = row['target']
            items.append((key, label))
            
    items = sorted(items, key=lambda x: x[0])
    start_idx = idx* chunk_size
    
    quota = items[start_idx: start_idx+chunk_size]
    return [q[0] for q in quota], [int(q[1]) for q in quota]   