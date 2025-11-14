# Copyright 2025 J Joe

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageGrab
import torch
from pathlib import Path
from open_clip import create_model_from_pretrained, get_tokenizer
import platform

REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

@torch.inference_mode()
def load_model(device=None):
    model, preprocess = create_model_from_pretrained(f"hf-hub:{REPO}")
    tok = get_tokenizer(f"hf-hub:{REPO}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, preprocess, tok, device

@torch.inference_mode()
def encode_image(model, preprocess, img_pil, device):
    x = preprocess(img_pil).unsqueeze(0).to(device)
    f_img = model.encode_image(x)
    f_img = f_img / f_img.norm(dim=-1, keepdim=True)
    return f_img

@torch.inference_mode()
def encode_texts(model, tokenizer, device, prompts, context_length=256, batch_size=128):
    feats = []
    for i in range(0, len(prompts), batch_size):
        toks = tokenizer(prompts[i:i+batch_size], context_length=context_length).to(device)
        f = model.encode_text(toks)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f)
    return torch.cat(feats, dim=0)

@torch.inference_mode()
def image_text_scores(model, f_img, f_txt):
    cos = (f_img @ f_txt.t()).squeeze(0)
    logits = (model.logit_scale.exp() * cos).squeeze(0)
    return cos, logits

S1_NORMAL_PROMPTS = [
    "Chest radiograph showing no acute cardiopulmonary abnormality",
    "Chest X-ray showing no acute cardiopulmonary abnormality",
    "Chest X-ray showing no significant abnormality",
    "Normal chest radiograph",
    "Normal chest X-ray",
]
S1_ABN_PROMPTS = [
    "Chest radiograph showing abnormal findings",
    "Chest X-ray showing abnormal findings",
    "Abnormal chest radiograph",
    "Abnormal chest X-ray",
]

@torch.inference_mode()
def stage1_normal_abnormal(model, tok, device, f_img):
    prompts = S1_NORMAL_PROMPTS + S1_ABN_PROMPTS
    labels = (["normal"] * len(S1_NORMAL_PROMPTS)) + (["abnormal"] * len(S1_ABN_PROMPTS))
    
    f_txt = encode_texts(model, tok, device, prompts, 256, 128)
    cos, logits = image_text_scores(model, f_img, f_txt)
    
    idx_normal = [i for i, l in enumerate(labels) if l == "normal"]
    idx_abn = [i for i, l in enumerate(labels) if l == "abnormal"]
    
    mean_normal = logits[idx_normal].mean().item()
    mean_abn = logits[idx_abn].mean().item()
    
    sm_mean = torch.softmax(torch.tensor([mean_normal, mean_abn]), dim=-1).tolist()
    is_abn = sm_mean[1] >= sm_mean[0]
    
    return is_abn, sm_mean

S2_BANK = {
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "mediastinal widening": "mediastinal widening",
    "infiltrate": "pulmonary infiltrates",
    "nodule": "pulmonary nodule",
    "mass": "pulmonary mass",
    "fracture": "fracture",
    "atelectasis": "atelectasis",
    "pleural effusion": "pleural effusion",
    "pulmonary edema": "pulmonary edema",
    "cardiomegaly": "cardiomegaly",
}

def make_yes_no_prompts(phrase):
    yes = f"Chest X-ray showing {phrase}"
    no  = f"Chest X-ray showing no {phrase}"
    return yes, no

@torch.inference_mode()
def stage2_findings(model, tok, device, f_img, findings=None,
                    context_length=256, batch_size=128):

    if findings is None:
        findings = list(S2_BANK.keys())

    prompts = []
    index_map = []  

    for name in findings:
        if name not in S2_BANK:
            continue
        phrase = S2_BANK[name]
        yes, no = make_yes_no_prompts(phrase)
        prompts.append(yes)
        index_map.append((name, True))
        prompts.append(no)
        index_map.append((name, False))

    f_txt = encode_texts(model, tok, device, prompts,
                         context_length=context_length,
                         batch_size=batch_size)

    cos_sim, logits = image_text_scores(model, f_img, f_txt)

    results = {}

    by_name = {name: {"yes": [], "no": []} for name in findings}

    for i, (name, is_yes) in enumerate(index_map):
        if is_yes:
            by_name[name]["yes"].append(logits[i])
        else:
            by_name[name]["no"].append(logits[i])

    for name in findings:
        L_yes = torch.stack(by_name[name]["yes"]).mean()
        L_no  = torch.stack(by_name[name]["no"]).mean()
        sm = torch.softmax(torch.stack([L_no, L_yes]), dim=0).tolist()
        print(f"{name:20s} yes={sm[1]:.4f}  no={sm[0]:.4f}")

        if sm[1] > sm[0]:
            results[name] = sm[1] * 100

    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

def analyze_image(img_pil, model, preprocess, tok, device):
    f_img = encode_image(model, preprocess, img_pil, device)
    
    is_abn, sm_mean = stage1_normal_abnormal(model, tok, device, f_img)
    
    if not is_abn:
        return {"status": "normal", "findings": {}}
    
    findings = stage2_findings(model, tok, device, f_img, None)
    
    return {"status": "abnormal", "findings": findings}

class RegionSelector:
    def __init__(self, callback):
        self.callback = callback
        self.root = None
        self.canvas = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        
    def start_selection(self):
        self.root = tk.Tk()
        self.root.withdraw() 
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.overrideredirect(True) 
        self.root.attributes('-topmost', True)
        
        if platform.system() == "Darwin": 
            self.root.attributes('-alpha', 0.3)
        elif platform.system() == "Windows":
            self.root.attributes('-alpha', 0.3)
        else:
            try:
                self.root.attributes('-alpha', 0.3)
            except:
                pass 
        
        self.root.configure(bg='black')
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        
        self.canvas = Canvas(self.root, cursor="cross", bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("<Escape>", lambda e: self.cancel())
        
        self.root.mainloop()
    
    def on_press(self, event):
        self.start_x = self.root.winfo_rootx() + event.x
        self.start_y = self.root.winfo_rooty() + event.y
        self.canvas_start_x = event.x
        self.canvas_start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline='red', width=2
        )
    
    def on_drag(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.canvas_start_x, self.canvas_start_y, event.x, event.y)
    
    def on_release(self, event):
        end_x = self.root.winfo_rootx() + event.x
        end_y = self.root.winfo_rooty() + event.y
        
        x1 = min(self.start_x, end_x)
        y1 = min(self.start_y, end_y)
        x2 = max(self.start_x, end_x)
        y2 = max(self.start_y, end_y)
        
        self.root.destroy()
        
        try:
            bbox = (x1, y1, x2, y2)
            img = ImageGrab.grab(bbox)
            self.callback(img)
        except Exception as e:
            print(f"Capture error: {e}")
    
    def cancel(self):
        if self.root:
            self.root.destroy()

class CXRCapture:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tok = None
        self.device = None
        self.root = None
        
        print("Loading BiomedCLIP model...")
        self.model, self.preprocess, self.tok, self.device = load_model()
        print(f"Model loaded on {self.device}")
        print("\nReady. Press the button or Ctrl+Shift+C to capture.")
    
    def on_capture_click(self):
        selector = RegionSelector(self.process_capture)
        selector.start_selection()
    
    def process_capture(self, img):
        self.update_results("Analyzing...")
        self.root.update()
        
        try:
            results = analyze_image(img, self.model, self.preprocess, self.tok, self.device)
            output_lines = []
            output_lines.append("=" * 45)
            output_lines.append("CXR ANALYSIS RESULTS")
            output_lines.append("=" * 45)
            output_lines.append("")
            
            if results["status"] == "normal":
                output_lines.append("Status: NORMAL")
                output_lines.append("")
                output_lines.append("No significant abnormalities detected")
            else:
                output_lines.append("Status: ABNORMAL")
                output_lines.append("")
                if results["findings"]:
                    output_lines.append("Detected findings:")
                    output_lines.append("")
                    for pathology, confidence in results["findings"].items():
                        output_lines.append(f"  ✓ {pathology}: {int(confidence)}%")
                else:
                    output_lines.append("Abnormal but no specific pathologies")
                    output_lines.append("detected above threshold")
            
            output_lines.append("")
            output_lines.append("=" * 45)
            
            result_text = '\n'.join(output_lines)
            
            self.update_results(result_text)
            
            print("\n" + result_text)
            
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            print(error_msg)
            self.update_results(error_msg)
    
    def run(self):
        """Run minimal GUI with capture button and results display"""
        self.root = tk.Tk()
        self.root.title("CXR Capture")
        self.root.geometry("400x350")
        self.root.resizable(False, False)
        
        self.root.attributes('-topmost', False)
        
        btn = tk.Button(
            self.root, 
            text="Capture Region (Ctrl+Shift+C)", 
            command=self.on_capture_click,
            height=2,
            width=30,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        btn.pack(pady=10)
        
        result_frame = tk.Frame(self.root)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        tk.Label(result_frame, text="Results:", anchor='w', font=('Arial', 9, 'bold')).pack(anchor='w')
        
        scroll = tk.Scrollbar(result_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(
            result_frame, 
            height=15, 
            width=45,
            wrap=tk.WORD,
            font=('Courier', 9),
            state='disabled',
            yscrollcommand=scroll.set
        )
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.result_text.yview)
        
        self.update_results("Waiting for capture...")
        
        self.root.bind('<Control-Shift-C>', lambda e: self.on_capture_click())
        self.root.bind('<Control-Shift-c>', lambda e: self.on_capture_click())
        
        self.root.mainloop()
    
    def update_results(self, text):
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert('1.0', text)
        self.result_text.config(state='disabled')

if __name__ == "__main__":
    app = CXRCapture()
    app.run()

# pyinstaller --onefile --windowed --name CXRCapture cxr_capture.py
