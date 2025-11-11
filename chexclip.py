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

import argparse, torch
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

REPO = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

def load_model(device=None):
    model, preprocess = create_model_from_pretrained(f"hf-hub:{REPO}")
    tok = get_tokenizer(f"hf-hub:{REPO}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, preprocess, tok, device

def load_image(model, preprocess, path_or_url, device):
    img = Image.open(urlopen(path_or_url)) if path_or_url.startswith(("http://","https://")) else Image.open(path_or_url)
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
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
    return torch.cat(feats, dim=0)  # [N,D]

def image_text_scores(model, f_img, f_txt):
    cos = (f_img @ f_txt.t()).squeeze(0)                    # [N]
    logits = (model.logit_scale.exp() * cos).squeeze(0)     # [N]
    return cos, logits

def print_pairwise(title, names, cosines):
    print(title)
    for n, c in zip(names, cosines.tolist()):
        print(f"{n:>8} :: cos_sim={c:.3f}")

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

def stage1_normal_abnormal(model, tok, device, f_img, batch_size):
    prompts = S1_NORMAL_PROMPTS + S1_ABN_PROMPTS
    labels   = (["normal"] * len(S1_NORMAL_PROMPTS)) + (["abnormal"] * len(S1_ABN_PROMPTS))

    f_txt = encode_texts(model, tok, device, prompts, 256, batch_size)
    cos, logits = image_text_scores(model, f_img, f_txt)

    print("\n[Stage 1: prompts and cosine similarities]")
    for p, l, c in zip(prompts, labels, cos.tolist()):
        print(f"{l:10} :: {p}  --> cos_sim={c:.6f}")

    import math
    idx_normal = [i for i, l in enumerate(labels) if l == "normal"]
    idx_abn    = [i for i, l in enumerate(labels) if l == "abnormal"]

    mean_normal = logits[idx_normal].mean().item()
    mean_abn    = logits[idx_abn].mean().item()
    max_normal  = logits[idx_normal].max().item()
    max_abn     = logits[idx_abn].max().item()

    sm_mean = torch.softmax(torch.tensor([mean_normal, mean_abn]), dim=-1).tolist()
    sm_max  = torch.softmax(torch.tensor([max_normal,  max_abn]),  dim=-1).tolist()

    print("\n[Stage 1: aggregates]")
    print(f"MEAN -> normal={mean_normal:.6f}, abnormal={mean_abn:.6f}  | softmax=[{sm_mean[0]}, {sm_mean[1]}]")
    print(f"MAX  -> normal={max_normal:.6f},  abnormal={max_abn:.6f}   | softmax=[{sm_max[0]}, {sm_max[1]}]")

    is_abn = sm_mean[1] >= sm_mean[0]
    pred = "abnormal" if is_abn else "normal"
    print(f"\nS1 summary -> pred(mean): {pred}  | softmax_mean(normal,abnormal)={sm_mean}  softmax_max(normal,abnormal)={sm_max}")
    return is_abn

S2_BANK = {
    "pneumonia":                ["pneumonia", "airspace consolidation"],
    "pneumothorax":             ["pneumothorax"],
    "mediastinal widening":     ["mediastinal widening"],
    "atelectasis":              ["atelectasis"],
    "pleural effusion":         ["pleural effusion"],
    "pulmonary edema":          ["pulmonary edema"],
    "infilatrate":              ["pulmonary infiltrates"],
    "cardiomegaly":             ["cardiomegaly"],
    "nodule":                   ["pulmonary nodule", "solitary pulmonary nodule", "diffuse pulmonary nodules"],
    "mass":                     ["pulmonary mass", "lung mass"],
    "fracture":                 [f"{c} fracture" for c in ['rib', 'clavicle', 'vertebral']],
    "pleural thickening":       ["pleural thickening"],
    "ild / fibrosis":           ["interstitial lung disease", "pulmonary fibrosis"],
    "emphysema":                ["emphysema"],
    "cavitation":               ["cavitation"],
    "bronchiectasis":           ["bronchiectasis"],
    "hiatal hernia":            ["hiatal hernia"],
    "pneumoperitoneum":         ["pneumoperitoneum"],
    "calcified granuloma":      ["calcified granuloma"],
    "lymphadenopathy":          ["lymphadenopathy"],
    "pericardial effusion":     ["pericardial effusion"],
    "tuberculosis":             ["pulmonary tuberculosis"],
}

def make_yes_no_prompts(phrases):
    yes = [f"Chest X-ray showing {ph}" for ph in phrases]
    no  = [f"Chest X-ray showing no {ph}" for ph in phrases]
    return yes, no

@torch.inference_mode()
def stage2_findings(model, tok, device, f_img, findings, batch_size, agg="mean"):
    print("\n[Stage 2: prompts and cosine similarities]")
    if findings is None:
        findings = S2_BANK.keys()
    results = {}
    for name in findings:
        phrases = S2_BANK[name]
        yes_prompts, no_prompts = make_yes_no_prompts(phrases)
        prompts = [(name, "yes", p) for p in yes_prompts] + [(name, "no", p) for p in no_prompts]
        texts = [p for (_,_,p) in prompts]

        f_txt = encode_texts(model, tok, device, texts, 256, batch_size)
        cos, logits = image_text_scores(model, f_img, f_txt)

        for (nm, yn, p), c in zip(prompts, cos.tolist()):
            print(f"{nm:25} :: {yn:3} :: {p} --> cos_sim={c:.3f}")

        n = len(yes_prompts)
        yes_logits = logits[:n]
        no_logits  = logits[n:]
        if agg == "max":
            L_yes = yes_logits.max()
            L_no  = no_logits.max()
        else:
            L_yes = yes_logits.mean()
            L_no  = no_logits.mean()

        sm = torch.softmax(torch.stack([L_no, L_yes], dim=0), dim=0).tolist()
        results[name] = {
            "mean_yes": float(yes_logits.mean().item()),
            "mean_no":  float(no_logits.mean().item()),
            "max_yes":  float(yes_logits.max().item()),
            "max_no":   float(no_logits.max().item()),
            "softmax_no_yes": sm,
        }

    print("\n[Stage 2: per-pathology aggregates]")
    positives = {}
    for k, v in results.items():
        my, mn, MxY, MxN = v["mean_yes"], v["mean_no"], v["max_yes"], v["max_no"]
        sm = v["softmax_no_yes"]
        print(f"{int(sm[1]*100):>4d}% --> {k:25} :: mean_yes={my:.3f}  mean_no={mn:.3f}  | 2-way softmax(no,yes)={sm}")
        if sm[1]> sm[0]:
            positives[k] = sm[1]*100
    return dict(sorted(positives.items(), key=lambda item: item[1], reverse=True))

SIDES = ["right sided","left sided", "bilateral", "no signs of"]
LOBES = ["right upper lobe","right middle lobe","right lower lobe","left upper lobe","left lower lobe", "no signs of"]

S3_SCHEMAS = {
    "pneumonia":        lambda: [f"{l} pneumonia" for l in ["atypical", "PCP", "varicella"] + LOBES],
    "pneumothorax":     lambda: [f"{s} pneumothorax" for s in SIDES],
    "nodule":           lambda: ['diffuse pulmonary nodules'] + [f"{l} pulmonary nodule" for l in LOBES],
    "mass":             lambda: [f"{l} lung mass" for l in LOBES],
    "fracture":         lambda: ["left rib fracture", "right rib fracture",
                                 "left clavicle fracture", "right clavicle fracture",
                                 "upper vertebral fracture", "mid vertebral fracture", "lower vertebral fracture",
                                 "no evidence of fracture or joint dislocation"],
    "atelectasis":      lambda: [f"{l} atelectasis" for l in LOBES],
    "infiltrate":       lambda: ["pulmonary infiltrates in the right lung", "pulmonary infiltrates in the left lung", "bilateral pulmonary infiltrates", "no signs of abnormal substances like pus, blood, or fluid in the lungs"],
    "pleural effusion": lambda: [f"{s} pleural effusion" for s in SIDES],
    "lymphadenopathy":  lambda: ["right hilar lymphadenopathy", "left hilar lymphadenopathy", "mediastinal lymphadenopathy", "no signs of lymphadenopathy"],
}

def s3_prompts_for(kind):
    if kind not in S3_SCHEMAS:
        return []
    items = S3_SCHEMAS[kind]()        
    prompts = [f"Chest X-ray showing {x}" for x in items]
    return items, prompts

@torch.inference_mode()
def stage3_locate(model, tok, device, f_img, kinds, batch_size):
    if kinds is None:
        kinds = S3_SCHEMAS.keys()
    print("\n[Stage 3: location/side softmax within each pathology (with negative control)]")
    summaries = {}
    for kind in kinds:
        labels, prompts = s3_prompts_for(kind)
        if not prompts:
            continue
        f_txt = encode_texts(model, tok, device, prompts, 256, batch_size)
        cos, logits = image_text_scores(model, f_img, f_txt)

        print(f"\n(S3 {kind}) prompts and cosine similarities:")
        for lab, prm, c in zip(labels, prompts, cos.tolist()):
            print(f"{lab:30} :: cos_sim={c:.3f} --> {prm}")

        probs = torch.softmax(logits, dim=0).tolist()
        dist = list(zip(labels, probs))
        print(f"(S3 {kind}) softmax over [{', '.join(labels)}]:")
        for lab, p in dist:
            print(f"{int(p*100):>4d}% --> {lab}")

        summaries[kind] = dist
    return summaries

@torch.inference_mode()
def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=False, default=None, help="path or URL")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--s2_list", nargs="*", default=None)
    ap.add_argument("--s3_kinds", nargs="*", default=None)
    ap.add_argument("--s2_agg", choices=["mean","max"], default="mean")
    ap.add_argument("--s4_list", nargs="*", default=['pneumonia', 'pneumothorax', 'mediastinal widening'])
    args = ap.parse_args()

    model, preprocess, tok, device = load_model()
    if args.image is None:
        url = "https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/chest_X-ray.jpg"
        print(f"[info] No --image given; using sample: {url}")
        f_img = load_image(model, preprocess, url, device)
    else:
        f_img = load_image(model, preprocess, args.image, device)

    is_abn = stage1_normal_abnormal(model, tok, device, f_img, args.batch_size)

    d_abns = stage2_findings(model, tok, device, f_img, args.s2_list, args.batch_size, agg=args.s2_agg)

    _ = stage3_locate(model, tok, device, f_img, args.s3_kinds, args.batch_size)

    print('∴∴∴ RESULT ∴∴∴')
    for k, v in d_abns.items():
        if k in args.s4_list:
            print(f'✓ {k}: {int(v)}%')


if __name__ == "__main__":
    run()
