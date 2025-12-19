import os
import argparse
import glob
import base64
from io import BytesIO
from typing import Union, List, Tuple, Dict, Any
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from pathlib import Path
from dolphin_gguf.tools.utils import *

'''
Element Type	Description
sec_0 - sec_5	Hierarchical headings (title, level 1-5)
para	Regular paragraphs
half_para	Spanning paragraphs
equ	Mathematical formulas (LaTeX)
tab	Tables (HTML)
code	Code blocks (with indentation)
fig	Figures
cap	Captions
list	Lists
catalogue	Catalogs
reference	References
header / foot	Headers/Footers
fnote	Footnotes
watermark	Watermarks
anno	Annotations
'''

color_map = {
    "sec_0": "red",
    "sec_1": "blue",
    "sec_2": "green",
    "sec_3": "purple",
    "sec_4": "orange",
    "sec_5": "yellow",
    "para": "red",
    "half_para": "blue",
    "equ": "green",
    "code": "purple",
    "fig": "orange",
    "cap": "red",
    "list": "blue",
    "catalogue": "green",
    "reference": "purple",
    "header / foot": "orange",
    "fnote": "red",
    "watermark": "blue",
    "anno": "green",
}

URL="http://127.0.0.1:8000/v1"
MODEL_NAME="Dolphin-v2-GGUF"
API_KEY="111"

bbox_resize=1.54 # 1.54

class DOLPHIN_OpenAI:
    def __init__(self, url, model_name, api_key=""):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=url,
            api_key=api_key
        )
        print(f"Using OpenAI API at: {url} with model: {self.model_name}")

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """
        Encode image as JPEG (smaller token footprint than PNG)
        """
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def chat(
        self,
        prompt: Union[str, List[str]],
        image: Union[Image.Image, List[Image.Image]],
        max_tokens: int = 512,
    ):
        is_batch = isinstance(image, list)

        if not is_batch:
            images = [image]
            prompts = [prompt]
        else:
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)

        assert len(images) == len(prompts)

        results = []

        for img, question in zip(images, prompts):
            img = resize_img(img)
            base64_image = self._encode_image_to_base64(img)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ]

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                if response.choices:
                    results.append(response.choices[0].message.content.strip())
                else:
                    results.append("⚠️ Empty response")

            except Exception as e:
                print(f"VLM API error: {e}")
                raise e

        return results[0] if not is_batch else results

def process_document(document_path, model, save_dir, max_batch_size=None):

    """Parse documents with two stages - Handles both images and PDFs"""
    file_ext = os.path.splitext(document_path)[1].lower()
    TARGET_HIGH_DIM = 1600
    
    if file_ext == '.pdf':
        # Convert PDF to images
        images = convert_pdf_to_images(document_path)
        if not images:
            raise Exception(f"Failed to convert PDF {document_path} to images")
        
        all_results = []
        
        # Process each page
        for page_idx, pil_image in tqdm(enumerate(images), desc=f"Processing pdf pages"):
            
            pil_image = get_high_res_pil_image(pil_image, TARGET_HIGH_DIM)
            print(f"Processing page {page_idx + 1}/{len(images)}")
            # Generate output name for this page
            base_name = os.path.splitext(os.path.basename(document_path))[0]
            page_name = f"{base_name}_page_{page_idx + 1:03d}"
            
            json_dir = os.path.join(save_dir, "output_json")
            json_path = os.path.join(json_dir, f"{page_name}.json")

            if os.path.exists(json_path):
                print(f"Skipping page {page_idx + 1} as it already exists")
                continue
            
            json_path, recognition_results = process_single_image(
                pil_image, model, save_dir, page_name, max_batch_size
            )
            
            # Add page information to results
            page_results = {
                "page_number": page_idx + 1,
                "elements": recognition_results
            }
            all_results.append(page_results)
        
        # Save combined results for multi-page PDF
        combined_json_path = save_combined_pdf_results(all_results, document_path, save_dir)
        
        return combined_json_path, all_results
    
    else:
        # Process regular image file
        img = Image.open(document_path).convert("RGB")

        pil_image_high_res = get_high_res_pil_image(img, TARGET_HIGH_DIM)
        base_name = os.path.splitext(os.path.basename(document_path))[0]
        return process_single_image(pil_image_high_res, model, save_dir, base_name, max_batch_size)

def process_single_image(image, model, save_dir, image_name, max_batch_size=None):
    """Process a single image (Stage 1 & 2)"""
    # Stage 1: Page-level layout and reading order parsing
    print("\n--- Stage 1: Layout Parsing ---")
    layout_output = model.chat("Parse the reading order of this document.", image)
    print(f"Layout Output : {layout_output}")

    # Stage 2: Element-level content parsing
    print("\n--- Stage 2: Element Recognition ---")

    recognition_results = process_elements(layout_output, image, model, max_batch_size, save_dir, image_name)

    # Save outputs only if requested
    json_path = save_outputs(recognition_results, image, image_name, save_dir)

    print(f" stage 2 finished {json_path}")
    return json_path, recognition_results


def process_elements(layout_results, image, model, max_batch_size, save_dir=None, image_name=None):
    """Parse all document elements with parallel decoding"""
    layout_results_list = parse_layout_string(layout_results, bbox_resize)
    
    tab_elements = []      
    equ_elements = []     
    code_elements = []    
    text_elements = []     
    figure_results = []    
    reading_order = 0

    labeled_elements = []

    # Collect elements and group
    for bbox, label, tags in layout_results_list:
        try:
            # get coordinates in the original image
            x1, y1, x2, y2 = process_coordinates(bbox, image)
            # crop the image
            pil_crop = image.crop((x1, y1, x2, y2))

            if pil_crop.size[0] > 3 and pil_crop.size[1] > 3:

                if label == "fig":
                    figure_filename = save_figure_to_local(pil_crop, save_dir, image_name, reading_order)
                    figure_results.append({
                        "label": label,
                        "text": f"![Figure](figures/{figure_filename})",
                        "figure_path": f"figures/{figure_filename}",
                        "bbox": [x1, y1, x2, y2],
                        "reading_order": reading_order,
                        "tags": tags,
                    })
                else:

                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [x1, y1, x2, y2],
                        "reading_order": reading_order,
                        "tags": tags,
                    }
                    if label == "tab":
                        tab_elements.append(element_info)
                    elif label == "equ":
                        equ_elements.append(element_info)
                    elif label == "code":
                        code_elements.append(element_info)
                    else:
                        text_elements.append(element_info)
    
            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            raise e

    recognition_results = figure_results.copy()

    if tab_elements:
        results = process_element_batch(tab_elements, model, "Parse the table in the image.", max_batch_size)
        recognition_results.extend(results)
    
    if equ_elements:
        results = process_element_batch(equ_elements, model, "Read formula in the image.", max_batch_size)
        recognition_results.extend(results)
    
    if code_elements:
        results = process_element_batch(code_elements, model, "Read code in the image.", max_batch_size)
        recognition_results.extend(results)
    
    if text_elements:
        results = process_element_batch(text_elements, model, "Read text in the image.", max_batch_size)
        recognition_results.extend(results)

    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=None):
    results = []

    for elem in elements:
        crop = elem["crop"]

        # 根据任务类型控制 token 上限
        if elem["label"] == "tab":
            max_tokens = 1024
        elif elem["label"] == "equ":
            max_tokens = 512
        else:
            max_tokens = 256

        text = model.chat(
            prompt=prompt,
            image=crop,
            max_tokens=max_tokens,
        )

        results.append({
            "label": elem["label"],
            "bbox": elem["bbox"],
            "text": text.strip(),
            "reading_order": elem["reading_order"],
            "tags": elem["tags"],
        })

    return results


def document_parsing(input_image_path: str, save_directory: str = "ocr_results"):
    """
    使用硬编码参数执行两阶段文档解析流程。
    """
    
    if not os.path.exists(input_image_path):
        print(f"ERROR: Image file not found at {input_image_path}")
        return

    # 1. 实例化模型（连接到 llama-cpp-python 服务器）
    model = DOLPHIN_OpenAI(url=URL, model_name=MODEL_NAME, api_key=API_KEY)

    # 2. 准备输出目录
    os.makedirs(save_directory, exist_ok=True)

    # 3. 开始处理
    try:
        print(f"\nStarting processing for: {input_image_path}")
        json_path, recognition_results = process_document(
            document_path=input_image_path,
            model=model,
            save_dir=save_directory,
            max_batch_size=1,
        )

        print("-" * 40)
        print("✅ Parsing completed successfully!")
        print(f"Final results (first 3 elements): {recognition_results[:3]}")
        print(f"Results saved to: {save_directory}")
        print("-" * 40)

    except Exception as e:
        print(f"Error during document parsing: {e}")

def read_pages_in_order(directory):
    PAGE_PATTERN = re.compile(r".*_page_(\d+)\.json$")
    files_with_index = []

    for filename in os.listdir(directory):
        match = PAGE_PATTERN.match(filename)
        if match:
            page_num = int(match.group(1))  # 关键：转成 int
            files_with_index.append((page_num, filename))

    # 按 page 数字排序
    files_with_index.sort(key=lambda x: x[0])

    contents = []

    for page_num, filename in files_with_index:
        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8") as f:
            page_results = {
                "page_number": page_num,
                "elements": json.load(f)
            }
            contents.append(page_results)

    return contents

IGNORE_LABELS = ['header', 'footer', 'watermark']

def generate_tts_text(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        page_results = json.load(f)
    final_text = []

    tts_text = os.path.join(os.path.dirname(json_path), "tts_text.txt")
    page_results['pages'].sort(key=lambda x: x['page_number'])

    pre_label = ""
    for page in page_results['pages']:
        page['elements'].sort(key=lambda x: x['reading_order'])
        for element in page['elements']:
            if element['label'] in IGNORE_LABELS:
                continue
            if element['label'] != pre_label:
                final_text.append("\n")
            pre_label = element['label']
            final_text.append(element['text'].replace("\n", ""))
    
    with open(tts_text, "w", encoding="utf-8") as f:
        f.write("".join(final_text))
    
    print(f'generate tts text {tts_text}')
def combine():

    parser = argparse.ArgumentParser(description="Document parsing based on DOLPHIN")
    parser.add_argument("--input_path", type=str, default="./demo", help="Path of the OCR results directory")
    print('start combine ocr results')
    args = parser.parse_args()
    input_path = args.input_path
    combine_json(input_path)

def combine_json(input_path):

    json_dir = os.path.join(input_path, "output_json")
    all_results = read_pages_in_order(json_dir)
    base_name = os.path.basename(input_path.rstrip("/"))

    json_path = os.path.join(input_path, "recognition_json", "all_combined.json")
    if not os.path.exists(json_path):
        print("Combined json file not found, combine json first")
        save_combined_pdf_results(all_results, base_name, input_path)
    else:
        print(f'Combined json file found {json_path}')
    
    generate_tts_text(json_path)

def ocr():
    parser = argparse.ArgumentParser(description="Document parsing based on DOLPHIN")
    parser.add_argument("--input_path", type=str, default="./demo", help="Path to input image/PDF or directory of files")
    parser.add_argument("--output_path", type=str, default="./data_output", help="Path to data output directory")

    args = parser.parse_args()

    print(args)
    document_parsing(
        input_image_path=args.input_path,
        save_directory=args.output_path,
    )

if __name__ == "__main__":

    # combine_json("/Users/larry/github.com/dolphin_gguf/data_output/small_book")

    # file_path = "demo_1.png"
    # file_path = "small_pic.png"
    # file_path = "demo_1_512.jpg"
    # file_path = "demo_1_896.jpg"
    # file_path = "small_book.pdf"
    file_path = "test_book.pdf"

    file_path = "./demo/" + file_path
    fine_name = Path(file_path).stem

    document_parsing(
        input_image_path=file_path,
        # input_image_path="./demo/demo_1_512.jpg", 
        # input_image_path="./demo/demo_1_1024.jpg", 
        # input_image_path="./demo/test_small_book.pdf",
        # input_image_path="./demo/small_pic.png",
        # save_directory="./data_output",
        save_directory=f"./data_output/{fine_name}",
    )
# dolphin-run --input_path /Users/larry/github.com/dolphin_gguf/demo/en_page_2.jpeg --output_path /Users/larry/github.com/dolphin_gguf/data_output
# dolphin-run --input_path /Users/larry/github.com/dolphin_gguf/demo/zh_page_14.png --output_path /Users/larry/github.com/dolphin_gguf/data_output
    
