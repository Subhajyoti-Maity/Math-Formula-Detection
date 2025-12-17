"""
Formula Extraction Module
Extracts detected math formulas from images and saves them with their LaTeX representations
"""

import os
import json
import csv
from datetime import datetime
from PIL import Image
import numpy as np
import cv2


def extract_formula_crops(image, bboxes):
    """
    Extract individual formula regions from the image based on bounding boxes
    
    Parameters:
        image: opencv image (numpy array)
        bboxes: list of bounding boxes in format [x1, y1, x2, y2, conf, cls]
    
    Returns:
        list of extracted formula images
    """
    crops = []
    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        crop = image[y1:y2, x1:x2]
        crops.append({
            'image': crop,
            'bbox': bbox,
            'coordinates': (x1, y1, x2, y2)
        })
    return crops


def recognize_formulas(extracted_crops, model_args, model_objs):
    """
    Recognize LaTeX formulas from extracted crop images
    
    Parameters:
        extracted_crops: list of extracted crop dictionaries
        model_args: recognition model arguments
        model_objs: recognition model objects (model, tokenizer)
    
    Returns:
        list of recognized formulas with their crops
    """
    import Recog_MathForm as RM
    
    formulas = []
    for idx, crop_data in enumerate(extracted_crops):
        crop_img = Image.fromarray(np.uint8(crop_data['image']))
        try:
            latex_pred = RM.call_model(model_args, *model_objs, img=crop_img)
            formulas.append({
                'id': idx + 1,
                'bbox': crop_data['bbox'],
                'coordinates': crop_data['coordinates'],
                'latex': latex_pred,
                'confidence': crop_data['bbox'][4]
            })
        except Exception as e:
            print(f"Error recognizing formula {idx + 1}: {e}")
            formulas.append({
                'id': idx + 1,
                'bbox': crop_data['bbox'],
                'coordinates': crop_data['coordinates'],
                'latex': 'ERROR',
                'confidence': crop_data['bbox'][4]
            })
    
    return formulas


def save_formulas_to_json(formulas, output_path='extracted_formulas.json'):
    """
    Save extracted formulas to JSON file
    
    Parameters:
        formulas: list of recognized formula dictionaries
        output_path: path to save JSON file
    """
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'total_formulas': len(formulas),
        'formulas': []
    }
    
    for formula in formulas:
        output_data['formulas'].append({
            'id': formula['id'],
            'coordinates': formula['coordinates'],
            'bbox': formula['bbox'],
            'latex': formula['latex'],
            'confidence': formula['confidence']
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_path


def save_formulas_to_csv(formulas, output_path='extracted_formulas.csv'):
    """
    Save extracted formulas to CSV file
    
    Parameters:
        formulas: list of recognized formula dictionaries
        output_path: path to save CSV file
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'X1', 'Y1', 'X2', 'Y2', 'LaTeX', 'Confidence'])
        
        for formula in formulas:
            coords = formula['coordinates']
            writer.writerow([
                formula['id'],
                coords[0],
                coords[1],
                coords[2],
                coords[3],
                formula['latex'],
                f"{formula['confidence']:.4f}"
            ])
    
    return output_path


def save_formula_images(extracted_crops, output_dir='extracted_formulas'):
    """
    Save individual formula images to directory
    
    Parameters:
        extracted_crops: list of extracted crop dictionaries
        output_dir: directory to save formula images
    
    Returns:
        list of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for idx, crop_data in enumerate(extracted_crops):
        filename = os.path.join(output_dir, f'formula_{idx + 1:04d}.png')
        cv2.imwrite(filename, crop_data['image'])
        saved_paths.append(filename)
    
    return saved_paths


def save_annotated_image(image, formulas, output_path='annotated_image.png'):
    """
    Save image with bounding boxes and LaTeX annotations
    
    Parameters:
        image: original opencv image
        formulas: list of recognized formula dictionaries
        output_path: path to save annotated image
    """
    annotated = image.copy()
    
    for formula in formulas:
        coords = formula['coordinates']
        x1, y1, x2, y2 = coords
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add LaTeX text above box
        text = f"ID: {formula['id']} | Conf: {formula['confidence']:.2f}"
        cv2.putText(annotated, text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add LaTeX formula as text
        latex_text = formula['latex'][:50]  # Truncate long formulas
        cv2.putText(annotated, latex_text, (x1, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    cv2.imwrite(output_path, annotated)
    return output_path


def save_html_report(formulas, image_path=None, output_path='formulas_report.html'):
    """
    Create an HTML report with extracted formulas
    
    Parameters:
        formulas: list of recognized formula dictionaries
        image_path: path to annotated image (optional)
        output_path: path to save HTML file
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Math Formula Extraction Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px; }
            .formula-card { 
                border: 1px solid #ddd; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px;
                background-color: #f9f9f9;
            }
            .latex { 
                background-color: #f0f0f0; 
                padding: 10px; 
                font-family: monospace; 
                border-left: 3px solid #4CAF50;
                margin: 10px 0;
            }
            .coordinates { color: #666; font-size: 0.9em; }
            .confidence { color: #4CAF50; font-weight: bold; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📐 Math Formula Extraction Report</h1>
            <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            <p>Total Formulas: <strong>""" + str(len(formulas)) + """</strong></p>
        </div>
    """
    
    if image_path and os.path.exists(image_path):
        html_content += f'<img src="{image_path}" style="max-width: 100%; border: 1px solid #ddd; margin: 20px 0;">'
    
    html_content += "<h2>Formulas Summary</h2><table><tr><th>ID</th><th>Coordinates (X1,Y1,X2,Y2)</th><th>LaTeX</th><th>Confidence</th></tr>"
    
    for formula in formulas:
        coords = formula['coordinates']
        coords_str = f"({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})"
        html_content += f"""
        <tr>
            <td>{formula['id']}</td>
            <td class="coordinates">{coords_str}</td>
            <td class="latex">{formula['latex']}</td>
            <td class="confidence">{formula['confidence']:.4f}</td>
        </tr>
        """
    
    html_content += "</table><h2>Detailed View</h2>"
    
    for formula in formulas:
        html_content += f"""
        <div class="formula-card">
            <h3>Formula #{formula['id']}</h3>
            <p><strong>Coordinates:</strong> {formula['coordinates']}</p>
            <p><strong>Confidence:</strong> <span class="confidence">{formula['confidence']:.4f}</span></p>
            <p><strong>LaTeX:</strong></p>
            <div class="latex">{formula['latex']}</div>
            <p><strong>Rendered (if LaTeX valid):</strong></p>
            <div class="latex">\\({formula['latex']}\\)</div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path
