# from flask import Flask, request, jsonify, send_from_directory, render_template
# from ultralytics import YOLO
# import os
# import shutil
# import cv2
# import numpy as np
# import glob

# app = Flask(__name__, static_folder='static', template_folder='templates')

# # Paths
# UPLOAD_FOLDER = "uploads"
# RESULTS_FOLDER = "results_of_predictions"
# CROP_FOLDER = "Crop_images"
# MODEL_PATH = "best.pt"

# # Ensure necessary directories exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(RESULTS_FOLDER, exist_ok=True)
# os.makedirs(CROP_FOLDER, exist_ok=True)

# # Load YOLO model
# model = YOLO(MODEL_PATH)

# # Function to clear the results directory
# def clear_directory(directory):
#     if os.path.exists(directory):
#         for file_name in os.listdir(directory):
#             file_path = os.path.join(directory, file_name)
#             if os.path.isfile(file_path):
#                 os.remove(file_path)
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)

# # Function to perform prediction and crop
# def predict_and_crop(image_path):
#     # Clear the results directory before prediction
#     clear_directory(RESULTS_FOLDER)

#     # Predict on the uploaded image
#     results = model.predict(
#         source=image_path,
#         show=False,  # Change this to False
#         save=True,
#         hide_labels=False,  # You can change this if needed
#         hide_conf=False,    # You can change this if needed
#         show_labels=True,   # New parameter
#         show_conf=True,     # New parameter
#         conf=0.5,
#         save_txt=True,
#         save_crop=False,
#         line_width=2,
#         project=RESULTS_FOLDER,
#         name="latest_prediction"  # Use the same name each time
#     )


#     # Define the label path for the predicted results
#     label_path = os.path.join(RESULTS_FOLDER, "latest_prediction/labels", os.path.basename(image_path).replace('.png', '.txt'))
#     draw_segmentation(image_path, label_path, CROP_FOLDER)

# # Function to process and crop images based on segmentation masks
# def draw_segmentation(image_path, label_path, output_path):
#     # Clear output folder before processing
#     if os.path.exists(output_path):
#         files = glob.glob(os.path.join(output_path, '*.png'))
#         for file in files:
#             os.remove(file)
#         print(f"Deleted {len(files)} existing images in {output_path}")
#     else:
#         os.makedirs(output_path)

#     # Load the original image
#     original_image = cv2.imread(image_path)
#     height, width = original_image.shape[:2]

#     with open(label_path, 'r') as f:
#         lines = f.readlines()

#     img_counter = 1
#     for line in lines:
#         data = line.strip().split()
#         points = np.array([float(coord) for coord in data[1:]]).reshape(-1, 2)

#         # Scale points to original image dimensions
#         points[:, 0] *= width
#         points[:, 1] *= height
#         points = points.astype(np.int32)

#         mask = np.zeros((height, width), dtype=np.uint8)
#         cv2.fillPoly(mask, [points], color=255)

#         # Create the masked image
#         masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

#         # Find the bounding box of the masked area
#         masked_indices = np.where(mask == 255)
#         if masked_indices[0].size == 0:
#             continue  # Skip if no mask found
#         min_y, max_y = np.min(masked_indices[0]), np.max(masked_indices[0])
#         min_x, max_x = np.min(masked_indices[1]), np.max(masked_indices[1])
#         cropped_panel = masked_image[min_y:max_y + 1, min_x:max_x + 1]

#         # Add text to the cropped image
#         text = f'Solar_plate_{img_counter}'
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 1
#         font_thickness = 2
#         text_color = (0, 0, 0)
#         background_color = (192, 192, 192)

#         (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
#         text_x = (cropped_panel.shape[1] - text_width) // 2
#         text_y = (cropped_panel.shape[0] + text_height) // 2

#         # Create text background
#         text_background_x1 = text_x - 10
#         text_background_y1 = text_y - text_height - 10
#         text_background_x2 = text_x + text_width + 10
#         text_background_y2 = text_y + 10
#         cv2.rectangle(cropped_panel, (text_background_x1, text_background_y1), (text_background_x2, text_background_y2), background_color, -1)
#         cv2.putText(cropped_panel, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

#         # Save the cropped image
#         crop_output_path = os.path.join(output_path, f'Solar_plate_{img_counter}.png')
#         cv2.imwrite(crop_output_path, cropped_panel)

#         img_counter += 1

#     print(f"Cropped images with text saved to {output_path}")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"success": False, "error": "No file part"})

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"success": False, "error": "No selected file"})

#     # Save uploaded file
#     image_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(image_path)

#     # Perform prediction and crop
#     predict_and_crop(image_path)

#     # Get cropped files
#     cropped_files = os.listdir(CROP_FOLDER)
#     return jsonify({"success": True, "files": cropped_files})

# @app.route('/download/<filename>')
# def download_file(filename):
#     return send_from_directory(CROP_FOLDER, filename)

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for
from ultralytics import YOLO
import os
import shutil
import cv2
import numpy as np
import glob

app = Flask(__name__, static_folder='static', template_folder='templates')

# Paths
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results_of_predictions"
CROP_FOLDER = "Crop_images"
MODEL_PATH = "best.pt"

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Function to clear a directory
def clear_directory(directory):
    if os.path.exists(directory):
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

# Before every request, clear the contents of specific folders
@app.before_request
def clear_folders_on_refresh():
    clear_directory(RESULTS_FOLDER)
    clear_directory(UPLOAD_FOLDER)
    clear_directory(CROP_FOLDER)

# Function to perform prediction and crop
def predict_and_crop(image_path):
    # Predict on the uploaded image
    results = model.predict(
        source=image_path,
        show=False,  
        save=True,
        hide_labels=False,
        hide_conf=False,
        show_labels=True,
        show_conf=True,
        conf=0.5,
        save_txt=True,
        save_crop=False,
        line_width=2,
        project=RESULTS_FOLDER,
        name="latest_prediction"
    )

    # Define the label path for the predicted results
    label_path = os.path.join(RESULTS_FOLDER, "latest_prediction/labels", os.path.basename(image_path).replace('.png', '.txt'))
    draw_segmentation(image_path, label_path, CROP_FOLDER)

# Function to process and crop images based on segmentation masks
def draw_segmentation(image_path, label_path, output_path):
    original_image = cv2.imread(image_path)
    height, width = original_image.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    img_counter = 1
    for line in lines:
        data = line.strip().split()
        points = np.array([float(coord) for coord in data[1:]]).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        points = points.astype(np.int32)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], color=255)

        masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

        masked_indices = np.where(mask == 255)
        if masked_indices[0].size == 0:
            continue

        min_y, max_y = np.min(masked_indices[0]), np.max(masked_indices[0])
        min_x, max_x = np.min(masked_indices[1]), np.max(masked_indices[1])
        cropped_panel = masked_image[min_y:max_y + 1, min_x:max_x + 1]

        text = f'Solar_plate_{img_counter}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_color = (0, 0, 0)
        background_color = (192, 192, 192)

        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = (cropped_panel.shape[1] - text_width) // 2
        text_y = (cropped_panel.shape[0] + text_height) // 2

        text_background_x1 = text_x - 10
        text_background_y1 = text_y - text_height - 10
        text_background_x2 = text_x + text_width + 10
        text_background_y2 = text_y + 10
        cv2.rectangle(cropped_panel, (text_background_x1, text_background_y1), (text_background_x2, text_background_y2), background_color, -1)
        cv2.putText(cropped_panel, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

        crop_output_path = os.path.join(output_path, f'Solar_plate_{img_counter}.png')
        cv2.imwrite(crop_output_path, cropped_panel)

        img_counter += 1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"})

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    predict_and_crop(image_path)

    cropped_files = os.listdir(CROP_FOLDER)
    return jsonify({"success": True, "files": cropped_files})

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(CROP_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
    #app.run(debug=True)
