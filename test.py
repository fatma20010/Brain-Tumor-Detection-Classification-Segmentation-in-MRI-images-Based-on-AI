from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import os
import torch
import joblib
import tensorflow as tf
import numpy as np
from torchvision import transforms
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import label, binary_dilation
from skimage.transform import resize
from together import Together
from gtts import gTTS
import playsound
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model  # Ensure TensorFlow is installed

app = Flask(__name__)
CORS(app)

# Together AI API Key for tumor analysis
TOGETHER_API_KEY = "5a5d3ff7a2fbae72418501e22ced7935f285982c800882c7ba03e2e44e999025"
client = Together(api_key=TOGETHER_API_KEY)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load models for tumor analysis
print("Loading tumor analysis models...")
tumor_detection_model = tf.keras.models.load_model("C:/Users/MSI/OneDrive/Desktop/PFA2F/resnet50_tumor_classifierbin.h5")
tumor_type_model = tf.keras.models.load_model("C:/Users/MSI/OneDrive/Desktop/PFA2F/tumor_type_classifier_densenet.h5")
from PFA2 import ImprovedUNet3D

tumor_segmentation_model = torch.load(
    "C:/Users/MSI/OneDrive/Desktop/PFA2F/best_brats_model_dice.pt",
    map_location=torch.device("cpu"),
    weights_only=False
)
tumor_segmentation_model.eval()

survival_model = joblib.load("C:/Users/MSI/OneDrive/Desktop/PFA2F/best_xgb_model.joblib")
print("Tumor analysis models loaded successfully.")

# Load resources for chatbot
print("Loading chatbot resources...")
lemmatizer = WordNetLemmatizer()
intents = json.load(open('intents.json', encoding='utf-8'))
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
chatbot_model = load_model('chatbot_model10.h5')
print("Chatbot resources loaded successfully.")

# Tumor Analysis Functions
def convert_to_json_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

def preprocess_image_for_detection(image_path):
    file_ext = os.path.splitext(image_path)[1].lower()
    if file_ext == '.nii' or file_ext == '.nii.gz':
        img = nib.load(image_path).get_fdata()
        mid_slice = img[:, :, img.shape[2] // 2]
        window_center = np.percentile(mid_slice, 99)
        window_width = window_center * 2
        window_min = max(0, window_center - window_width/2)
        window_max = window_center + window_width/2
        mid_slice = np.clip(mid_slice, window_min, window_max)
        mid_slice = (mid_slice - window_min) / (window_max - window_min + 1e-6)
        img_rgb = np.stack([mid_slice] * 3, axis=-1)
    else:
        img = Image.open(image_path).convert('RGB')
        img_rgb = np.array(img) / 255.0

    image = Image.fromarray(np.uint8(img_rgb * 255))
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.permute(0, 2, 3, 1).numpy()

def preprocess_image_for_segmentation(image_path):
    if not (image_path.endswith('.nii') or image_path.endswith('.nii.gz')):
        raise ValueError("Segmentation for glioma requires a 3D NIfTI file (.nii or .nii.gz)")
    img = nib.load(image_path).get_fdata()
    original_volume = img.copy()
    img = np.stack([img] * 4, axis=0)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
    img_tensor = torch.tensor(img).float()
    img_tensor = img_tensor.permute(0, 3, 1, 2)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, original_volume

def extract_features_from_mask(segmentation_tensor, original_volume):
    segmentation = segmentation_tensor.squeeze(0).cpu()
    pred_mask = torch.argmax(segmentation, dim=0).numpy()
    original_shape = original_volume.shape
    pred_mask_resized = resize(pred_mask, (original_shape[2], original_shape[0], original_shape[1]), 
                              order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    pred_mask_resized = np.transpose(pred_mask_resized, (1, 2, 0))
    labeled_mask, num_features = label(pred_mask_resized > 0)
    if num_features == 0:
        return None
    t1_3d_tumor_volume = np.sum(pred_mask_resized > 0)
    tumor_region = original_volume[pred_mask_resized > 0]
    t1_3d_max_intensity = np.max(tumor_region) if tumor_region.size > 0 else 0
    mid_slice = pred_mask_resized[:, :, pred_mask_resized.shape[2] // 2]
    contours, _ = cv2.findContours((mid_slice > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        t1_3d_major_axis_length = MA
        t1_3d_minor_axis_length = ma
    else:
        t1_3d_major_axis_length = 0
        t1_3d_minor_axis_length = 0
    t1_3d_area = np.sum(mid_slice > 0)
    t1_3d_extent = t1_3d_area / (t1_3d_major_axis_length * t1_3d_minor_axis_length) if t1_3d_major_axis_length * t1_3d_minor_axis_length > 0 else 0
    binary_mask = pred_mask_resized > 0
    dilated = binary_dilation(binary_mask)
    surface = np.sum(binary_mask & ~dilated)
    t1_3d_surface_to_volume_ratio = surface / t1_3d_tumor_volume if t1_3d_tumor_volume > 0 else 0
    t1_3d_glcm_contrast = np.var(mid_slice[mid_slice > 0]) if np.any(mid_slice > 0) else 0
    t1_3d_mean_intensity = np.mean(tumor_region) if tumor_region.size > 0 else 0
    areas = [np.sum(pred_mask_resized[:, :, i] > 0) for i in range(pred_mask_resized.shape[2])]
    t1_2d_area_median = np.median(areas) if areas else 0
    return {
        "t1_3d_tumor_volume": float(t1_3d_tumor_volume),
        "t1_3d_max_intensity": float(t1_3d_max_intensity),
        "t1_3d_major_axis_length": float(t1_3d_major_axis_length),
        "t1_3d_area": float(t1_3d_area),
        "t1_3d_minor_axis_length": float(t1_3d_minor_axis_length),
        "t1_3d_extent": float(t1_3d_extent),
        "t1_3d_surface_to_volume_ratio": float(t1_3d_surface_to_volume_ratio),
        "t1_3d_glcm_contrast": float(t1_3d_glcm_contrast),
        "t1_3d_mean_intensity": float(t1_3d_mean_intensity),
        "t1_2d_area_median": float(t1_2d_area_median)
    }

def visualize_segmentation(original_volume, segmentation_tensor, save_path="segmentation_result.png"):
    segmentation = segmentation_tensor.squeeze(0).cpu()
    pred_mask = torch.argmax(segmentation, dim=0).numpy()
    pred_mask_resized = resize(pred_mask, (original_volume.shape[2], original_volume.shape[0], original_volume.shape[1]), 
                              order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    pred_mask_resized = np.transpose(pred_mask_resized, (1, 2, 0))
    depth = pred_mask_resized.shape[2]
    slice_idx = depth // 2
    original_slice = original_volume[:, :, slice_idx]
    original_slice = cv2.normalize(original_slice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    original_bgr = cv2.cvtColor(original_slice, cv2.COLOR_GRAY2BGR)
    mask_slice = pred_mask_resized[:, :, slice_idx]
    green_mask = np.zeros_like(original_bgr)
    green_mask[mask_slice > 0] = [0, 255, 0]
    blended = cv2.addWeighted(original_bgr, 0.7, green_mask, 0.3, 0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_slice, cmap='gray')
    axes[0].set_title("Original MRI Slice")
    axes[0].axis('off')
    axes[1].imshow(blended)
    axes[1].set_title("Segmentation Overlay (Tumor in Green)")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Segmentation visualization saved to {save_path}")
    plt.close()

def generate_patient_report(result, patient_info=None):
    if "Diagnosis" in result and result["Diagnosis"] == "No Tumor Detected":
        diagnosis = "No tumor detected."
        details = ""
        survival = "No survival prediction needed (no tumor detected)."
    else:
        diagnosis = f"Tumor detected: {result['Tumor Type']}."
        survival = f"Predicted survival days: {result['Predicted Survival Days']}" if result['Predicted Survival Days'] else "Survival prediction unavailable."
        if patient_info:
            details = "Key tumor features:\n"
            for key, value in patient_info.items():
                details += f"  - {key.replace('t1_3d_', '').replace('_', ' ').title()}: {value:.2f}\n"
        else:
            details = "No detailed tumor features available."
    
    prompt = f"""
    Generate a concise patient report in English based on the following analysis results:
    - Diagnosis: {diagnosis}
    - Survival: {survival}
    - Details: {details}
    Format the report as a short paragraph summarizing the patient's state.
    """
    response = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=[
            {"role": "system", "content": "You are a medical assistant tasked with generating concise patient reports based on tumor analysis data."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

def text_to_speech(report, output_file="patient_report.mp3"):
    tts = gTTS(text=report, lang='en', slow=False)
    tts.save(output_file)
    print(f"🎙️ Audio report saved to {output_file}")

def tumor_analysis_pipeline(image_path, nifti_path=None, manual_features=None):
    print(f"\n🔄 Processing image: {image_path}")
    try:
        image_np = preprocess_image_for_detection(image_path)
    except Exception as e:
        print(f"⚠️ Error preprocessing image: {e}")
        return {"Diagnosis": "Error Processing Image"}, None

    tumor_prob = tumor_detection_model.predict(image_np)[0, 0]
    print(f"🔍 Initial tumor probability: {tumor_prob:.4f}")
    
    tumor_type_probs = tumor_type_model.predict(image_np)[0]
    tumor_types = ["Glioma", "Meningioma", "Pituitary"]
    
    sorted_indices = np.argsort(tumor_type_probs)[::-1]
    probs_sorted = tumor_type_probs[sorted_indices]
    max_type_prob = probs_sorted[0]
    second_type_prob = probs_sorted[1]
    third_type_prob = probs_sorted[2]
    max_type_idx = sorted_indices[0]
    
    type_margin = max_type_prob - second_type_prob
    second_margin = second_type_prob - third_type_prob
    probability_distribution = tumor_type_probs / np.sum(tumor_type_probs)
    entropy = -np.sum(probability_distribution * np.log2(probability_distribution + 1e-10))
    
    print("Tumor Type Probabilities:")
    for t, p in zip(tumor_types, tumor_type_probs):
        print(f"  {t}: {p:.4f}")
    
    has_tumor = tumor_prob > 0
    tumor_type = tumor_types[max_type_idx]
    
    if max_type_prob > 0.6:
        has_tumor = True
    else:
        glioma_prob = tumor_type_probs[0]
        meningioma_prob = tumor_type_probs[1]
        pituitary_prob = tumor_type_probs[2]
        
        if tumor_type == "Meningioma":
            has_tumor = has_tumor and (
                (meningioma_prob > 0.4 and type_margin > 0.15) or
                (meningioma_prob > 0.5) or
                (meningioma_prob > 0.35 and type_margin > 0.25) or
                (meningioma_prob > glioma_prob and meningioma_prob > pituitary_prob * 1.5)
            )
            if glioma_prob > 0.6 or pituitary_prob > 0.55:
                has_tumor = False
        elif tumor_type == "Glioma":
            has_tumor = has_tumor and (
                (max_type_prob > 0.5) or
                (max_type_prob > 0.4 and type_margin > 0.2 and glioma_prob > meningioma_prob * 1.2)
            )
        else:
            has_tumor = has_tumor and (
                (max_type_prob > 0.5 and type_margin > 0.2) or
                (max_type_prob > 0.6 and pituitary_prob > meningioma_prob * 1.3)
            )

    if not has_tumor:
        print("No tumor detected (probability too low or type classification uncertain)")
        print(f"   Entropy: {entropy:.4f}")
        print(f"   Margin: {type_margin:.4f}")
        return {"Diagnosis": "No Tumor Detected"}, None

    print(f"✅ Tumor Type Detected: {tumor_type}")
    print(f"   Confidence: {max_type_prob:.4f}")
    print(f"   Margin: {type_margin:.4f}")
    print(f"   Entropy: {entropy:.4f}")
    
    patient_info = None
    segmentation_image = None
    predicted_survival = None
    is_nifti = image_path.endswith('.nii') or image_path.endswith('.nii.gz')
    
    if tumor_type == "Glioma":
        if is_nifti:
            try:
                image_tensor, original_volume = preprocess_image_for_segmentation(image_path)
                with torch.no_grad():
                    segmentation_result = tumor_segmentation_model(image_tensor)
                print("📌 Segmentation performed.")
                visualize_segmentation(original_volume, segmentation_result)
                patient_info = extract_features_from_mask(segmentation_result, original_volume)
                segmentation_image = "/get_segmentation_image"
                if patient_info:
                    print("📏 Features extracted from segmentation mask:")
                    for key, value in patient_info.items():
                        print(f"  {key}: {value:.4f}")
                else:
                    print("⚠️ Segmentation found no tumor.")
            except ValueError as e:
                print(f"⚠️ {e}. Skipping segmentation.")
            except Exception as e:
                print(f"⚠️ Error during segmentation: {e}. Skipping segmentation.")
        elif nifti_path:
            try:
                image_tensor, original_volume = preprocess_image_for_segmentation(nifti_path)
                with torch.no_grad():
                    segmentation_result = tumor_segmentation_model(image_tensor)
                print("📌 Segmentation performed with provided NIfTI file.")
                visualize_segmentation(original_volume, segmentation_result)
                patient_info = extract_features_from_mask(segmentation_result, original_volume)
                segmentation_image = "/get_segmentation_image"
                if patient_info:
                    print("📏 Features extracted from segmentation mask:")
                    for key, value in patient_info.items():
                        print(f"  {key}: {value:.4f}")
                else:
                    print("⚠️ Segmentation found no tumor.")
            except Exception as e:
                print(f"⚠️ Error with provided file: {e}. Skipping segmentation.")
        else:
            print("⚠️ Glioma detected. A 3D NIfTI file is required for segmentation.")
            return {
                "Tumor Detected": True,
                "Tumor Type": tumor_type,
                "Requires NIfTI": True
            }, None

        if patient_info:
            patient_features = np.array([[
                patient_info["t1_3d_tumor_volume"], patient_info["t1_3d_max_intensity"],
                patient_info["t1_3d_major_axis_length"], patient_info["t1_3d_area"],
                patient_info["t1_3d_minor_axis_length"], patient_info["t1_3d_extent"],
                patient_info["t1_3d_surface_to_volume_ratio"], patient_info["t1_3d_glcm_contrast"],
                patient_info["t1_3d_mean_intensity"], patient_info["t1_2d_area_median"]
            ]])
            predicted_survival = survival_model.predict(patient_features)[0]
            print(f"📅 Predicted Survival Days: {predicted_survival}")

    else:
        if manual_features:
            try:
                patient_info = manual_features
                patient_features = np.array([[
                    manual_features["t1_3d_tumor_volume"], manual_features["t1_3d_max_intensity"],
                    manual_features["t1_3d_major_axis_length"], manual_features["t1_3d_area"],
                    manual_features["t1_3d_minor_axis_length"], manual_features["t1_3d_extent"],
                    manual_features["t1_3d_surface_to_volume_ratio"], manual_features["t1_3d_glcm_contrast"],
                    manual_features["t1_3d_mean_intensity"], manual_features["t1_2d_area_median"]
                ]])
                predicted_survival = survival_model.predict(patient_features)[0]
                print(f"📅 Predicted Survival Days (from manual features): {predicted_survival}")
            except Exception as e:
                print(f"⚠️ Error processing manual features: {e}")
                patient_info = None
        else:
            print("📌 No segmentation or manual features provided for non-glioma tumor.")

    result = {
        "Tumor Detected": True,
        "Tumor Type": tumor_type,
        "Predicted Survival Days": predicted_survival,
        "Segmentation Image": segmentation_image
    }
    return result, patient_info

# Chatbot Functions
def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    # Check for medical terms BEFORE running the model prediction
    sentence_lower = sentence.lower()
    
    # Direct keyword matching for critical medical terms and question patterns
    medical_patterns = {
        # Definition patterns
        'what is brain tumor': 'brain_tumor_definition',
        'what are brain tumors': 'brain_tumor_definition',
        'what is a brain tumor': 'brain_tumor_definition',
        'brain tumor mean': 'brain_tumor_definition',
        'define brain tumor': 'brain_tumor_definition',
        'explain brain tumor': 'brain_tumor_definition',
        'tell me about brain tumor': 'brain_tumor_definition',
        'understand brain tumor': 'brain_tumor_definition',
        'what does brain tumor mean': 'brain_tumor_definition',
        
        # Rest of the patterns...
        'glioblastoma': 'glioblastoma_info',
        'meningioma': 'meningioma_info',
        'glioma': 'glioblastoma_info',
        'what types of brain': 'brain_tumor_types',
        'what kinds of brain': 'brain_tumor_types',
        'different brain tumor': 'brain_tumor_types',
        'types of brain tumor': 'brain_tumor_types',
        'kinds of brain tumor': 'brain_tumor_types',
        'categories of brain tumor': 'brain_tumor_types',
        'classify brain tumor': 'brain_tumor_types',
        'list brain tumor': 'brain_tumor_types',
        
        # Surgery patterns
        'brain tumor surgery': 'brain_tumor_surgery',
        'surgical treatment': 'brain_tumor_surgery',
        'operate': 'brain_tumor_surgery',
        'operation for': 'brain_tumor_surgery',
        'remove brain tumor': 'brain_tumor_surgery',
        'surgery for brain': 'brain_tumor_surgery',
        'brain surgery': 'brain_tumor_surgery',
        'craniotomy': 'brain_tumor_surgery',
        
        # Radiation patterns
        'radiation therapy': 'brain_tumor_radiation',
        'radiotherapy': 'brain_tumor_radiation',
        'radiation treatment': 'brain_tumor_radiation',
        'radiation for brain': 'brain_tumor_radiation',
        'gamma knife': 'brain_tumor_radiation',
        'radiation side effects': 'brain_tumor_radiation',
        'radiosurgery': 'brain_tumor_radiation',
        
        # Chemotherapy patterns
        'chemotherapy': 'brain_tumor_chemotherapy',
        'chemo': 'brain_tumor_chemotherapy',
        'chemical therapy': 'brain_tumor_chemotherapy',
        'chemotherapy side effects': 'brain_tumor_chemotherapy',
        'chemotherapy treatment': 'brain_tumor_chemotherapy',
        'temozolomide': 'brain_tumor_chemotherapy',
        
        # Support patterns
        'support for brain': 'brain_tumor_support',
        'brain tumor support': 'brain_tumor_support',
        'help with brain tumor': 'brain_tumor_support',
        'brain tumor resources': 'brain_tumor_support',
        'support group': 'brain_tumor_support',
        'counseling': 'brain_tumor_support',
        'support services': 'brain_tumor_support',
        
        # Research patterns
        'brain tumor research': 'brain_tumor_research',
        'new treatments': 'brain_tumor_research',
        'latest research': 'brain_tumor_research',
        'research advances': 'brain_tumor_research',
        'new studies': 'brain_tumor_research',
        'research developments': 'brain_tumor_research',
        'latest advances': 'brain_tumor_research',
        
        # Clinical trials patterns
        'clinical trial': 'brain_tumor_clinical_trials',
        'clinical studies': 'brain_tumor_clinical_trials',
        'experimental treatment': 'brain_tumor_clinical_trials',
        'treatment trial': 'brain_tumor_clinical_trials',
        'research trial': 'brain_tumor_clinical_trials',
        'participate in trial': 'brain_tumor_clinical_trials',
        'join trial': 'brain_tumor_clinical_trials',
        
        # Diagnostic patterns
        'how is diagnosed': 'brain_tumor_diagnosis',
        'how do they diagnose': 'brain_tumor_diagnosis',
        'how are diagnosed': 'brain_tumor_diagnosis',
        'how do doctors diagnose': 'brain_tumor_diagnosis',
        'how do you diagnose': 'brain_tumor_diagnosis',
        'what tests': 'brain_tumor_diagnosis',
        'diagnosis': 'brain_tumor_diagnosis',
        
        # Treatment patterns
        'how is treated': 'brain_tumor_treatment',
        'how do they treat': 'brain_tumor_treatment',
        'treatment options': 'brain_tumor_treatment',
        'how to treat': 'brain_tumor_treatment',
        'treatments available': 'brain_tumor_treatment',
        'therapy options': 'brain_tumor_treatment',
        
        # Symptoms patterns
        'what are the symptoms': 'brain_tumor_symptoms',
        'signs of': 'brain_tumor_symptoms',
        'symptoms': 'brain_tumor_symptoms',
        
        # Prognosis patterns
        'what is the prognosis': 'brain_tumor_prognosis',
        'survival rate': 'brain_tumor_prognosis',
        'life expectancy': 'brain_tumor_prognosis',
        
        # Prevention patterns
        'how to prevent': 'brain_tumor_prevention',
        'can you prevent': 'brain_tumor_prevention',
        'prevention': 'brain_tumor_prevention',
        
        # Causes patterns
        'what causes': 'brain_tumor_causes',
        'why do people get': 'brain_tumor_causes',
        'risk factors': 'brain_tumor_causes',
        
        # Brain Tumor Specialist patterns
        'which doctor': 'brain_tumor_specialist',
        'what kind of doctor': 'brain_tumor_specialist',
        'what type of doctor': 'brain_tumor_specialist',
        'who treats': 'brain_tumor_specialist',
        'specialist for brain': 'brain_tumor_specialist',
        'brain tumor doctor': 'brain_tumor_specialist',
        'neuro oncologist': 'brain_tumor_specialist',
        'neurosurgeon': 'brain_tumor_specialist',
        'specialist doctor': 'brain_tumor_specialist',
        'find a doctor': 'brain_tumor_specialist',
        
        # Side Effects patterns
        'side effects': 'brain_tumor_side_effects',
        'treatment effects': 'brain_tumor_side_effects',
        'after treatment': 'brain_tumor_side_effects',
        'complications': 'brain_tumor_side_effects',
        'what to expect': 'brain_tumor_side_effects',
        'problems after': 'brain_tumor_side_effects',
        'treatment reaction': 'brain_tumor_side_effects',
        
        # Coping patterns
        'how to cope': 'brain_tumor_coping',
        'coping with': 'brain_tumor_coping',
        'deal with': 'brain_tumor_coping',
        'managing': 'brain_tumor_coping',
        'handle diagnosis': 'brain_tumor_coping',
        'emotional support': 'brain_tumor_coping',
        'mental health': 'brain_tumor_coping',
        
        # Pediatric/Children patterns
        'child brain tumor': 'brain_tumor_children',
        'kids brain tumor': 'brain_tumor_children',
        'pediatric brain': 'brain_tumor_children',
        'children brain': 'brain_tumor_children',
        'young patients': 'brain_tumor_children',
        'childhood brain': 'brain_tumor_children',
        'brain tumor in children': 'brain_tumor_children',
        
        # Recurrence patterns
        'tumor come back': 'tumor_recurrence',
        'recurrence': 'tumor_recurrence',
        'return after': 'tumor_recurrence',
        'come back after': 'tumor_recurrence',
        'chance of returning': 'tumor_recurrence',
        'risk of return': 'tumor_recurrence',
        'prevent return': 'tumor_recurrence',
        
        # Genetics patterns
        'genetic': 'brain_tumor_genetics',
        'hereditary': 'brain_tumor_genetics',
        'inherited': 'brain_tumor_genetics',
        'family history': 'brain_tumor_genetics',
        'gene mutation': 'brain_tumor_genetics',
        'genetic testing': 'brain_tumor_genetics',
        'dna test': 'brain_tumor_genetics',
        'runs in family': 'brain_tumor_genetics',
        
        # Tumor Location Effects patterns
        'tumor location': 'tumor_location_effects',
        'where tumor': 'tumor_location_effects',
        'tumor position': 'tumor_location_effects',
        'affect brain': 'tumor_location_effects',
        'location impact': 'tumor_location_effects',
        'brain area': 'tumor_location_effects',
        'part of brain': 'tumor_location_effects',
        
        # Alternative Treatment patterns
        'alternative': 'alternative_treatments',
        'natural treatment': 'alternative_treatments',
        'complementary': 'alternative_treatments',
        'holistic': 'alternative_treatments',
        'herbal': 'alternative_treatments',
        'non traditional': 'alternative_treatments',
        'supplement': 'alternative_treatments',
        'diet therapy': 'alternative_treatments',
        
        # Quality of Life patterns
        'quality of life': 'quality_of_life',
        'daily life': 'quality_of_life',
        'lifestyle': 'quality_of_life',
        'normal activities': 'quality_of_life',
        'living with': 'quality_of_life',
        'day to day': 'quality_of_life',
        'work with tumor': 'quality_of_life',
        'life changes': 'quality_of_life',
        'routine': 'quality_of_life'
    }
    
    # Check for exact medical term/pattern matches first
    for pattern, intent in medical_patterns.items():
        if pattern in sentence_lower:
            return [{'intent': intent, 'probability': '0.95'}]
    
    # Handle variations of definition questions
    if any(phrase in sentence_lower for phrase in [
        'what', 'define', 'explain', 'tell me about', 'understand', 'meaning'
    ]) and any(term in sentence_lower for term in [
        'brain tumor', 'brain tumour', 'brain cancer'
    ]):
        return [{'intent': 'brain_tumor_definition', 'probability': '0.95'}]
    
    # Only run model prediction if no medical terms/patterns were found
    bow = bag_of_words(sentence)
    res = chatbot_model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.05
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    
    # Additional pattern matching for questions that might be missed
    if not return_list or float(return_list[0]['probability']) < 0.3:
        # Check for treatment-related questions
        if any(word in sentence_lower for word in ['surgery', 'operation', 'operate', 'remove']):
            return [{'intent': 'brain_tumor_surgery', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['radiation', 'radiotherapy', 'gamma']):
            return [{'intent': 'brain_tumor_radiation', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['chemo', 'chemotherapy']):
            return [{'intent': 'brain_tumor_chemotherapy', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['support', 'help', 'resources', 'groups']):
            return [{'intent': 'brain_tumor_support', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['research', 'new treatment', 'advancement']):
            return [{'intent': 'brain_tumor_research', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['trial', 'study', 'experimental']):
            return [{'intent': 'brain_tumor_clinical_trials', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['type', 'kind', 'different', 'classify']):
            return [{'intent': 'brain_tumor_types', 'probability': '0.8'}]
        
        # Additional topic checks
        elif any(word in sentence_lower for word in ['doctor', 'specialist', 'oncologist', 'surgeon']):
            return [{'intent': 'brain_tumor_specialist', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['side effect', 'complication', 'reaction']):
            return [{'intent': 'brain_tumor_side_effects', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['cope', 'deal', 'manage', 'handling']):
            return [{'intent': 'brain_tumor_coping', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['child', 'kid', 'pediatric', 'young']):
            return [{'intent': 'brain_tumor_children', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['return', 'recur', 'come back']):
            return [{'intent': 'tumor_recurrence', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['genetic', 'hereditary', 'inherited', 'dna']):
            return [{'intent': 'brain_tumor_genetics', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['location', 'position', 'area', 'part']):
            return [{'intent': 'tumor_location_effects', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['alternative', 'natural', 'holistic', 'complementary']):
            return [{'intent': 'alternative_treatments', 'probability': '0.8'}]
        elif any(word in sentence_lower for word in ['quality', 'daily life', 'lifestyle', 'living with']):
            return [{'intent': 'quality_of_life', 'probability': '0.8'}]
    
    # Filter out incorrect classifications
    if return_list and return_list[0]['intent'] in ['greeting', 'goodbye', 'thanks']:
        if any(word in sentence_lower for word in [
            'doctor', 'specialist', 'effect', 'cope', 'child', 'recur', 'genetic',
            'location', 'alternative', 'quality', 'life', 'daily', 'hereditary',
            'position', 'natural', 'holistic', 'side effect', 'pediatric'
        ]):
            # Try to determine the most relevant intent based on context
            if any(word in sentence_lower for word in ['doctor', 'specialist', 'oncologist']):
                return [{'intent': 'brain_tumor_specialist', 'probability': '0.8'}]
            elif any(word in sentence_lower for word in ['side effect', 'complication']):
                return [{'intent': 'brain_tumor_side_effects', 'probability': '0.8'}]
            elif any(word in sentence_lower for word in ['cope', 'deal', 'manage']):
                return [{'intent': 'brain_tumor_coping', 'probability': '0.8'}]
            elif any(word in sentence_lower for word in ['child', 'kid', 'pediatric']):
                return [{'intent': 'brain_tumor_children', 'probability': '0.8'}]
            elif any(word in sentence_lower for word in ['return', 'recur', 'back']):
                return [{'intent': 'tumor_recurrence', 'probability': '0.8'}]
            elif any(word in sentence_lower for word in ['genetic', 'hereditary', 'dna']):
                return [{'intent': 'brain_tumor_genetics', 'probability': '0.8'}]
            elif any(word in sentence_lower for word in ['location', 'area', 'part']):
                return [{'intent': 'tumor_location_effects', 'probability': '0.8'}]
            elif any(word in sentence_lower for word in ['alternative', 'natural', 'holistic']):
                return [{'intent': 'alternative_treatments', 'probability': '0.8'}]
            elif any(word in sentence_lower for word in ['quality', 'daily', 'life', 'living']):
                return [{'intent': 'quality_of_life', 'probability': '0.8'}]
            
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Tumor Analysis Routes
@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    nifti_path = None
    if 'nifti_file' in request.files:
        nifti_file = request.files['nifti_file']
        if nifti_file.filename:
            nifti_upload_folder = "nifti_uploads"
            if not os.path.exists(nifti_upload_folder):
                os.makedirs(nifti_upload_folder)
            nifti_path = os.path.join(nifti_upload_folder, nifti_file.filename)
            nifti_file.save(nifti_path)

    upload_folder = "uploads"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    result, patient_info = tumor_analysis_pipeline(file_path, nifti_path)
    if "Diagnosis" in result and result["Diagnosis"] == "Error Processing Image":
        return jsonify({"error": "Error processing image"}), 500

    if "Requires NIfTI" in result:
        response = {
            "tumor_detected": True,
            "tumor_type": result["Tumor Type"],
            "requires_nifti": True
        }
        return jsonify(convert_to_json_serializable(response))

    report = generate_patient_report(result, patient_info)
    audio_path = "patient_report.mp3"
    text_to_speech(report, audio_path)

    response = {
        "tumor_detected": result.get("Diagnosis", "Tumor Detected") != "No Tumor Detected",
        "tumor_type": result.get("Tumor Type", "Unknown"),
        "survival_days": result.get("Predicted Survival Days", None),
        "segmentation_image": result.get("Segmentation Image", None),
        "report": report,
        "audio_url": "/get_audio",
        "features": patient_info
    }

    return jsonify(convert_to_json_serializable(response))

@app.route('/submit_manual_features', methods=['POST'])
def submit_manual_features():
    try:
        manual_features = {
            "t1_3d_tumor_volume": float(request.form.get('t1_3d_tumor_volume', 0)),
            "t1_3d_max_intensity": float(request.form.get('t1_3d_max_intensity', 0)),
            "t1_3d_major_axis_length": float(request.form.get('t1_3d_major_axis_length', 0)),
            "t1_3d_area": float(request.form.get('t1_3d_area', 0)),
            "t1_3d_minor_axis_length": float(request.form.get('t1_3d_minor_axis_length', 0)),
            "t1_3d_extent": float(request.form.get('t1_3d_extent', 0)),
            "t1_3d_surface_to_volume_ratio": float(request.form.get('t1_3d_surface_to_volume_ratio', 0)),
            "t1_3d_glcm_contrast": float(request.form.get('t1_3d_glcm_contrast', 0)),
            "t1_3d_mean_intensity": float(request.form.get('t1_3d_mean_intensity', 0)),
            "t1_2d_area_median": float(request.form.get('t1_2d_area_median', 0))
        }
        tumor_type = request.form.get('tumor_type', 'Unknown')

        patient_features = np.array([[
            manual_features["t1_3d_tumor_volume"], manual_features["t1_3d_max_intensity"],
            manual_features["t1_3d_major_axis_length"], manual_features["t1_3d_area"],
            manual_features["t1_3d_minor_axis_length"], manual_features["t1_3d_extent"],
            manual_features["t1_3d_surface_to_volume_ratio"], manual_features["t1_3d_glcm_contrast"],
            manual_features["t1_3d_mean_intensity"], manual_features["t1_2d_area_median"]
        ]])
        predicted_survival = survival_model.predict(patient_features)[0]

        result = {
            "Tumor Detected": True,
            "Tumor Type": tumor_type,
            "Predicted Survival Days": predicted_survival
        }

        report = generate_patient_report(result, manual_features)
        audio_path = "patient_report_manual.mp3"
        text_to_speech(report, audio_path)

        response = {
            "tumor_detected": True,
            "tumor_type": tumor_type,
            "survival_days": predicted_survival,
            "report": report,
            "audio_url": "/get_audio_manual",
            "features": manual_features
        }

        return jsonify(convert_to_json_serializable(response))
    except Exception as e:
        return jsonify({"error": f"Error processing manual features: {e}"}), 400

@app.route('/get_audio', methods=['GET'])
def get_audio():
    audio_path = "patient_report.mp3"
    if os.path.exists(audio_path):
        return send_file(audio_path, as_attachment=True)
    return jsonify({"error": "Audio file not found"}), 404

@app.route('/get_audio_manual', methods=['GET'])
def get_audio_manual():
    audio_path = "patient_report_manual.mp3"
    if os.path.exists(audio_path):
        return send_file(audio_path, as_attachment=True)
    return jsonify({"error": "Audio file not found"}), 404

@app.route('/get_segmentation_image', methods=['GET'])
def get_segmentation_image():
    image_path = "segmentation_result.png"
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    return jsonify({"error": "Segmentation image not found"}), 404

# Chatbot Routes
@app.route('/')
def api_status():
    return jsonify({"status": "API is running", "message": "Welcome to the Brain Tumor Chatbot API"})

@app.route('/predict', methods=['POST'])
def predict():
    message = request.json['message']
    print(f"User message: {message}")  # Log the input message
    ints = predict_class(message)
    print(f"Predicted intents: {ints}")  # Log the predicted intents with probabilities
    if not ints:  # If no intents are above the threshold
        print("No intents above threshold. Using fallback response.")
        return jsonify({'response': "I’m not sure how to respond to that. Can you rephrase your question?"})
    res = get_response(ints, intents)
    
    ethical_keywords = ['confidentiel', 'secret', 'données sensibles']
    if any(keyword in message.lower() for keyword in ethical_keywords):
        res = "⚠️ Considération éthique : Je ne peux pas traiter les informations sensibles. Veuillez consulter un expert humain."
    
    print(f"Selected response: {res}")  # Log the final response
    return jsonify({'response': res})
if __name__ == '__main__':
    app.run(debug=True, port=5000)