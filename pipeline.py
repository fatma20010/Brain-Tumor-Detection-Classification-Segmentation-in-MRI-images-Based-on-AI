def tumor_analysis_pipeline(image_path):
    print(f"\n🔄 Processing image: {image_path}")
    try:
        image_np = preprocess_image_for_detection(image_path)
    except Exception as e:
        print(f"⚠️ Error preprocessing image: {e}")
        return {"Diagnosis": "Error Processing Image"}, None

    # Get tumor probability from binary classifier
    tumor_prob = tumor_detection_model.predict(image_np)[0, 0]
    print(f"🔍 Initial tumor probability: {tumor_prob:.4f}")
    
    # Get tumor type probabilities
    tumor_type_probs = tumor_type_model.predict(image_np)[0]
    tumor_types = ["Glioma", "Meningioma", "Pituitary"]
    
    # Sort probabilities to get top predictions and calculate metrics
    sorted_indices = np.argsort(tumor_type_probs)[::-1]
    max_type_prob = tumor_type_probs[sorted_indices[0]]
    second_type_prob = tumor_type_probs[sorted_indices[1]]
    max_type_idx = sorted_indices[0]
    
    print("Tumor Type Probabilities:")
    for t, p in zip(tumor_types, tumor_type_probs):
        print(f"  {t}: {p:.4f}")

    # Calculate confidence metrics
    type_margin = max_type_prob - second_type_prob
    probability_distribution = tumor_type_probs / np.sum(tumor_type_probs)
    entropy = -np.sum(probability_distribution * np.log2(probability_distribution + 1e-10))
    
    # Base tumor detection on binary classifier first
    has_tumor = tumor_prob > 0

    # Then apply type-specific refinements with more emphasis on high probabilities
    tumor_type = tumor_types[max_type_idx]
    
    # New detection logic that better handles high probability cases
    if max_type_prob > 0.6:  # If any type has >60% probability, it's likely a tumor
        has_tumor = True
    elif tumor_type == "Glioma":
        # Special handling for Glioma since it's most common
        has_tumor = has_tumor and (
            (max_type_prob > 0.5) or  # Clear Glioma signal
            (max_type_prob > 0.4 and type_margin > 0.2) or  # Good margin between predictions
            (entropy < 1.2 and max_type_prob > 0.35)  # Very clear prediction pattern
        )
    elif tumor_type == "Meningioma":
        has_tumor = has_tumor and (
            (max_type_prob > 0.5 and type_margin > 0.2) or  # Clear Meningioma with good margin
            (max_type_prob > 0.65)  # Very strong Meningioma signal
        )
    else:  # Pituitary
        has_tumor = has_tumor and (
            (max_type_prob > 0.45 and type_margin > 0.2) or  # Clear Pituitary with good margin
            (max_type_prob > 0.6)  # Strong Pituitary signal
        )

    if not has_tumor:
        print("No tumor detected (probability too low or type classification uncertain)")
        print(f"   Entropy: {entropy:.4f}")
        print(f"   Margin: {type_margin:.4f}")
        return {"Diagnosis": "No Tumor Detected"}, None

    # Print detailed classification metrics
    print(f"✅ Tumor Type Detected: {tumor_type}")
    print(f"   Confidence: {max_type_prob:.4f}")
    print(f"   Margin: {type_margin:.4f}")
    print(f"   Entropy: {entropy:.4f}")
    
    # Rest of the pipeline remains unchanged
    patient_info = None