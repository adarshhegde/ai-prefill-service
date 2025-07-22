import streamlit as st
import json
import tempfile
import os
from pathlib import Path
import time
from PIL import Image
import io

# Import our functions from main.py
from main import (
    extract_car_info_with_gemini,
    setup_faiss_vector_search,
    search_vector_database,
    enhance_match_info_with_form_data,
    generate_ikman_form_submission_json,
    load_env_vars
)

# Load environment variables
load_env_vars()

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None
if 'vector_dataset' not in st.session_state:
    st.session_state.vector_dataset = None

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file {uploaded_file.name}: {str(e)}")
        return None

def validate_image_file(uploaded_file):
    """Validate uploaded image file"""
    try:
        # Check file size (50MB limit)
        file_size = len(uploaded_file.getvalue())
        if file_size == 0:
            return False, "File is empty (0 bytes)"
        
        max_size = 50 * 1024 * 1024  # 50MB
        if file_size > max_size:
            return False, f"File size ({file_size/1024/1024:.1f}MB) exceeds 50MB limit"
        
        # Reset file pointer for validation
        uploaded_file.seek(0)
        
        # Try to open with PIL to validate it's a real image
        try:
            # Use BytesIO to create a file-like object from the uploaded file bytes
            from io import BytesIO
            image_bytes = uploaded_file.getvalue()
            image_stream = BytesIO(image_bytes)
            
            with Image.open(image_stream) as img:
                # Basic validation - ensure it has dimensions
                if img.size[0] == 0 or img.size[1] == 0:
                    return False, "Invalid image dimensions"
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
        except Exception as e:
            return False, f"Cannot open as image: {str(e)}"
        
        # Reset file pointer after validation
        uploaded_file.seek(0)
        return True, "Valid image"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def safe_float_conversion(value, default=0.0):
    """Safely convert value to float"""
    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            return float(value)
        else:
            return default
    except (ValueError, TypeError):
        return default

def merge_extracted_data(extraction_results):
    """Merge extraction results from multiple images of the same car"""
    if not extraction_results:
        return {}
    
    # If only one image, return its data
    if len(extraction_results) == 1:
        return extraction_results[0]['extracted_data']
    
    merged_data = {}
    
    # Helper function to get most confident value
    def get_best_value(field_name):
        values = []
        for result in extraction_results:
            extracted = result.get('extracted_data', {})
            if field_name in extracted and extracted[field_name]:
                # Safely convert confidence to float
                confidence = safe_float_conversion(extracted.get('confidence', 0.5))
                values.append((extracted[field_name], confidence))
        
        if not values:
            return None
        
        # Return value with highest confidence, or most common if tied
        values.sort(key=lambda x: x[1], reverse=True)
        return values[0][0]
    
    # Helper function to merge lists
    def merge_lists(field_name):
        all_items = []
        for result in extraction_results:
            extracted = result.get('extracted_data', {})
            if field_name in extracted and extracted[field_name]:
                if isinstance(extracted[field_name], list):
                    all_items.extend(extracted[field_name])
                else:
                    all_items.append(extracted[field_name])
        return list(set(all_items)) if all_items else []
    
    # Helper function to merge notes/observations
    def merge_notes():
        all_notes = []
        for i, result in enumerate(extraction_results):
            extracted = result.get('extracted_data', {})
            notes = extracted.get('extraction_notes', '')
            if notes:
                all_notes.append(f"Image {i+1}: {notes}")
        return " | ".join(all_notes) if all_notes else ""
    
    # Merge core fields using best confidence
    core_fields = ['brand', 'model', 'year', 'condition', 'body_type', 'fuel_type', 'color']
    for field in core_fields:
        best_value = get_best_value(field)
        if best_value:
            merged_data[field] = best_value
    
    # Merge feature lists
    merged_data['visible_features'] = merge_lists('visible_features')
    
    # Calculate overall confidence (average of all confidences)
    confidences = []
    for result in extraction_results:
        conf = safe_float_conversion(result.get('extracted_data', {}).get('confidence', 0))
        if conf > 0:
            confidences.append(conf)
    
    merged_data['confidence'] = sum(confidences) / len(confidences) if confidences else 0.5
    
    # Merge notes
    merged_data['extraction_notes'] = merge_notes()
    
    # Add metadata about merge process
    merged_data['merge_info'] = {
        'total_images_processed': len(extraction_results),
        'successful_extractions': len([r for r in extraction_results if 'extracted_data' in r]),
        'merge_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return merged_data

def efficient_vector_search_for_merged_data(merged_extracted_data, faiss_data):
    """Perform vector search on merged data using pre-built FAISS index"""
    # Search for brand and model
    search_query = f"{merged_extracted_data.get('brand', '')} {merged_extracted_data.get('model', '')}".strip()
    
    if search_query:
        search_results = search_vector_database(search_query, faiss_data, top_k=5)
        
        # Find best matching brand-model combination
        best_match = None
        for result in search_results:
            if result['metadata']['type'] == 'brand_model':
                best_match = result
                break
        
        if not best_match:
            # Try brand-only search
            brand_query = merged_extracted_data.get('brand', '')
            if brand_query:
                brand_results = search_vector_database(brand_query, faiss_data, top_k=5)
                for result in brand_results:
                    if result['metadata']['type'] == 'brand':
                        best_match = result
                        break
        
        if best_match:
            form_data = best_match['metadata'].get('form_autofill_data', {})
            
            match_info = {
                'method': 'vector_search',
                'confidence': best_match['score'],
                'matched_text': best_match['original_text'],  # Use original text for display
                'search_query': search_query,
                'debug_info': {
                    'normalized_query': best_match.get('normalized_query', ''),
                    'matched_metadata': {
                        'brand_key': best_match['metadata'].get('brand_key'),
                        'brand_label': best_match['metadata'].get('brand_label'),
                        'model_key': best_match['metadata'].get('model_key'),
                        'model_label': best_match['metadata'].get('model_label'),
                        'type': best_match['metadata'].get('type')
                    }
                }
            }
        else:
            form_data = {}
            match_info = {'method': 'vector_search', 'confidence': 0, 'error': 'No matches found'}
    else:
        form_data = {}
        match_info = {'method': 'vector_search', 'confidence': 0, 'error': 'No search query'}
    
    return {
        'extracted_data': merged_extracted_data,
        'form_autofill': form_data,
        'match_info': match_info,
        'available_suggestions': {
            'colors': merged_extracted_data.get('color'),
            'features': merged_extracted_data.get('visible_features', []),
            'notes': merged_extracted_data.get('extraction_notes')
        }
    }

def apply_confidence_based_form_generation(ikman_form_json, extracted_data, match_info):
    """Apply confidence thresholds and mark low-confidence fields for manual filling"""
    
    # Define confidence thresholds for different field types
    field_confidence_thresholds = {
        'brand': 0.7,       # High threshold - critical field
        'model': 0.6,       # High threshold - critical field  
        'model_year': 0.5,  # Medium threshold
        'condition': 0.4,   # Lower threshold - can be estimated
        'body': 0.4,        # Lower threshold - can be estimated
        'fuel_type': 0.4,   # Lower threshold - can be estimated
        'transmission': 0.3, # Low threshold - often not visible
        'engine_capacity': 0.3, # Low threshold - estimated
        'mileage': 0.3,     # Low threshold - estimated
        'price': 0.2        # Very low threshold - always estimated
    }
    
    # Get overall extraction confidence
    overall_confidence = safe_float_conversion(extracted_data.get('confidence', 0))
    vector_search_confidence = safe_float_conversion(match_info.get('confidence', 0))
    
    # Combine confidences for field-specific assessment
    combined_confidence = (overall_confidence + vector_search_confidence) / 2
    
    enhanced_form = ikman_form_json.copy()
    field_stats = {
        'ai_prefilled': [],
        'manual_required': [],
        'total_fields': len(ikman_form_json)
    }
    
    # Check each field against its confidence threshold
    for field_key, field_value in ikman_form_json.items():
        threshold = field_confidence_thresholds.get(field_key, 0.5)  # Default threshold
        
        # Determine if field needs manual filling
        needs_manual = False
        
        # Check if field was detected with sufficient confidence
        if field_key in ['brand', 'model']:
            # For brand/model, use vector search confidence
            if vector_search_confidence < threshold:
                needs_manual = True
        else:
            # For other fields, use combined confidence
            if combined_confidence < threshold:
                needs_manual = True
        
        # Special cases for estimated fields
        if field_key in ['engine_capacity', 'mileage', 'price']:
            # These are always estimated, so check if they're reasonable
            if field_key == 'price' and isinstance(field_value, (int, float)):
                # If price seems unreasonable (too low/high), mark for manual
                if field_value < 100000 or field_value > 50000000:  # 100k to 50M LKR
                    needs_manual = True
        
        # Apply manual filling marker if needed
        if needs_manual:
            enhanced_form[field_key] = "manual_filling_required"
            field_stats['manual_required'].append(field_key)
        else:
            field_stats['ai_prefilled'].append(field_key)
    
    # Add confidence metadata
    enhanced_form['_confidence_metadata'] = {
        'overall_extraction_confidence': overall_confidence,
        'vector_search_confidence': vector_search_confidence,
        'combined_confidence': combined_confidence,
        'field_thresholds_used': field_confidence_thresholds,
        'processing_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return enhanced_form, field_stats

def process_multiple_car_images(uploaded_files, vector_dataset):
    """Process multiple images of the same car efficiently with single FAISS index"""
    st.write("## 🔄 Processing Your Car Images")
    
    # Create FAISS index once for all images (EFFICIENCY IMPROVEMENT)
    with st.spinner("🔍 Setting up AI vector search index..."):
        faiss_data = setup_faiss_vector_search(vector_dataset)
    st.success("✅ Vector search index ready")
    
    # Create form schema once for all images
    form_schema = {
        'brands': [entry['brand_label'] for entry in vector_dataset['vector_entries'] 
                  if entry['type'] in ['brand', 'brand_model']][:20],
        'conditions': [v['label'] for v in vector_dataset['form_field_mappings']['condition']['values']],
        'body_types': [v['label'] for v in vector_dataset['form_field_mappings']['body']['values']],
        'fuel_types': [v['label'] for v in vector_dataset['form_field_mappings']['fuel_type']['values']]
    }
    
    # Create progress tracking
    total_images = len(uploaded_files)
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    
    extraction_results = []
    successful_extractions = 0
    
    # Process each image (only AI extraction, no redundant vector search)
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (i + 1) / total_images
        progress_bar.progress(progress)
        status_placeholder.write(f"📸 Processing image {i+1}/{total_images}: {uploaded_file.name}")
        
        try:
            # Validate the image
            is_valid, validation_message = validate_image_file(uploaded_file)
            
            if not is_valid:
                st.warning(f"⚠️ Skipping {uploaded_file.name}: {validation_message}")
                continue
            
            # Save to temporary file
            temp_path = save_uploaded_file(uploaded_file)
            if not temp_path:
                st.error(f"❌ Failed to process {uploaded_file.name}")
                continue
            
            try:
                # Only do AI extraction (no vector search yet - efficiency improvement)
                extracted_data = extract_car_info_with_gemini(temp_path, form_schema)
                
                if 'error' in extracted_data:
                    st.error(f"❌ {uploaded_file.name}: {extracted_data['error']}")
                else:
                    # Store just the extraction result
                    extraction_results.append({'extracted_data': extracted_data})
                    successful_extractions += 1
                    
                    # Show what was extracted
                    brand = extracted_data.get('brand', 'Unknown')
                    model = extracted_data.get('model', 'Unknown') 
                    confidence = safe_float_conversion(extracted_data.get('confidence', 0))
                    st.write(f"✅ **{uploaded_file.name}**: {brand} {model} (confidence: {confidence:.2f})")
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            st.error(f"❌ {uploaded_file.name}: Unexpected error - {str(e)}")
    
    # Final progress update
    progress_bar.progress(1.0)
    status_placeholder.write("✅ All images processed!")
    
    # Calculate success rate
    success_rate = (successful_extractions / total_images * 100) if total_images > 0 else 0
    
    # Show processing summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Successfully Analyzed", successful_extractions)
    with col3:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    if successful_extractions == 0:
        st.error("❌ No images were successfully processed.")
        st.write("**Please check your images and try again.**")
        return None
    
    # Merge extraction results from all images
    st.write("### 🔗 Combining Results from All Images")
    merged_extracted_data = merge_extracted_data(extraction_results)
    
    # Display merged results
    st.write("#### 📋 Combined Analysis Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Vehicle Identification:**")
        st.write(f"🏷️ Brand: **{merged_extracted_data.get('brand', 'Unknown')}**")
        st.write(f"🚗 Model: **{merged_extracted_data.get('model', 'Unknown')}**")
        st.write(f"📅 Year: **{merged_extracted_data.get('year', 'Unknown')}**")
        st.write(f"⭐ Overall Confidence: **{merged_extracted_data.get('confidence', 0):.2f}**")
    
    with col2:
        st.write("**Vehicle Specifications:**")
        st.write(f"🏗️ Body Type: **{merged_extracted_data.get('body_type', 'Unknown')}**")
        st.write(f"⛽ Fuel Type: **{merged_extracted_data.get('fuel_type', 'Unknown')}**")
        st.write(f"🎨 Color: **{merged_extracted_data.get('color', 'Unknown')}**")
        st.write(f"🔧 Condition: **{merged_extracted_data.get('condition', 'Unknown')}**")
    
    # Now do vector search ONCE on merged data (EFFICIENCY IMPROVEMENT)
    with st.spinner("🔍 Performing vector search on combined data..."):
        matched_result = efficient_vector_search_for_merged_data(merged_extracted_data, faiss_data)
    
    st.success(f"✅ Vector search completed (confidence: {matched_result['match_info'].get('confidence', 0):.2f})")
    
    # Generate ikman form JSON with confidence-based filling
    with st.spinner("📝 Generating ikman.lk form data..."):
        enhanced_match_info = enhance_match_info_with_form_data(merged_extracted_data, vector_dataset, matched_result['match_info'])
        raw_ikman_form_json = generate_ikman_form_submission_json(merged_extracted_data, vector_dataset, matched_result['match_info'])
        
        # Apply confidence-based field handling
        final_ikman_form_json, field_stats = apply_confidence_based_form_generation(
            raw_ikman_form_json, merged_extracted_data, enhanced_match_info
        )
    
    # Create final result structure
    final_result = {
        'extracted_data': merged_extracted_data,
        'form_autofill': matched_result['form_autofill'],
        'match_info': enhanced_match_info,
        'ikman_form_submission': final_ikman_form_json,
        'field_statistics': field_stats,
        'available_suggestions': matched_result['available_suggestions'],
        'processing_summary': {
            'total_images': total_images,
            'successful_extractions': successful_extractions,
            'success_rate': success_rate
        }
    }
    
    return final_result

def load_vector_dataset():
    """Load the vector dataset"""
    dataset_path = 'car_vector_dataset.json'
    if not os.path.exists(dataset_path):
        st.error(f"Vector dataset not found at {dataset_path}")
        st.write("Please run `python main.py` first to generate the vector dataset.")
        return None
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading vector dataset: {e}")
        return None

def render_page_1_upload():
    """Page 1: Upload Images"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79, #2e8b57); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>🚗 Step 1: Upload Your Car Images</h1>
        <p style="font-size: 1.2em;">Upload multiple photos of the same car from different angles</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div style="background-color: #f8f9fa; color:black; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1f4e79; margin-bottom: 2rem;">
    <h3>📸 Photo Guidelines</h3>
    <ul>
        <li><strong>Exterior shots:</strong> Front, back, sides for brand/model identification</li>
        <li><strong>Interior photos:</strong> Dashboard, seats, gear shifter for transmission type</li>
        <li><strong>Engine bay:</strong> For engine specifications (if accessible)</li>
        <li><strong>Detail shots:</strong> Badges, emblems, or any specific features</li>
        <li><strong>Quality tips:</strong> Clear, well-lit photos work best</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose car images...",
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=True,
        help="Upload multiple images of the same car. Supports JPG, PNG, WEBP formats up to 50MB each.",
        key="file_uploader"
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        
        st.success(f"✅ {len(uploaded_files)} images uploaded successfully!")
        
        # Display thumbnails
        st.write("#### 🖼️ Preview of Uploaded Images")
        cols = st.columns(min(len(uploaded_files), 4))
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i % 4]:
                try:
                    # Use BytesIO to create a file-like object from the uploaded file bytes
                    from io import BytesIO
                    image_bytes = uploaded_file.getvalue()
                    image_stream = BytesIO(image_bytes)
                    
                    image = Image.open(image_stream)
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                    uploaded_file.seek(0)  # Reset file pointer
                except Exception as e:
                    st.write(f"📄 {uploaded_file.name} (Preview failed: {str(e)[:50]}...)")
        
        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Proceed to Analysis", type="primary", use_container_width=True):
                st.session_state.current_page = 2
                st.rerun()
    
    else:
        st.info("👆 Please upload images of your car to continue")

def render_page_2_processing():
    """Page 2: Processing Images"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79, #2e8b57); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>🔄 Step 2: AI Analysis in Progress</h1>
        <p style="font-size: 1.2em;">Processing your car images with advanced AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.uploaded_files:
        st.error("No images found. Please go back to upload images.")
        if st.button("← Back to Upload"):
            st.session_state.current_page = 1
            st.rerun()
        return
    
    # Load vector dataset if not already loaded
    if not st.session_state.vector_dataset:
        with st.spinner("Loading AI models..."):
            st.session_state.vector_dataset = load_vector_dataset()
    
    if not st.session_state.vector_dataset:
        st.error("Failed to load AI models. Please check the setup.")
        return
    
    # Process images
    if not st.session_state.processing_result:
        result = process_multiple_car_images(st.session_state.uploaded_files, st.session_state.vector_dataset)
        st.session_state.processing_result = result
    
    if st.session_state.processing_result:
        st.success("🎉 Processing completed successfully!")
        
        # Show results immediately after processing
        result = st.session_state.processing_result
        field_stats = result.get('field_statistics', {})
        ikman_json = result['ikman_form_submission']
        
        # Field Statistics
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #e8f5e8; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #2e8b57; margin-bottom: 2rem;">
        <h3>📊 Auto-Fill Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        ai_prefilled_count = len(field_stats.get('ai_prefilled', []))
        manual_required_count = len(field_stats.get('manual_required', []))
        total_fields = field_stats.get('total_fields', ai_prefilled_count + manual_required_count)
        ai_percentage = (ai_prefilled_count / total_fields * 100) if total_fields > 0 else 0
        
        with col1:
            st.metric("Total Fields", total_fields)
        with col2:
            st.metric("AI Pre-filled", ai_prefilled_count, delta=f"{ai_percentage:.0f}%")
        with col3:
            st.metric("Manual Required", manual_required_count)
        with col4:
            automation_level = "High" if ai_percentage >= 70 else "Medium" if ai_percentage >= 50 else "Low"
            st.metric("Automation Level", automation_level)
        
        # Show which fields need manual filling
        if field_stats.get('manual_required'):
            st.warning(f"⚠️ **Manual filling required for:** {', '.join(field_stats['manual_required'])}")
        
        if field_stats.get('ai_prefilled'):
            st.success(f"✅ **AI pre-filled successfully:** {', '.join(field_stats['ai_prefilled'])}")
        
        # ikman.lk Form JSON
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1f4e79; margin-bottom: 2rem;">
        <h3>📋 ikman.lk Form Submission Data</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Clean JSON for display (remove metadata)
        display_json = {k: v for k, v in ikman_json.items() if not k.startswith('_')}
        st.json(display_json)
        
        # Quick download buttons
        st.markdown("### 📥 Quick Downloads")
        col1, col2 = st.columns(2)
        
        with col1:
            json_str = json.dumps(display_json, indent=2)
            st.download_button(
                label="📋 Download ikman.lk Form JSON",
                data=json_str,
                file_name="ikman_car_form.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            full_result_str = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="📊 Download Complete Analysis",
                data=full_result_str,
                file_name="complete_car_analysis.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Navigation
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("← Back to Upload"):
                st.session_state.current_page = 1
                st.rerun()
        with col2:
            st.info("🎯 Your results are ready! You can download them above or view detailed analysis below.")
        with col3:
            if st.button("View Details →", type="primary"):
                st.session_state.current_page = 3
                st.rerun()

def render_page_3_results():
    """Page 3: Detailed Analysis and Additional Options"""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f4e79, #2e8b57); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>🔍 Step 3: Detailed Analysis</h1>
        <p style="font-size: 1.2em;">Deep dive into your car analysis results</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.processing_result:
        st.error("No results found. Please process images first.")
        if st.button("← Back to Upload"):
            st.session_state.current_page = 1
            st.rerun()
        return
    
    result = st.session_state.processing_result
    
    # Quick summary at the top
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
    <h4>📋 Quick Summary</h4>
    </div>
    """, unsafe_allow_html=True)
    
    extracted_data = result['extracted_data']
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Vehicle Identification:**")
        st.write(f"🏷️ **Brand:** {extracted_data.get('brand', 'Unknown')}")
        st.write(f"🚗 **Model:** {extracted_data.get('model', 'Unknown')}")
        st.write(f"📅 **Year:** {extracted_data.get('year', 'Unknown')}")
        st.write(f"⭐ **Confidence:** {extracted_data.get('confidence', 0):.2f}")
    
    with col2:
        st.write("**Processing Summary:**")
        summary = result['processing_summary']
        field_stats = result.get('field_statistics', {})
        st.write(f"📸 **Images:** {summary['successful_extractions']}/{summary['total_images']}")
        st.write(f"🎯 **Success Rate:** {summary['success_rate']:.1f}%")
        st.write(f"🤖 **AI Fields:** {len(field_stats.get('ai_prefilled', []))}")
        st.write(f"✋ **Manual Fields:** {len(field_stats.get('manual_required', []))}")
    
    # Detailed analysis in expandable sections
    with st.expander("🔍 AI Extraction Details", expanded=True):
        st.write("**Complete AI Analysis Results:**")
        
        # Organize extraction data in a nice table format
        extraction_details = []
        for key, value in extracted_data.items():
            if key not in ['merge_info', 'visible_features', 'extraction_notes']:
                extraction_details.append({
                    "Field": key.replace('_', ' ').title(),
                    "Detected Value": str(value) if value else "Not detected",
                    "Confidence": "High" if extracted_data.get('confidence', 0) > 0.7 else "Medium" if extracted_data.get('confidence', 0) > 0.5 else "Low"
                })
        
        if extraction_details:
            import pandas as pd
            df = pd.DataFrame(extraction_details)
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Show detected features
        if extracted_data.get('visible_features'):
            st.write("**🔧 Detected Features:**")
            features_cols = st.columns(3)
            for i, feature in enumerate(extracted_data['visible_features']):
                with features_cols[i % 3]:
                    st.write(f"• {feature}")
        
        # Show extraction notes
        if extracted_data.get('extraction_notes'):
            st.write("**📝 AI Analysis Notes:**")
            st.info(extracted_data['extraction_notes'])
    
    with st.expander("📈 Vector Search Analysis"):
        match_info = result['match_info']
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Search Performance:**")
            st.write(f"🎯 **Method:** {match_info.get('method', 'Unknown')}")
            st.write(f"📊 **Confidence:** {match_info.get('confidence', 0):.3f}")
            if 'search_query' in match_info:
                st.write(f"🔍 **Search Query:** {match_info['search_query']}")
            if 'matched_text' in match_info:
                st.write(f"✅ **Best Match:** {match_info['matched_text']}")
        
        with col2:
            st.write("**Database Information:**")
            # Show some vector database stats if available
            if st.session_state.vector_dataset:
                metadata = st.session_state.vector_dataset.get('metadata', {})
                st.write(f"📦 **Database Size:** {metadata.get('total_entries', 'Unknown'):,} entries")
                st.write(f"🏭 **Brand+Model Entries:** {metadata.get('brand_model_entries', 'Unknown'):,}")
                st.write(f"🏷️ **Brand-only Entries:** {metadata.get('brand_only_entries', 'Unknown'):,}")
    
    with st.expander("⚙️ Field Confidence Analysis"):
        field_stats = result.get('field_statistics', {})
        confidence_metadata = result['ikman_form_submission'].get('_confidence_metadata', {})
        
        st.write("**Confidence Scores:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"🧠 **AI Extraction:** {confidence_metadata.get('overall_extraction_confidence', 0):.2f}")
            st.write(f"🔍 **Vector Search:** {confidence_metadata.get('vector_search_confidence', 0):.2f}")
            st.write(f"🎯 **Combined Score:** {confidence_metadata.get('combined_confidence', 0):.2f}")
        
        with col2:
            st.write("**Field Categories:**")
            if field_stats.get('ai_prefilled'):
                st.success(f"✅ **High Confidence:** {len(field_stats['ai_prefilled'])} fields")
            if field_stats.get('manual_required'):
                st.warning(f"⚠️ **Needs Review:** {len(field_stats['manual_required'])} fields")
        
        # Show thresholds used
        thresholds = confidence_metadata.get('field_thresholds_used', {})
        if thresholds:
            st.write("**Confidence Thresholds Used:**")
            threshold_df = pd.DataFrame([
                {"Field Type": k.replace('_', ' ').title(), "Threshold": f"{v:.1f}"} 
                for k, v in thresholds.items()
            ])
            st.dataframe(threshold_df, use_container_width=True, hide_index=True)
    
    with st.expander("💡 Suggestions & Tips"):
        suggestions = result.get('available_suggestions', {})
        
        st.write("**🎨 Additional Information Detected:**")
        if suggestions.get('colors'):
            st.write(f"• **Color variations:** {suggestions['colors']}")
        if suggestions.get('features'):
            st.write(f"• **Features:** {', '.join(suggestions['features'][:5])}")
        
        st.write("\n**🔧 Tips for Better Results:**")
        st.write("• Upload clear, well-lit photos from multiple angles")
        st.write("• Include close-ups of badges and emblems for better brand/model detection")
        st.write("• Interior shots help identify transmission and fuel type")
        st.write("• Engine bay photos can improve engine capacity estimates")
        
        st.write("\n**⚠️ Manual Review Recommended For:**")
        manual_fields = field_stats.get('manual_required', [])
        if manual_fields:
            for field in manual_fields:
                st.write(f"• **{field.replace('_', ' ').title()}** - Low confidence or estimated value")
        else:
            st.success("All fields have sufficient confidence scores!")
    
    # Navigation
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Results"):
            st.session_state.current_page = 2
            st.rerun()
    with col2:
        if st.button("🔄 Start New Analysis", use_container_width=True):
            # Reset session state
            st.session_state.current_page = 1
            st.session_state.uploaded_files = None
            st.session_state.processing_result = None
            st.rerun()
    with col3:
        # Additional export option
        if st.button("📤 Export Report"):
            # Create a comprehensive report
            report_data = {
                'analysis_summary': {
                    'vehicle': f"{extracted_data.get('brand', '')} {extracted_data.get('model', '')}".strip(),
                    'confidence': extracted_data.get('confidence', 0),
                    'processing_date': confidence_metadata.get('processing_timestamp'),
                    'images_processed': summary['total_images'],
                    'success_rate': summary['success_rate']
                },
                'ikman_form_data': {k: v for k, v in result['ikman_form_submission'].items() if not k.startswith('_')},
                'field_statistics': field_stats,
                'detailed_analysis': extracted_data
            }
            
            report_str = json.dumps(report_data, indent=2, default=str)
            st.download_button(
                label="📋 Download Full Report",
                data=report_str,
                file_name=f"car_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

def main():
    st.set_page_config(
        page_title="AI Car Autofill Service", 
        page_icon="🚗",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    .stButton > button {
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Page navigation
    if st.session_state.current_page == 1:
        render_page_1_upload()
    elif st.session_state.current_page == 2:
        render_page_2_processing()
    elif st.session_state.current_page == 3:
        render_page_3_results()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <strong>AI Car Autofill Service</strong> - Automatically generates ikman.lk form data from car images<br>
        <small>Page {}/3 | Powered by Gemini Vision AI + FAISS Vector Search</small>
    </div>
    """.format(st.session_state.current_page), unsafe_allow_html=True)

if __name__ == "__main__":
    main() 