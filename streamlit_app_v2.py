
import streamlit as st
import json
import tempfile
import os
import time
from PIL import Image
import requests
from urllib.parse import urlparse

# Import functions from main.py
from main import (
    extract_car_info_with_gemini,
    extract_additional_details_with_gemini,
    setup_faiss_vector_search,
    search_vector_database,
    generate_ikman_form_submission_json,
    generate_vehicle_description_with_gemini,
    load_env_vars,
    get_preliminary_query,
    cost_tracker,
)

# Load environment variables
load_env_vars()

def initialize_session_state():
    """Initialize session state variables."""
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None
    if 'ad_info' not in st.session_state:
        st.session_state.ad_info = None
    if 'processing_result' not in st.session_state:
        st.session_state.processing_result = None
    if 'vector_dataset' not in st.session_state:
        st.session_state.vector_dataset = None
    if 'faiss_data' not in st.session_state:
        st.session_state.faiss_data = None
    if 'cost_analysis' not in st.session_state:
        st.session_state.cost_analysis = {
            'api_calls': 0,
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'estimated_cost_usd': 0.0
        }

def extract_slug_from_ikman_url(url):
    """Extract the ad slug from an ikman.lk URL."""
    try:
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        if 'ad' in path_parts:
            ad_index = path_parts.index('ad')
            if ad_index + 1 < len(path_parts):
                return path_parts[ad_index + 1]
    except Exception as e:
        st.error(f"Error extracting slug from URL: {e}")
    return None

def fetch_ikman_ad_data(slug):
    """Fetch ad data from ikman.lk API."""
    try:
        api_url = f"https://api.ikman.lk/v1/ads/{slug}"
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching ad data: {e}")
    except json.JSONDecodeError:
        st.error("Error parsing ad data.")
    return None

def download_image_from_url(image_url, temp_dir, image_id=None):
    """Download image from URL and save to temp directory."""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        if image_id:
            filename = f"{image_id}.jpg"
        else:
            filename = image_url.split('/')[-1]
            if not filename or '.' not in filename:
                filename = f"image_{hash(image_url)}.jpg"

        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return file_path
    except Exception as e:
        st.error(f"Error downloading image {image_url}: {e}")
    return None

def extract_images_from_ikman_ad(ad_url):
    """Extract images from ikman.lk ad URL."""
    slug = extract_slug_from_ikman_url(ad_url)
    if not slug:
        st.error("Could not extract ad slug from URL.")
        return None, None

    ad_data = fetch_ikman_ad_data(slug)
    if not ad_data or 'ad' not in ad_data:
        st.error("Could not fetch ad data.")
        return None, None

    ad_info = ad_data['ad']
    image_urls = []
    image_ids = []
    if 'images' in ad_info and isinstance(ad_info.get('images'), dict) and 'ids' in ad_info['images']:
        base_uri = ad_info['images'].get('base_uri', 'https://i.ikman-st.com')
        image_ids = ad_info['images']['ids']
        for img_id in image_ids:
            image_urls.append(f"{base_uri}/{img_id}/620/466/fitted.jpg")

    if not image_urls:
        st.error("No images found in the ad.")
        return None, None

    temp_dir = tempfile.mkdtemp()
    downloaded_files = []
    with st.spinner(f"Downloading {len(image_urls)} images..."):
        for i, url in enumerate(image_urls):
            img_id = image_ids[i]
            file_path = download_image_from_url(url, temp_dir, image_id=img_id)
            if file_path:
                downloaded_files.append(file_path)

    if not downloaded_files:
        st.error("Failed to download any images.")
        return None, None

    return downloaded_files, ad_info

def render_home_page():
    """Render the home page for selecting upload method."""
    st.title("🚗 AI Car Analysis")
    st.markdown("Upload your car images or extract from an ikman.lk ad for instant analysis.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📁 Upload Images", use_container_width=True):
            st.session_state.page = 'upload_images'
            st.rerun()

    with col2:
        if st.button("🔗 Extract from ikman.lk", use_container_width=True):
            st.session_state.page = 'extract_url'
            st.rerun()
    


def render_upload_images_page():
    """Render the page for uploading images."""
    st.header("📁 Upload Your Own Images")
    uploaded_files = st.file_uploader(
        "Choose car images...",
        type=['jpg', 'jpeg', 'png', 'webp'],
        accept_multiple_files=True,
        help="Upload multiple images of the same car."
    )

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.success(f"{len(uploaded_files)} images uploaded successfully!")
        st.header("🖼️ Image Preview")
        cols = st.columns(len(uploaded_files))
        for i, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            cols[i].image(image, caption=f"Image {i+1}", width=150)

        if st.button("🚀 Proceed to Analysis", use_container_width=True):
            st.session_state.page = 'processing'
            st.rerun()

def render_extract_url_page():
    """Render the page for extracting images from a URL."""
    st.header("🔗 Extract from ikman.lk Ad")
    ad_url = st.text_input(
        "Enter ikman.lk ad URL:",
        placeholder="https://ikman.lk/en/ad/toyota-axio-non-hybrid-2017-for-sale-colombo-2"
    )

    if st.button("🔍 Extract Images", use_container_width=True):
        if ad_url:
            downloaded_files, ad_info = extract_images_from_ikman_ad(ad_url)
            if downloaded_files:
                st.session_state.uploaded_files = downloaded_files
                st.session_state.ad_info = ad_info
                st.rerun()
        else:
            st.warning("Please enter a URL.")

    if st.session_state.uploaded_files:
        st.success(f"{len(st.session_state.uploaded_files)} images extracted successfully!")
        st.header("🖼️ Image Preview")
        
        downloaded_files = st.session_state.uploaded_files
        if all(isinstance(p, str) for p in downloaded_files):
            cols = st.columns(len(downloaded_files))
            for i, file_path in enumerate(downloaded_files):
                try:
                    image = Image.open(file_path)
                    cols[i].image(image, caption=f"Image {i+1}", width=150)
                except Exception as e:
                    cols[i].error(f"Error loading image {i+1}")

        if st.button("🚀 Proceed to Analysis", use_container_width=True, key="proceed_from_url"):
            st.session_state.page = 'processing'
            st.rerun()

def render_processing_page():
    """Render the processing page with two-step extraction."""
    st.header("🔄 AI Analysis in Progress")
    
    # Reset the global cost tracker for this session
    cost_tracker.total_input_tokens = 0
    cost_tracker.total_output_tokens = 0
    cost_tracker.total_images = 0
    cost_tracker.total_cost = 0.0
    cost_tracker.requests = []
    
    st.write("🔄 **Session Reset**: Using fresh cost tracker")
    st.markdown("Processing your car images with advanced AI technology. Please wait...")

    if not st.session_state.vector_dataset:
        with st.spinner("Loading AI models..."):
            st.session_state.vector_dataset = load_vector_dataset()
            try:
                st.session_state.faiss_data = setup_faiss_vector_search(st.session_state.vector_dataset)
            except (FileNotFoundError, RuntimeError) as e:
                st.error("❌ Vector index setup failed!")
                st.info("💡 This might be the first run. The system will create the index automatically.")
                st.info("⏳ Please wait while the AI models are being prepared...")
                
                # Show a progress bar for the setup
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Try to create the index
                    from create_vector_index import create_vector_index
                    status_text.text("🔧 Creating vector index...")
                    progress_bar.progress(25)
                    
                    success = create_vector_index("car_vector_dataset.json", ".")
                    progress_bar.progress(75)
                    
                    if success:
                        status_text.text("✅ Loading created index...")
                        st.session_state.faiss_data = setup_faiss_vector_search(st.session_state.vector_dataset)
                        progress_bar.progress(100)
                        status_text.text("✅ Setup complete!")
                        st.success("🎉 AI models are ready! You can now process images.")
                    else:
                        st.error("❌ Failed to create vector index")
                        st.stop()
                        
                except Exception as setup_error:
                    st.error(f"❌ Setup failed: {setup_error}")
                    st.info("💡 Please ensure all dependencies are installed:")
                    st.code("pip install -r requirements.txt")
                    st.stop()

    first_file_path = st.session_state.uploaded_files[0]
    if not isinstance(first_file_path, str):
        first_file_path = save_uploaded_file(first_file_path)

    with st.spinner("Step 1 of 3: Getting preliminary query from image..."):
        preliminary_result = get_preliminary_query(first_file_path)
        if preliminary_result and isinstance(preliminary_result, dict):
            preliminary_query = preliminary_result['query']
            token_usage = preliminary_result['token_usage']
            # Track actual token usage (will be added to total later)
            st.session_state.preliminary_tokens = token_usage
            st.write(f"🔍 Preliminary query: **{preliminary_query}**")
        elif preliminary_result:
            # Fallback for old format
            preliminary_query = preliminary_result
            st.session_state.preliminary_tokens = {'input_tokens': 500, 'output_tokens': 50}  # Estimate
            st.write(f"🔍 Preliminary query: **{preliminary_query}**")
        else:
            st.write("⚠️ Could not determine a preliminary query from the image.")
            preliminary_query = None
            st.session_state.preliminary_tokens = {'input_tokens': 0, 'output_tokens': 0}

    with st.spinner("Step 2 of 3: Performing vector search..."):
        if preliminary_query:
            vector_search_results = search_vector_database(preliminary_query, st.session_state.faiss_data, top_k=5)
            if vector_search_results:
                st.write(f"✅ Found {len(vector_search_results)} potential matches.")
            else:
                st.write("⚠️ No matches found in the vector database.")
        else:
            vector_search_results = None

    # Step 3: Extract car information from images in parallel
    st.write("🔍 **Step 3 of 3: Extracting car information from images (parallel processing)...**")
    
    total_files = len(st.session_state.uploaded_files)
    st.info(f"🚀 **Processing {total_files} image(s) in parallel** - This will be much faster!")
    
    # Prepare files and schemas
    file_paths = []
    filenames = []
    
    for i, file in enumerate(st.session_state.uploaded_files, 1):
        if isinstance(file, str):
            filename = os.path.basename(file)
            file_paths.append(file)
        else:
            filename = file.name
            file_path = save_uploaded_file(file)
            if file_path:
                file_paths.append(file_path)
            else:
                st.error(f"❌ Failed to save {filename}")
                continue
        filenames.append(filename)
    
    if not file_paths:
        st.error("❌ No valid files to process")
        return
    
    # Ensure vector dataset and faiss data are loaded
    if not st.session_state.vector_dataset or not st.session_state.faiss_data:
        st.error("❌ Vector dataset or FAISS data not properly loaded!")
        st.stop()
    
    # Function to process a single image
    def process_single_image(file_path, filename, vector_dataset, faiss_data):
        """Process a single image and return results"""
        try:
            # Add small delay to avoid overwhelming the API
            import time
            time.sleep(1)  # 1 second delay between requests
            
            # Step 1: Extract brand/model (simple, no constraints)
            extracted_data = extract_car_info_with_gemini(file_path)
            
            if 'error' in extracted_data:
                return {
                    'filename': filename,
                    'extracted_data': extracted_data,
                    'success': False
                }
            
            # Step 2: Vector search to find matching car
            query = f"{extracted_data.get('brand', '')} {extracted_data.get('model', '')}".strip()
            vector_results = search_vector_database(query, faiss_data, top_k=5)
            
            # Step 3: Extract additional details if we have a match
            additional_data = {}
            if vector_results:
                best_match = vector_results[0]
                additional_data = extract_additional_details_with_gemini(file_path, best_match, vector_dataset)
            
            # Combine the data
            combined_data = {**extracted_data, **additional_data}
            
            return {
                'filename': filename,
                'extracted_data': combined_data,
                'vector_results': vector_results,
                'success': 'error' not in combined_data
            }
        except Exception as e:
            return {
                'filename': filename,
                'extracted_data': {'error': str(e)},
                'success': False
            }
    
    # Process images in parallel
    st.write("🤖 **Starting parallel analysis...**")
    
    import concurrent.futures
    
    # Configure parallel processing - reduce workers to avoid rate limiting
    max_workers = min(3, total_files)  # Reduced from 5 to 3
    st.write(f"⚙️ **Parallel workers**: {max_workers} (reduced to avoid API timeouts)")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        st.write("📤 **Submitting tasks to parallel workers...**")
        future_to_file = {
            executor.submit(process_single_image, file_path, filename, st.session_state.vector_dataset, st.session_state.faiss_data): filename
            for file_path, filename in zip(file_paths, filenames)
        }
        
        st.write(f"✅ **Submitted {len(future_to_file)} tasks** - Waiting for completion...")
        
        # Track completed tasks
        completed = 0
        extraction_results = []
        
        # Process completed futures
        for future in concurrent.futures.as_completed(future_to_file):
            completed += 1
            result = future.result()
            
            # Show result for this image
            if result['success']:
                brand = result['extracted_data'].get('brand', 'Unknown')
                model = result['extracted_data'].get('model', 'Unknown')
                confidence = result['extracted_data'].get('confidence', 0)
                st.success(f"✅ **{result['filename']}**: {brand} {model} (confidence: {confidence:.2f})")
            else:
                st.error(f"❌ **{result['filename']}**: {result['extracted_data'].get('error', 'Unknown error')}")
            
            extraction_results.append({'extracted_data': result['extracted_data'], 'success': result['success']})
    
    # Clean up temporary files
    for file_path in file_paths:
        if file_path not in st.session_state.uploaded_files:  # Only delete if it was a temp file
            try:
                os.unlink(file_path)
            except:
                pass
    
    st.success(f"🎉 **Parallel processing complete!**")

    # Get cost analysis from the global cost tracker (no manual calculations)
    cost_summary = cost_tracker.get_cost_summary()
    st.session_state.cost_analysis = {
        'api_calls': cost_summary['total_requests'],
        'total_tokens': cost_summary['total_input_tokens'] + cost_summary['total_output_tokens'],
        'input_tokens': cost_summary['total_input_tokens'],
        'output_tokens': cost_summary['total_output_tokens'],
        'estimated_cost_usd': cost_summary['total_cost_usd']
    }
    
    successful_images = sum(1 for result in extraction_results if result.get('success', False))
    total_images = len(extraction_results)
    
    st.info(f"📊 **Cost Summary**: Processed {total_images} images ({successful_images} successful, {total_images-successful_images} failed)")
    st.info(f"🔍 **Token Usage**: {cost_summary['total_input_tokens']:,} input + {cost_summary['total_output_tokens']:,} output tokens")
    st.info(f"💰 **Total Cost**: ${cost_summary['total_cost_usd']:.4f} ({cost_summary['total_requests']} API calls)")

    with st.spinner("Merging extracted data..."):
        merged_data = merge_extracted_data(extraction_results)

    with st.spinner("Generating form submission data..."):
        # Since the new prompt is more accurate, we can simplify the confidence logic
        # and directly use the extracted data.
        # The vector search is now part of the extraction, so we don't need a separate match_info
        ikman_form_json = generate_ikman_form_submission_json(
            merged_data, st.session_state.vector_dataset, {'confidence': merged_data.get('confidence', 0)}
        )
    
    # Check if we should generate vehicle description (>60% completion)
    ai_generated = ikman_form_json.get('ai_generated', {})
    filled_fields = [k for k, v in ai_generated.items() if v != "manual_fill_required"]
    total_fields = len(ai_generated)
    completion_percentage = (len(filled_fields) / total_fields * 100) if total_fields > 0 else 0
    
    if completion_percentage > 60:
        with st.spinner("Generating vehicle description..."):
            # Get the first image path for description generation
            first_image_path = None
            if file_paths:
                first_image_path = file_paths[0]
            
            description_result = generate_vehicle_description_with_gemini(
                ikman_form_json, 
                merged_data, 
                first_image_path
            )
            ikman_form_json['vehicle_description'] = description_result
            
            if description_result.get('description'):
                st.success(f"✅ Vehicle description generated! ({completion_percentage:.1f}% field completion)")
                
                # Show description generation cost info (already tracked by cost_tracker)
                desc_tokens = description_result.get('token_usage', {})
                if desc_tokens:
                    desc_input_tokens = desc_tokens.get('input_tokens', 0)
                    desc_output_tokens = desc_tokens.get('output_tokens', 0)
                    desc_cost = desc_tokens.get('cost_usd', 0)
                    st.info(f"💰 Description cost: ${desc_cost:.4f} ({desc_input_tokens:,}→{desc_output_tokens:,} tokens)")
                
                # Update cost analysis from tracker (description cost already included)
                cost_summary = cost_tracker.get_cost_summary()
                st.session_state.cost_analysis.update({
                    'api_calls': cost_summary['total_requests'],
                    'total_tokens': cost_summary['total_input_tokens'] + cost_summary['total_output_tokens'],
                    'input_tokens': cost_summary['total_input_tokens'],
                    'output_tokens': cost_summary['total_output_tokens'],
                    'estimated_cost_usd': cost_summary['total_cost_usd']
                })
            else:
                st.info(f"ℹ️ Description generation attempted but failed ({completion_percentage:.1f}% completion)")
    else:
        ikman_form_json['vehicle_description'] = {
            'description': None,
            'completion_percentage': completion_percentage,
            'reason': f'Insufficient data: Only {completion_percentage:.1f}% of fields available (need >60%)'
        }
        st.info(f"ℹ️ Skipping description generation - only {completion_percentage:.1f}% of fields completed (need >60%)")

    st.session_state.processing_result = {
        'ikman_form_submission': ikman_form_json,
        'extracted_data': merged_data,
        'field_statistics': { # Simplified stats
            'ai_prefilled': [k for k, v in ikman_form_json.get('ai_generated', {}).items() if v != "manual_fill_required"],
            'manual_required': [k for k, v in ikman_form_json.get('ai_generated', {}).items() if v == "manual_fill_required"],
            'total_fields': len(ikman_form_json.get('ai_generated', {}))
        }
    }
    st.session_state.page = 'results'
    st.rerun()

def render_results_page():
    """Render the results page."""
    st.title("🚗 Car Analysis Results")
    
    result = st.session_state.processing_result
    if result:
        # Quick summary metrics
        field_stats = result.get('field_statistics', {})
        ai_prefilled = len(field_stats.get('ai_prefilled', []))
        manual_required = len(field_stats.get('manual_required', []))
        total_fields = field_stats.get('total_fields', 0)
        success_rate = (ai_prefilled / total_fields * 100) if total_fields > 0 else 0
        
        # Top-level success indicator
        if success_rate >= 80:
            st.success(f"🎯 **Excellent Results** - {ai_prefilled} of {total_fields} fields auto-filled ({success_rate:.0f}%)")
        elif success_rate >= 60:
            st.info(f"📊 **Good Results** - {ai_prefilled} of {total_fields} fields auto-filled ({success_rate:.0f}%)")
        else:
            st.warning(f"⚠️ **Partial Results** - {ai_prefilled} of {total_fields} fields auto-filled ({success_rate:.0f}%)")

        st.markdown("---")

        # Main results
        display_json = {k: v for k, v in result['ikman_form_submission'].items() if not k.startswith('_')}
        ai_generated_data = display_json.get('ai_generated', {})
        confidence_scores = display_json.get('confidence_scores', {})
        
        # Field labels
        field_labels = {
            'condition': '🚗 Condition',
            'brand': '🏭 Brand', 
            'model': '🚙 Model',
            'model_year': '📅 Year',
            'mileage': '🛣️ Mileage',
            'engine_capacity': '⚙️ Engine Capacity',
            'fuel_type': '⛽ Fuel Type',
            'transmission': '🔧 Transmission',
            'body': '🚐 Body Type',
            'price': '💰 Price',
            'edition': '🏷️ Edition',
            'color': '🎨 Color'
        }
        
        # Simple confidence indicator
        def get_confidence_level(confidence):
            if confidence >= 0.8:
                return "High", "🟢"
            elif confidence >= 0.6:
                return "Good", "🟡"
            elif confidence > 0:
                return "Low", "🟠"
            else:
                return "Est.", "⚪"

        # Display fields in a clean two-column layout
        col1, col2 = st.columns(2)
        
        # Separate filled fields from manual fields and organize by priority
        filled_fields = []
        manual_fields = []
        
        # Define field priority order (brand and model first, then others)
        field_priority = [
            'brand', 'model', 'condition', 'model_year', 'body', 'fuel_type', 
            'transmission', 'color', 'price', 'engine_capacity', 'mileage', 'edition'
        ]
        
        # Organize fields by priority
        for field_key in field_priority:
            if field_key in ai_generated_data:
                field_value = ai_generated_data[field_key]
                if field_value and field_value != "manual_fill_required":
                    filled_fields.append((field_key, field_value))
                else:
                    manual_fields.append((field_key, field_value))
        
        # Add any remaining fields not in priority list
        for field_key, field_value in ai_generated_data.items():
            if field_key not in field_priority and field_key not in ['mileage_confidence']:
                if field_value and field_value != "manual_fill_required":
                    filled_fields.append((field_key, field_value))
                else:
                    manual_fields.append((field_key, field_value))
        
        # Combine filled fields first, then manual fields
        all_fields = filled_fields + manual_fields
        mid_point = len(all_fields) // 2
        
        with col1:
            st.subheader("📋 Vehicle Details")
            for field_key, field_value in all_fields[:mid_point]:
                label = field_labels.get(field_key, field_key.replace('_', ' ').title())
                confidence = confidence_scores.get(field_key, 0.0)
                conf_level, conf_emoji = get_confidence_level(confidence)
                
                if field_value and field_value != "manual_fill_required":
                    # Format the value
                    if field_key == 'price':
                        try:
                            price_int = int(field_value)
                            formatted_value = f"LKR {price_int:,}"
                        except (ValueError, TypeError):
                            formatted_value = f"LKR {field_value}"
                    elif field_key == 'mileage':
                        try:
                            mileage_int = int(field_value)
                            formatted_value = f"{mileage_int:,} km"
                        except (ValueError, TypeError):
                            formatted_value = f"{field_value} km"
                    elif field_key == 'engine_capacity':
                        formatted_value = f"{field_value} cc"
                    else:
                        formatted_value = str(field_value).title()
                    
                    if confidence > 0:
                        st.write(f"{label}: **{formatted_value}** {conf_emoji} `{confidence:.2f}`")
                    else:
                        st.write(f"{label}: **{formatted_value}** {conf_emoji}")
                else:
                    # Manual input required with red chip styling
                    st.markdown(f"{label}: :red[🔴 **Manual input required**]")
            
        with col2:
            st.subheader("📊 Additional Info")
            for field_key, field_value in all_fields[mid_point:]:
                label = field_labels.get(field_key, field_key.replace('_', ' ').title())
                confidence = confidence_scores.get(field_key, 0.0)
                conf_level, conf_emoji = get_confidence_level(confidence)
                
                if field_value and field_value != "manual_fill_required":
                    # Format the value
                    if field_key == 'price':
                        try:
                            price_int = int(field_value)
                            formatted_value = f"LKR {price_int:,}"
                        except (ValueError, TypeError):
                            formatted_value = f"LKR {field_value}"
                    elif field_key == 'mileage':
                        try:
                            mileage_int = int(field_value)
                            formatted_value = f"{mileage_int:,} km"
                        except (ValueError, TypeError):
                            formatted_value = f"{field_value} km"
                    elif field_key == 'engine_capacity':
                        formatted_value = f"{field_value} cc"
                    else:
                        formatted_value = str(field_value).title()
                    
                    if confidence > 0:
                        st.write(f"{label}: **{formatted_value}** {conf_emoji} `{confidence:.2f}`")
                    else:
                        st.write(f"{label}: **{formatted_value}** {conf_emoji}")
                else:
                    # Manual input required with red chip styling
                    st.markdown(f"{label}: :red[🔴 **Manual input required**]")

        # Vehicle description section
        vehicle_description = result['ikman_form_submission'].get('vehicle_description', {})
        if vehicle_description.get('description'):
            st.markdown("---")
            st.subheader("📝 Generated Vehicle Description")
            
            completion_rate = vehicle_description.get('completion_percentage', 0)
            st.info(f"Generated based on {completion_rate:.1f}% field completion")
            
            # Display the description in an expandable section
            with st.expander("View Generated Description", expanded=True):
                st.write(vehicle_description['description'])
            
            # Show token usage for description generation if available
            desc_tokens = vehicle_description.get('token_usage', {})
            if desc_tokens:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Description Input Tokens", f"{desc_tokens.get('input_tokens', 0):,}")
                with col2:
                    st.metric("Description Output Tokens", f"{desc_tokens.get('output_tokens', 0):,}")
                with col3:
                    st.metric("Description Cost", f"${desc_tokens.get('cost_usd', 0):.4f}")
        elif vehicle_description.get('reason'):
            st.markdown("---")
            st.subheader("📝 Vehicle Description")
            completion_rate = vehicle_description.get('completion_percentage', 0)
            st.warning(f"Description not generated: {vehicle_description['reason']}")

        # Detailed API cost breakdown
        st.markdown("---")
        st.subheader("💰 API Cost Breakdown")
        
        cost_data = st.session_state.cost_analysis
        
        # Main cost metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("API Calls", cost_data.get('api_calls', 0))
        with col2:
            st.metric("Input Tokens", f"{cost_data.get('input_tokens', 0):,}")
        with col3:
            st.metric("Output Tokens", f"{cost_data.get('output_tokens', 0):,}")
        with col4:
            estimated_cost = cost_data.get('estimated_cost_usd', 0)
            st.metric("Total Cost", f"${estimated_cost:.4f}")

        # Summary stats
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Auto-filled", f"{ai_prefilled}/{total_fields}")
        with col2:
            st.metric("⚠️ Manual review", manual_required)
        with col3:
            if total_fields > 0:
                efficiency = f"{(ai_prefilled/total_fields)*100:.0f}%"
                st.metric("🎯 Efficiency", efficiency)

        # Actions
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔄 Analyze Another", type="primary", use_container_width=True):
                st.session_state.page = 'upload'
                st.rerun()
        
        with col2:
            if st.button("📋 Copy JSON", use_container_width=True):
                st.code(json.dumps(display_json, indent=2), language='json')
        
        with col3:
            if st.button("🚀 Submit", use_container_width=True):
                st.success("Ready for ikman.lk submission!")

    else:
        st.error("❌ No results found. Please process an image first.")
        if st.button("⬅️ Back to Upload"):
            st.session_state.page = 'upload'
            st.rerun()

def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary location."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file {uploaded_file.name}: {e}")
        return None

def load_vector_dataset():
    """Load the vector dataset."""
    dataset_path = 'car_vector_dataset.json'
    if not os.path.exists(dataset_path):
        st.error(f"Vector dataset not found at {dataset_path}")
        return None
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading vector dataset: {e}")
        return None

def create_form_schema(vector_dataset, vector_search_results=None):
    """Create a simplified form schema for Gemini."""
    # Only include relevant brands from vector search, not all 19k+ brands
    if vector_search_results and len(vector_search_results) > 0:
        # Get unique brands from top search results
        relevant_brands = list(set([
            entry['brand_label'] for entry in vector_search_results 
            if 'brand_label' in entry
        ]))
        # Limit to top 20 brands to keep prompt size reasonable
        relevant_brands = relevant_brands[:20]
    else:
        # Fallback to a small list of common brands
        relevant_brands = ['Toyota', 'Honda', 'BMW', 'Mercedes-Benz', 'Audi', 'Volkswagen', 'Ford', 'Nissan', 'Mazda', 'Hyundai']
    
    return {
        'brands': relevant_brands,
        'conditions': [v['label'] for v in vector_dataset['form_field_mappings']['condition']['values']],
        'body_types': [v['label'] for v in vector_dataset['form_field_mappings']['body']['values']],
        'fuel_types': [v['label'] for v in vector_dataset['form_field_mappings']['fuel_type']['values']]
    }

# Removed manual cost calculation function - using cost_tracker only

def merge_extracted_data(extraction_results):
    """Merge extraction results from multiple images."""
    if not extraction_results:
        return {}
    if len(extraction_results) == 1:
        return extraction_results[0]['extracted_data']

    # Simple merge: prioritize the first valid result
    for result in extraction_results:
        if result.get('extracted_data') and 'error' not in result['extracted_data']:
            return result['extracted_data']
    return {}

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="AI Car Autofill Service", layout="wide")
    initialize_session_state()

    pages = {
        'home': render_home_page,
        'upload_images': render_upload_images_page,
        'extract_url': render_extract_url_page,
        'processing': render_processing_page,
        'results': render_results_page,
    }

    # Get the current page function from the session state
    page_function = pages.get(st.session_state.page, render_home_page)
    page_function()

if __name__ == "__main__":
    main()
