import requests
import json
import time
from urllib.parse import quote
import os
from pathlib import Path
import base64
import numpy as np

# Global quota tracking
DAILY_REQUEST_COUNT = 0
DAILY_REQUEST_LIMIT = 50  # Default fallback, will be auto-detected
MINUTE_REQUEST_LIMIT = 15  # Default fallback, will be auto-detected
REQUEST_START_TIME = time.time()
QUOTA_AUTO_DETECTED = False

def track_api_usage():
    """Track API usage to stay within quotas"""
    global DAILY_REQUEST_COUNT, DAILY_REQUEST_LIMIT
    
    DAILY_REQUEST_COUNT += 1
    remaining = DAILY_REQUEST_LIMIT - DAILY_REQUEST_COUNT
    
    print(f"📊 API Usage: {DAILY_REQUEST_COUNT}/{DAILY_REQUEST_LIMIT} requests ({(DAILY_REQUEST_COUNT/DAILY_REQUEST_LIMIT)*100:.1f}%)")
    
    if remaining <= 5:
        print(f"⚠️  WARNING: Only {remaining} requests remaining today!")
    
    if DAILY_REQUEST_COUNT >= DAILY_REQUEST_LIMIT:
        print(f"🚫 DAILY QUOTA EXCEEDED! Please wait for reset or upgrade plan.")
        return False
    
    return True

def estimate_quota_usage(num_images):
    """Estimate if we have enough quota for batch processing"""
    global DAILY_REQUEST_COUNT, DAILY_REQUEST_LIMIT
    
    remaining = DAILY_REQUEST_LIMIT - DAILY_REQUEST_COUNT
    
    print(f"📊 Quota Check:")
    print(f"   Current usage: {DAILY_REQUEST_COUNT}/{DAILY_REQUEST_LIMIT}")
    print(f"   Remaining: {remaining} requests")
    print(f"   Requested: {num_images} images")
    
    if num_images > remaining:
        print(f"❌ Insufficient quota! Need {num_images} but only {remaining} available.")
        print(f"💡 Suggestions:")
        print(f"   - Process {remaining} images now")
        print(f"   - Wait for daily reset (midnight PT)")
        print(f"   - Upgrade to paid plan for higher limits")
        return False, remaining
    
    print(f"✅ Sufficient quota available")
    return True, remaining

def show_quota_status_and_options():
    """Show current quota status and available options"""
    global DAILY_REQUEST_COUNT, DAILY_REQUEST_LIMIT
    
    remaining = DAILY_REQUEST_LIMIT - DAILY_REQUEST_COUNT
    usage_percent = (DAILY_REQUEST_COUNT / DAILY_REQUEST_LIMIT) * 100
    
    print(f"\n📊 GEMINI API QUOTA STATUS")
    print(f"{'='*50}")
    print(f"Daily Usage: {DAILY_REQUEST_COUNT}/{DAILY_REQUEST_LIMIT} ({usage_percent:.1f}%)")
    print(f"Remaining: {remaining} requests")
    
    # Calculate time until reset (midnight Pacific Time)
    import datetime
    import pytz
    
    try:
        pacific = pytz.timezone('US/Pacific')
        now_pacific = datetime.datetime.now(pacific)
        
        # Next midnight Pacific Time
        next_midnight = (now_pacific + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        time_until_reset = next_midnight - now_pacific
        
        hours = int(time_until_reset.total_seconds() // 3600)
        minutes = int((time_until_reset.total_seconds() % 3600) // 60)
        
        print(f"Reset in: {hours}h {minutes}m (midnight Pacific Time)")
    except:
        print(f"Reset: Midnight Pacific Time")
    
    print(f"{'='*50}")
    
    if remaining <= 0:
        print(f"🚫 QUOTA EXHAUSTED")
        print(f"Options:")
        print(f"1. ⏰ Wait for daily reset")
        print(f"2. 💳 Upgrade to paid plan:")
        print(f"   - Pay-per-use: $0.075 per 1K requests")
        print(f"   - Higher daily limits")
        print(f"3. 🔄 Optimize usage with caching")
    elif remaining <= 5:
        print(f"⚠️  LOW QUOTA WARNING")
        print(f"Recommend upgrading soon or waiting for reset")
    else:
        print(f"✅ Quota healthy - {remaining} requests available")
    
    return remaining

def detect_api_quota_limits():
    """Automatically detect API quota limits by analyzing response headers"""
    global DAILY_REQUEST_LIMIT, MINUTE_REQUEST_LIMIT, QUOTA_AUTO_DETECTED
    
    api_key = setup_gemini_client()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
    
    # Make a minimal test request to check headers
    test_payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Test quota detection"}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 10
        }
    }
    
    try:
        print("🔍 Auto-detecting API quota limits...")
        response = requests.post(url, json=test_payload)
        
        # Parse quota information from response headers
        headers = response.headers
        
        # Common quota header patterns
        quota_headers = [
            'x-ratelimit-limit-requests-per-day',
            'x-quota-limit-requests-per-day', 
            'x-daily-quota-limit',
            'x-ratelimit-limit',
            'quota-limit-requests-per-day'
        ]
        
        minute_headers = [
            'x-ratelimit-limit-requests-per-minute',
            'x-quota-limit-requests-per-minute',
            'x-minute-quota-limit',
            'quota-limit-requests-per-minute'
        ]
        
        detected_daily = None
        detected_minute = None
        
        # Check for daily quota headers
        for header in quota_headers:
            if header in headers:
                try:
                    detected_daily = int(headers[header])
                    print(f"✅ Detected daily limit: {detected_daily} requests/day")
                    break
                except (ValueError, TypeError):
                    continue
        
        # Check for minute quota headers  
        for header in minute_headers:
            if header in headers:
                try:
                    detected_minute = int(headers[header])
                    print(f"✅ Detected minute limit: {detected_minute} requests/minute")
                    break
                except (ValueError, TypeError):
                    continue
        
        # If we got a 429 error, try to parse quota info from error message
        if response.status_code == 429:
            try:
                error_data = response.json()
                error_message = str(error_data)
                
                # Look for quota patterns in error message
                import re
                
                daily_match = re.search(r'(\d+)\s*requests?\s*per\s*day', error_message, re.IGNORECASE)
                if daily_match:
                    detected_daily = int(daily_match.group(1))
                    print(f"✅ Detected daily limit from error: {detected_daily} requests/day")
                
                minute_match = re.search(r'(\d+)\s*requests?\s*per\s*minute', error_message, re.IGNORECASE)
                if minute_match:
                    detected_minute = int(minute_match.group(1))
                    print(f"✅ Detected minute limit from error: {detected_minute} requests/minute")
                    
            except:
                pass
        
        # Update global limits if detected
        if detected_daily:
            DAILY_REQUEST_LIMIT = detected_daily
            
        if detected_minute:
            MINUTE_REQUEST_LIMIT = detected_minute
            
        # Try to infer model tier based on response
        model_tier = "unknown"
        
        if detected_daily:
            if detected_daily <= 50:
                model_tier = "free"
            elif detected_daily <= 1000:
                model_tier = "basic-paid"
            else:
                model_tier = "premium-paid"
        
        if detected_daily or detected_minute:
            QUOTA_AUTO_DETECTED = True
            print(f"🎯 Quota auto-detection successful!")
            print(f"   Model: gemini-1.5-pro")
            print(f"   Tier: {model_tier}")
            print(f"   Daily Limit: {DAILY_REQUEST_LIMIT}")
            print(f"   Minute Limit: {MINUTE_REQUEST_LIMIT}")
        else:
            print(f"⚠️  Could not auto-detect quotas, using defaults:")
            print(f"   Daily Limit: {DAILY_REQUEST_LIMIT} (default)")
            print(f"   Minute Limit: {MINUTE_REQUEST_LIMIT} (default)")
            
            # Try to detect based on response success/failure patterns
            if response.status_code == 200:
                print(f"✅ API responding normally, quotas likely higher than defaults")
            elif response.status_code == 429:
                print(f"⚠️  Already at quota limit - may need to adjust tracking")
        
        return {
            "daily_limit": DAILY_REQUEST_LIMIT,
            "minute_limit": MINUTE_REQUEST_LIMIT,
            "auto_detected": QUOTA_AUTO_DETECTED,
            "model_tier": model_tier
        }
        
    except Exception as e:
        print(f"❌ Quota detection failed: {e}")
        print(f"Using default limits: {DAILY_REQUEST_LIMIT} daily, {MINUTE_REQUEST_LIMIT} per minute")
        return {
            "daily_limit": DAILY_REQUEST_LIMIT,
            "minute_limit": MINUTE_REQUEST_LIMIT,
            "auto_detected": False,
            "error": str(e)
        }

def get_model_specific_limits(model_name="gemini-1.5-pro"):
    """Get model-specific quota limits with fallback values"""
    
    # Known limits for different Gemini models (as of 2024)
    model_limits = {
        "gemini-1.5-pro": {
            "free_tier": {"daily": 50, "minute": 2},
            "paid_tier": {"daily": 1000, "minute": 10}
        },
        "gemini-1.5-flash": {
            "free_tier": {"daily": 50, "minute": 15}, 
            "paid_tier": {"daily": 1000, "minute": 1000}
        },
        "gemini-pro": {
            "free_tier": {"daily": 50, "minute": 2},
            "paid_tier": {"daily": 1000, "minute": 10}
        }
    }
    
    # Try auto-detection first
    detected = detect_api_quota_limits()
    
    if detected["auto_detected"]:
        return detected
    
    # Fallback to known limits
    if model_name in model_limits:
        # Default to free tier limits
        limits = model_limits[model_name]["free_tier"]
        print(f"📋 Using known limits for {model_name} (free tier)")
        print(f"   Daily: {limits['daily']}, Minute: {limits['minute']}")
        
        global DAILY_REQUEST_LIMIT, MINUTE_REQUEST_LIMIT
        DAILY_REQUEST_LIMIT = limits["daily"]
        MINUTE_REQUEST_LIMIT = limits["minute"]
        
        return {
            "daily_limit": limits["daily"],
            "minute_limit": limits["minute"],
            "auto_detected": False,
            "model_tier": "free",
            "source": "known_limits"
        }
    
    # Final fallback
    print(f"⚠️  Unknown model {model_name}, using conservative defaults")
    return {
        "daily_limit": 50,
        "minute_limit": 2,
        "auto_detected": False,
        "model_tier": "unknown",
        "source": "conservative_default"
    }

def reset_quota_tracking(current_usage=None):
    """Reset quota tracking (call this after daily reset or when starting fresh)"""
    global DAILY_REQUEST_COUNT, REQUEST_START_TIME
    
    if current_usage is not None:
        DAILY_REQUEST_COUNT = current_usage
        print(f"📊 Quota tracking initialized to {current_usage} requests")
    else:
        DAILY_REQUEST_COUNT = 0
        print(f"📊 Quota tracking reset to 0")
    
    REQUEST_START_TIME = time.time()
    print(f"⏰ Session started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

def initialize_quota_from_dashboard():
    """Initialize quota based on current dashboard status with auto-detection"""
    print(f"🚗 AI Car Autofill - Initializing with Gemini Pro Model")
    print(f"{'='*60}")
    
    # Auto-detect quota limits for the Pro model
    quota_info = get_model_specific_limits("gemini-1.5-pro")
    
    print(f"\n🔧 Based on your dashboard showing 53/50 requests used...")
    print(f"Setting quota tracking to reflect current usage")
    
    # Since they're over the limit, set to the detected/known limit to prevent further requests
    reset_quota_tracking(quota_info["daily_limit"])
    
    print(f"\n📊 MODEL UPGRADE BENEFITS:")
    print(f"✅ Switched to: gemini-1.5-pro (more accurate)")
    print(f"✅ Better car identification and technical specs")
    print(f"✅ More context-aware field completion")
    print(f"✅ Auto-detected quota limits: {quota_info['daily_limit']}/day, {quota_info['minute_limit']}/min")
    
    print(f"\n💡 Recommendations:")
    print(f"1. ⏰ Wait for daily quota reset (midnight Pacific Time)")
    print(f"2. 💳 Consider upgrading to paid plan for immediate access")
    print(f"3. 🔍 Use show_quota_status_and_options() to check reset time")
    print(f"4. 🎯 Pro model gives much better car analysis results")
    
    return quota_info


# Load environment variables
def load_env_vars():
    """Load environment variables from .env file"""
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"').strip("'")

load_env_vars()


def setup_gemini_client():
    """Setup Gemini AI client"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return api_key


def encode_image_to_base64(image_path):
    """Encode image to base64 for Gemini API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_car_info_with_gemini(image_path, form_schema=None, max_retries=3):
    """Extract car information from image using Gemini Vision API with rate limiting"""
    # Check quota before making request
    if not track_api_usage():
        return {"error": "Daily quota exceeded. Please wait for reset or upgrade plan."}
    
    api_key = setup_gemini_client()
    
    # Encode image
    image_base64 = encode_image_to_base64(image_path)
    
    # Create prompt for structured extraction
    prompt = """
    Analyze this car image and extract comprehensive information in JSON format.
    
    IMPORTANT: Use only English characters (A-Z, a-z, 0-9, spaces, hyphens) in your response. 
    Convert any special characters or accents to their English equivalents (e.g., Citroën → Citroen, Peugeot → Peugeot).
    
    CRITICAL: Be CONTEXT-AWARE and MODEL-SPECIFIC for ALL fields. Use your knowledge of the specific car model to provide accurate specifications:
    - Research the typical specifications of the exact model you identify
    - Consider model variants (e.g., e-C3 = electric, AMG = high performance, etc.)
    - Provide realistic market values based on year, condition, and model
    - Use actual technical specifications for the identified model
    
    {
        "brand": "exact brand name using only English characters",
        "model": "specific model name using only English characters",
        "year": "estimated manufacturing year (YYYY format)",
        "condition": "estimated condition (new, used, reconditioned)",
        "body_type": "body style (sedan, hatchback, suv, coupe, convertible, pickup, van, etc.)",
        "fuel_type": "MODEL-SPECIFIC fuel type (electric, hybrid, diesel, petrol, or null if uncertain)",
        "transmission": "MODEL-SPECIFIC transmission (manual, automatic, cvt, or null if uncertain/not applicable)",
        "engine_capacity": "MODEL-SPECIFIC engine displacement in CC (e.g., 1200, 1500, 2000, or null for electric)",
        "estimated_mileage": "realistic mileage estimate based on year and condition (in kilometers, or null if uncertain)",
        "estimated_price_lkr": "realistic market price estimate in Sri Lankan Rupees based on model, year, condition (or null if uncertain)",
        "color": "primary exterior color",
        "confidence": "overall confidence score 0-1",
        "visible_features": ["list of clearly visible features"],
        "extraction_notes": "detailed observations about the specific model, its specifications, and market context"
    }
    
    Focus on MODEL-SPECIFIC accuracy. Use your knowledge of car specifications, not generic estimates.
    If you cannot determine a specific value with reasonable confidence, use null - do not guess.
    For prices, consider Sri Lankan car market conditions and depreciation patterns.
    """
    
    if form_schema:
        prompt += f"\n\nAvailable form options:\n{json.dumps(form_schema, indent=2)}"
        prompt += "\n\nEnsure your extracted values match the available options where possible."
    
    # Make API request to Gemini Pro (more accurate for complex analysis)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.05,  # Lower for more consistent technical analysis
            "topK": 3,
            "topP": 0.95,
            "maxOutputTokens": 1500,  # More tokens for detailed analysis
            "candidateCount": 1
        }
    }
    
    # Retry logic with exponential backoff for rate limiting
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(url, json=payload)
            
            # Handle rate limiting specifically
            if response.status_code == 429:
                if attempt < max_retries:
                    # Exponential backoff: 2^attempt seconds + random jitter
                    import random
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"⏳ Rate limited. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                else:
                    return {"error": f"Rate limit exceeded after {max_retries} retries. Please try again later."}
            
            response.raise_for_status()
            
            result = response.json()
            if 'candidates' in result and result['candidates']:
                content = result['candidates'][0]['content']['parts'][0]['text']
                
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    extracted_data = json.loads(json_match.group())
                    return extracted_data
                else:
                    return {"error": "No JSON found in response", "raw_response": content}
            else:
                return {"error": "No response from Gemini", "result": result}
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries and "429" in str(e):
                # Additional retry for network-level 429 errors
                import random
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⏳ Network rate limit. Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            return {"error": f"API request failed: {e}"}
        except json.JSONDecodeError as e:
            return {"error": f"JSON parsing failed: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}


def normalize_text_for_search(text):
    """Normalize text for better vector search matching"""
    if not text:
        return ""
    
    import re
    import unicodedata
    
    # Convert to string if not already
    text = str(text)
    
    # Normalize unicode characters (remove accents, special chars)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces and normalize separators
    text = re.sub(r'[_\-\s]+', ' ', text)
    
    # Remove special characters except spaces and hyphens
    text = re.sub(r'[^a-z0-9\s\-]', '', text)
    
    # Clean up extra spaces
    text = ' '.join(text.split())
    
    return text.strip()


def setup_faiss_vector_search(vector_dataset):
    """Setup FAISS index for vector similarity search"""
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError:
        print("❌ FAISS not installed. Installing...")
        os.system("pip install faiss-cpu")
        import faiss
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✅ scikit-learn imported successfully")
    except ImportError:
        print("❌ scikit-learn not installed. Installing...")
        os.system("pip install scikit-learn")
        from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Prepare vectors for indexing
    texts = []
    metadata = []
    
    for entry in vector_dataset['vector_entries']:
        if entry['type'] in ['brand', 'brand_model']:
            # Normalize embedding text for consistency
            normalized_text = normalize_text_for_search(entry['embedding_text'])
            if normalized_text:  # Only add non-empty normalized text
                texts.append(normalized_text)
                metadata.append(entry)
                
                # Add search variations as additional entries (also normalized)
                for variation in entry.get('search_variations', []):
                    normalized_variation = normalize_text_for_search(variation)
                    if normalized_variation and normalized_variation != normalized_text:
                        texts.append(normalized_variation)
                        metadata.append(entry)
    
    print(f"📊 Prepared {len(texts)} normalized text entries for vector indexing")
    
    # Create TF-IDF vectorizer optimized for car brand/model matching
    tfidf_vectorizer = TfidfVectorizer(
        analyzer='char_wb',  # Use character n-grams with word boundaries
        ngram_range=(2, 4),  # Use 2-4 character n-grams
        max_features=1000,   # Limit features for efficiency
        lowercase=True,
        strip_accents='unicode'
    )
    
    # Fit and transform the texts
    print("🔍 Generating TF-IDF embeddings...")
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # Convert to dense array for FAISS
    embeddings = tfidf_matrix.toarray().astype(np.float32)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    print(f"🔍 FAISS index created with {index.ntotal} vectors, dimension: {dimension}")
    
    def query_embedding_function(query_text):
        """Transform query text to TF-IDF embedding"""
        normalized_query = normalize_text_for_search(query_text)
        query_vector = tfidf_vectorizer.transform([normalized_query]).toarray().astype(np.float32)
        faiss.normalize_L2(query_vector)
        return query_vector
    
    return {
        'index': index,
        'embeddings': embeddings,
        'texts': texts,
        'metadata': metadata,
        'embedding_function': query_embedding_function,
        'tfidf_vectorizer': tfidf_vectorizer
    }


def search_vector_database(query_text, faiss_data, top_k=5):
    """Search the FAISS vector database for similar entries"""
    # Normalize query text for consistent search
    normalized_query = normalize_text_for_search(query_text)
    
    # Generate TF-IDF embedding for normalized query
    query_embedding = faiss_data['embedding_function'](normalized_query)
    
    # Search
    scores, indices = faiss_data['index'].search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(faiss_data['metadata']):
            result = {
                'score': float(score),
                'text': faiss_data['texts'][idx],
                'original_text': faiss_data['metadata'][idx]['embedding_text'],  # Keep original for reference
                'normalized_query': normalized_query,
                'metadata': faiss_data['metadata'][idx]
            }
            results.append(result)
    
    return results


def match_gemini_extraction_to_form(extracted_data, vector_dataset, use_vector_search=True):
    """Match Gemini extracted data to form values using vector search or direct matching"""
    
    if use_vector_search:
        # Setup FAISS if not already done
        faiss_data = setup_faiss_vector_search(vector_dataset)
        
        # Search for brand and model
        search_query = f"{extracted_data.get('brand', '')} {extracted_data.get('model', '')}".strip()
        if search_query:
            search_results = search_vector_database(search_query, faiss_data, top_k=3)
            
            # Find best matching brand-model combination
            best_match = None
            for result in search_results:
                if result['metadata']['type'] == 'brand_model':
                    best_match = result
                    break
            
            if not best_match:
                # Try brand-only search
                brand_query = extracted_data.get('brand', '')
                if brand_query:
                    brand_results = search_vector_database(brand_query, faiss_data, top_k=3)
                    for result in brand_results:
                        if result['metadata']['type'] == 'brand':
                            best_match = result
                            break
            
            if best_match:
                form_data = best_match['metadata'].get('form_autofill_data', {})
                match_info = {
                    'method': 'vector_search',
                    'confidence': best_match['score'],
                    'matched_text': best_match['text'],
                    'search_query': search_query
                }
            else:
                form_data = {}
                match_info = {'method': 'vector_search', 'confidence': 0, 'error': 'No matches found'}
        else:
            form_data = {}
            match_info = {'method': 'vector_search', 'confidence': 0, 'error': 'No search query'}
    else:
        # Direct matching approach
        form_data = {}
        match_info = {'method': 'direct_matching', 'confidence': 0.5}
        
        # Try to match brand directly
        brand_name = extracted_data.get('brand', '').lower()
        model_name = extracted_data.get('model', '').lower()
        
        for entry in vector_dataset['vector_entries']:
            if entry['type'] == 'brand_model':
                if (brand_name in entry.get('brand_label', '').lower() and 
                    model_name in entry.get('model_label', '').lower()):
                    form_data = entry.get('form_autofill_data', {})
                    match_info['confidence'] = 0.9
                    break
    
    # Map other fields
    form_mappings = vector_dataset.get('form_field_mappings', {})
    additional_fields = {}
    
    # Map condition
    if 'condition' in extracted_data and 'condition' in form_mappings:
        condition_value = extracted_data['condition']
        for value in form_mappings['condition'].get('values', []):
            if condition_value and condition_value.lower() == value['label'].lower():
                additional_fields['condition'] = value
                break
    
    # Map body type  
    if 'body_type' in extracted_data and 'body' in form_mappings:
        body_value = extracted_data['body_type']
        for value in form_mappings['body'].get('values', []):
            if body_value and body_value.lower() in value['label'].lower():
                additional_fields['body'] = value
                break
    
    # Map fuel type
    if 'fuel_type' in extracted_data and 'fuel_type' in form_mappings:
        fuel_value = extracted_data['fuel_type']
        for value in form_mappings['fuel_type'].get('values', []):
            if fuel_value and fuel_value.lower() == value['label'].lower():
                additional_fields['fuel_type'] = value
                break
    
    # Add year if provided
    if 'year' in extracted_data and extracted_data['year']:
        try:
            year_constraints = form_mappings.get('model_year', {}).get('constraints', {})
            year = int(extracted_data['year'])
            min_year = year_constraints.get('min_year', 1900)
            max_year = year_constraints.get('max_year', 2030)
            
            if min_year <= year <= max_year:
                additional_fields['model_year'] = year
        except (ValueError, TypeError):
            pass  # If LLM year is invalid, don't add the field
    
    return {
        'extracted_data': extracted_data,
        'form_autofill': {**form_data, **additional_fields},
        'match_info': match_info,
        'available_suggestions': {
            'colors': extracted_data.get('color'),
            'features': extracted_data.get('visible_features', []),
            'notes': extracted_data.get('extraction_notes')
        }
    }


def process_car_image_end_to_end(image_path, vector_dataset, add_delay=True):
    """Complete end-to-end processing: image -> extraction -> form autofill"""
    print(f"🚗 Processing car image: {image_path}")
    
    # Add throttling delay to respect API rate limits
    if add_delay:
        print("⏱️  Adding delay to respect API rate limits...")
        time.sleep(1.5)  # 1.5 second delay between requests
    
    # Step 1: Extract car info with Gemini
    print("📸 Step 1: Extracting car information with Gemini Vision...")
    form_schema = {
        'brands': [entry['brand_label'] for entry in vector_dataset['vector_entries'] 
                  if entry['type'] in ['brand', 'brand_model']][:20],  # Sample for prompt
        'conditions': [v['label'] for v in vector_dataset['form_field_mappings']['condition']['values']],
        'body_types': [v['label'] for v in vector_dataset['form_field_mappings']['body']['values']],
        'fuel_types': [v['label'] for v in vector_dataset['form_field_mappings']['fuel_type']['values']]
    }
    
    extracted_data = extract_car_info_with_gemini(image_path, form_schema)
    
    if 'error' in extracted_data:
        return {'error': extracted_data['error']}
    
    print("✅ Extraction completed")
    print(f"   Brand: {extracted_data.get('brand', 'Unknown')}")
    print(f"   Model: {extracted_data.get('model', 'Unknown')}")
    print(f"   Confidence: {extracted_data.get('confidence', 0)}")
    
    # Step 2: Match to form values
    print("\n🔍 Step 2: Matching to form values...")
    matched_result = match_gemini_extraction_to_form(extracted_data, vector_dataset, use_vector_search=True)
    
    print("✅ Matching completed")
    print(f"   Method: {matched_result['match_info']['method']}")
    print(f"   Confidence: {matched_result['match_info']['confidence']:.2f}")
    
    # Step 3: Enhance match info and generate exact form submission JSON
    print("\n📝 Step 3: Generating ikman.lk form submission JSON...")
    enhanced_match_info = enhance_match_info_with_form_data(extracted_data, vector_dataset, matched_result['match_info'])
    ikman_form_json = generate_ikman_form_submission_json(extracted_data, vector_dataset, matched_result['match_info'])
    
    print("✅ Form JSON generation completed")
    print(f"   Generated {len(ikman_form_json)} form fields")
    
    # Add the ikman form JSON to the result
    matched_result['ikman_form_submission'] = ikman_form_json
    matched_result['enhanced_match_info'] = enhanced_match_info
    
    return matched_result


def generate_ikman_form_submission_json(extracted_data, vector_dataset, match_info):
    """Generate exact ikman.lk form submission JSON with fuzzy matching"""
    
    # Get the form field mappings from our dataset
    form_mappings = vector_dataset.get('form_field_mappings', {})
    
    # Initialize the form submission structure
    form_submission = {}
    
    # Helper function for fuzzy string matching
    def fuzzy_match_string(target, options, threshold=0.6):
        """Find best matching option using fuzzy string similarity"""
        if not target or not options:
            return None
        
        target_lower = target.lower().strip()
        best_match = None
        best_score = 0
        
        for option in options:
            option_text = option.get('label', '').lower().strip()
            if not option_text:
                continue
                
            # Exact match gets highest priority
            if target_lower == option_text:
                return option
            
            # Check if target contains option or vice versa
            if target_lower in option_text or option_text in target_lower:
                score = min(len(target_lower), len(option_text)) / max(len(target_lower), len(option_text))
                if score > best_score:
                    best_score = score
                    best_match = option
            
            # Simple word overlap scoring
            target_words = set(target_lower.split())
            option_words = set(option_text.split())
            if target_words and option_words:
                overlap = len(target_words.intersection(option_words))
                total_words = len(target_words.union(option_words))
                score = overlap / total_words if total_words > 0 else 0
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = option
        
        return best_match if best_score >= threshold else None
    


    # Helper function to generate text for text fields
    def generate_text_field_value(field_key, field_info, extracted_data):
        """Generate appropriate text for text fields based on extracted data"""
        field_label = field_info.get('label', '').lower()
        max_length = field_info.get('maximum_length', 100)
        
        if 'edition' in field_label or 'trim' in field_label:
            # Try to extract edition/trim info from model or generate based on brand
            model = extracted_data.get('model', '')
            brand = extracted_data.get('brand', '')
            if model and len(model) <= max_length:
                return model
            elif brand and len(brand) <= max_length:
                return f"{brand} Edition"
            else:
                return "Standard"
        
        elif 'description' in field_label:
            # Generate description based on LLM's extracted data
            brand = extracted_data.get('brand', 'Car')
            model = extracted_data.get('model', '')
            year = extracted_data.get('year', '')
            condition = extracted_data.get('condition', 'used')
            color = extracted_data.get('color', '')
            fuel_type = extracted_data.get('fuel_type', '')
            body_type = extracted_data.get('body_type', '')
            transmission = extracted_data.get('transmission', '')
            engine_capacity = extracted_data.get('engine_capacity', '')
            
            description_parts = []
            if year:
                description_parts.append(f"{year}")
            if brand:
                description_parts.append(brand)
            if model:
                description_parts.append(model)
            if color:
                description_parts.append(f"{color} color")
            if body_type:
                description_parts.append(body_type)
            if engine_capacity and fuel_type != 'electric':
                description_parts.append(f"{engine_capacity}CC")
            if fuel_type:
                if fuel_type == 'electric':
                    description_parts.append("electric vehicle")
                else:
                    description_parts.append(f"{fuel_type} engine")
            if transmission:
                description_parts.append(f"{transmission} transmission")
            if condition:
                description_parts.append(f"in {condition} condition")
            
            description = " ".join(description_parts)
            if len(description) > max_length:
                description = description[:max_length-3] + "..."
            
            return description if description else "Well maintained car for sale"
        
        else:
            # Default text generation
            return "Please specify"
    
    # Process each form field
    for field_key, field_info in form_mappings.items():
        field_type = field_info.get('type')
        field_label = field_info.get('label', '')
        
        if field_type == 'enum':
            # Handle enum fields (condition, body, fuel_type, transmission)
            field_values = field_info.get('values', [])
            matched_value = None
            
            if field_key == 'condition':
                condition = extracted_data.get('condition')
                if condition:
                    matched_value = fuzzy_match_string(condition, field_values)
            
            elif field_key == 'body':
                body_type = extracted_data.get('body_type')
                if body_type:
                    matched_value = fuzzy_match_string(body_type, field_values)
            
            elif field_key == 'fuel_type':
                # Use LLM's intelligent fuel type determination
                fuel_type = extracted_data.get('fuel_type')
                if fuel_type:
                    matched_value = fuzzy_match_string(fuel_type, field_values)
                # If LLM returned null or no match, don't add the field (manual fill required)
            
            elif field_key == 'transmission':
                # Use LLM's intelligent transmission determination
                transmission = extracted_data.get('transmission')
                if transmission:
                    matched_value = fuzzy_match_string(transmission, field_values)
                # If LLM returned null or no match, don't add the field (manual fill required)
            
            # Add to form submission if we found a match
            if matched_value:
                form_submission[field_key] = matched_value['key']
        
        elif field_type == 'tree':
            # Handle tree fields (brand and model)
            if field_key == 'brand':
                # Use the matched brand from vector search
                if match_info.get('confidence', 0) > 0.5:
                    brand_data = match_info.get('matched_brand_data')
                    if brand_data:
                        form_submission[field_key] = brand_data.get('key')
                    else:
                        # Try to find brand from extracted data
                        brand = extracted_data.get('brand')
                        if brand:
                            brand_values = field_info.get('values', [])
                            matched_brand = fuzzy_match_string(brand, brand_values, threshold=0.5)
                            if matched_brand:
                                form_submission[field_key] = matched_brand['key']
        
        elif field_type == 'year':
            # Handle year fields using LLM's intelligent year estimation
            if field_key == 'model_year':
                year = extracted_data.get('year')
                if year:
                    try:
                        year_int = int(year)
                        min_year = field_info.get('constraints', {}).get('min_year', 1926)
                        max_year = field_info.get('constraints', {}).get('max_year', 2026)
                        
                        if min_year <= year_int <= max_year:
                            form_submission[field_key] = year_int
                        # If LLM's year is out of range, don't add the field (manual fill required)
                    except (ValueError, TypeError):
                        pass  # If LLM year is invalid, don't add the field
        
        elif field_type == 'measurement':
            # Handle measurement fields using LLM's intelligent estimates
            if field_key == 'mileage':
                # Use LLM's intelligent mileage estimation
                estimated_mileage = extracted_data.get('estimated_mileage')
                if estimated_mileage:
                    try:
                        mileage_int = int(estimated_mileage)
                        min_mileage = field_info.get('constraints', {}).get('minimum', 0)
                        max_mileage = field_info.get('constraints', {}).get('maximum', 1000000)
                        
                        if min_mileage <= mileage_int <= max_mileage:
                            form_submission[field_key] = mileage_int
                        # If LLM's mileage is out of range, don't add the field (manual fill required)
                    except (ValueError, TypeError):
                        pass  # If LLM mileage is invalid, don't add the field
            
            elif field_key == 'engine_capacity':
                # Use LLM's intelligent engine capacity estimation
                engine_capacity = extracted_data.get('engine_capacity')
                if engine_capacity:
                    try:
                        capacity_int = int(engine_capacity)
                        min_capacity = field_info.get('constraints', {}).get('minimum', 0)
                        max_capacity = field_info.get('constraints', {}).get('maximum', 10000)
                        
                        if min_capacity <= capacity_int <= max_capacity:
                            form_submission[field_key] = capacity_int
                        # If LLM's capacity is out of range, don't add the field (manual fill required)
                    except (ValueError, TypeError):
                        pass  # If LLM capacity is invalid, don't add the field
        
        elif field_type == 'money':
            # Handle money fields using LLM's intelligent price estimation
            if field_key == 'price':
                # Use LLM's intelligent price estimation based on Sri Lankan market
                estimated_price_lkr = extracted_data.get('estimated_price_lkr')
                if estimated_price_lkr:
                    try:
                        price_int = int(estimated_price_lkr)
                        min_price = field_info.get('constraints', {}).get('minimum', 0)
                        max_price = field_info.get('constraints', {}).get('maximum', 9999999999999)
                        
                        if min_price <= price_int <= max_price:
                            form_submission[field_key] = price_int
                        # If LLM's price is out of range, don't add the field (manual fill required)
                    except (ValueError, TypeError):
                        pass  # If LLM price is invalid, don't add the field
        
        elif field_type in ['text', 'description']:
            # Handle text fields
            if field_info.get('required', False):
                generated_text = generate_text_field_value(field_key, field_info, extracted_data)
                form_submission[field_key] = generated_text
    
    # Handle brand and model fields specifically
    # ikman.lk expects both brand and model in the form submission
    brand = extracted_data.get('brand', '').lower()
    model = extracted_data.get('model', '').lower()
    
    if brand and model:
        # First try to use matched model data from vector search
        model_data = match_info.get('matched_model_data')
        if model_data and model_data.get('model_key'):
            form_submission['brand'] = model_data.get('brand_key')
            form_submission['model'] = model_data.get('model_key')
        else:
            # Fallback: Search through vector entries to find matching brand-model combination
            for entry in vector_dataset.get('vector_entries', []):
                if entry.get('type') == 'brand_model':
                    entry_brand = entry.get('brand_label', '').lower()
                    entry_model = entry.get('model_label', '').lower()
                    
                    if (brand in entry_brand and model in entry_model):
                        form_submission['brand'] = entry.get('brand_key')
                        form_submission['model'] = entry.get('model_key')
                        break
    elif brand:
        # If we only have brand, try to match just the brand
        brand_data = match_info.get('matched_brand_data')
        if brand_data:
            form_submission['brand'] = brand_data.get('key')
        else:
            # Fallback: Search for brand in form mappings
            brand_values = form_mappings.get('brand', {}).get('values', [])
            matched_brand = fuzzy_match_string(brand, brand_values, threshold=0.5)
            if matched_brand:
                form_submission['brand'] = matched_brand['key']
    
    return form_submission


def enhance_match_info_with_form_data(extracted_data, vector_dataset, match_info):
    """Use FAISS vector search results directly - no additional pattern matching needed"""
    enhanced_match_info = match_info.copy()
    
    # Get form field mappings
    form_mappings = vector_dataset.get('form_field_mappings', {})
    
    # Extract brand and model from the vector search results 
    brand_name = extracted_data.get('brand', '').lower()
    
    # Use vector search to find the best matching entry
    if match_info.get('confidence', 0) > 0.5:
        # Setup FAISS and search - reuse the vector search that already worked
        faiss_data = setup_faiss_vector_search(vector_dataset)
        search_query = f"{extracted_data.get('brand', '')} {extracted_data.get('model', '')}".strip()
        
        if search_query:
            search_results = search_vector_database(search_query, faiss_data, top_k=3)
            
            # Find the best brand_model match from vector search results
            for result in search_results:
                metadata = result['metadata']
                if metadata.get('type') == 'brand_model':
                    # Extract brand and model data directly from the vector search result
                    enhanced_match_info['matched_brand_data'] = {
                        'key': metadata.get('brand_key'),
                        'label': metadata.get('brand_label')
                    }
                    enhanced_match_info['matched_model_data'] = {
                        'brand_key': metadata.get('brand_key'),
                        'brand_label': metadata.get('brand_label'),
                        'model_key': metadata.get('model_key'),
                        'model_label': metadata.get('model_label')
                    }
                    break
    
    # Fallback: Search for brand in form values if not found via vector search
    if 'matched_brand_data' not in enhanced_match_info and 'brand' in form_mappings:
        brand_values = form_mappings['brand'].get('values', [])
        for brand_option in brand_values:
            if brand_name in brand_option.get('label', '').lower():
                enhanced_match_info['matched_brand_data'] = brand_option
                break
    
    return enhanced_match_info


def process_multiple_images_with_rate_limiting(image_paths, vector_dataset, delay_between_requests=2.0):
    """Process multiple car images with intelligent rate limiting"""
    # Check quota before starting batch processing
    can_process, remaining = estimate_quota_usage(len(image_paths))
    
    if not can_process:
        # Process only what we can with remaining quota
        if remaining > 0:
            print(f"🔄 Processing only {remaining} images due to quota limits...")
            image_paths = image_paths[:remaining]
        else:
            return {"error": "No quota remaining. Please wait for daily reset."}
    
    results = {}
    successful_count = 0
    failed_count = 0
    
    print(f"🚗 Starting batch processing of {len(image_paths)} images...")
    print(f"⏱️  Using {delay_between_requests}s delay between requests to respect API limits")
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n📸 Processing image {i}/{len(image_paths)}: {image_path}")
        
        try:
            # Process with rate limiting
            result = process_car_image_end_to_end(image_path, vector_dataset, add_delay=(i > 1))
            
            if 'error' in result:
                print(f"❌ Failed: {result['error']}")
                results[image_path] = result
                failed_count += 1
            else:
                brand = result['extracted_data'].get('brand', 'Unknown')
                model = result['extracted_data'].get('model', 'Unknown')
                confidence = result['extracted_data'].get('confidence', 0)
                print(f"✅ Success: {brand} {model} (confidence: {confidence:.2f})")
                results[image_path] = result
                successful_count += 1
                
        except Exception as e:
            error_result = {"error": f"Processing failed: {e}"}
            print(f"❌ Exception: {e}")
            results[image_path] = error_result
            failed_count += 1
        
        # Add extra delay between requests to be respectful
        if i < len(image_paths):
            print(f"⏳ Waiting {delay_between_requests}s before next request...")
            time.sleep(delay_between_requests)
    
    print(f"\n📊 Batch processing complete!")
    print(f"✅ Successful: {successful_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📈 Success rate: {(successful_count / len(image_paths)) * 100:.1f}%")
    
    return results


if __name__ == "__main__":
    """Main execution block for testing and demonstration"""
    print("🚗 AI Car Autofill Service - Testing Setup")
    print("=" * 50)
    
    # Load environment variables
    load_env_vars()
    
    # Check if vector dataset exists
    if not os.path.exists('car_vector_dataset.json'):
        print("❌ Vector dataset not found. Please ensure car_vector_dataset.json exists.")
        exit(1)
    
    # Load vector dataset
    try:
        with open('car_vector_dataset.json', 'r', encoding='utf-8') as f:
            vector_dataset = json.load(f)
        print(f"✅ Vector dataset loaded: {vector_dataset['metadata']['total_entries']} entries")
    except Exception as e:
        print(f"❌ Error loading vector dataset: {e}")
        exit(1)
    
    # Check API key
    try:
        api_key = setup_gemini_client()
        print("✅ Gemini API key configured")
    except ValueError as e:
        print(f"❌ {e}")
        print("Please create a .env file with your GEMINI_API_KEY")
        exit(1)
    
    # Test FAISS setup
    try:
        print("🔍 Testing FAISS vector search setup...")
        faiss_data = setup_faiss_vector_search(vector_dataset)
        print(f"✅ FAISS index created with {faiss_data['index'].ntotal} vectors")
    except Exception as e:
        print(f"❌ FAISS setup failed: {e}")
        exit(1)
    
    # Test vector search
    try:
        print("🔍 Testing vector search...")
        test_query = "BMW 3 Series"
        results = search_vector_database(test_query, faiss_data, top_k=3)
        print(f"✅ Vector search test successful: {len(results)} results for '{test_query}'")
        if results:
            print(f"   Best match: {results[0]['metadata'].get('brand_label', '')} {results[0]['metadata'].get('model_label', '')}")
    except Exception as e:
        print(f"❌ Vector search test failed: {e}")
        exit(1)
    
    print("\n🎉 All tests passed! The system is ready to use.")
    print("\nNext steps:")
    print("1. Run 'streamlit run streamlit_app.py' to start the web interface")
    print("2. Or use the functions directly in your code:")
    print("   from main import process_car_image_end_to_end")
    print("   result = process_car_image_end_to_end('your_image.jpg', vector_dataset)")