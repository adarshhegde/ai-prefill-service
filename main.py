import requests
import json
import time
from urllib.parse import quote
import os
from pathlib import Path
import base64
import numpy as np
import pickle
from datetime import datetime

# Load environment variables
def load_env_vars():
    """Load environment variables from .env file"""
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"').strip("' ")

load_env_vars()


class GeminiCostTracker:
    """Track and analyze Gemini API costs"""
    
    def __init__(self):
        # Gemini 1.5 Pro pricing (as of 2024)
        # Input tokens: $1.25 per 1M tokens (≤128k), $2.50 per 1M tokens (>128k)
        # Output tokens: $5.00 per 1M tokens (≤128k), $10.00 per 1M tokens (>128k)
        # Images: $0.0025 per image
        self.input_token_cost_per_1m_128k = 1.25
        self.input_token_cost_per_1m_above_128k = 2.50
        self.output_token_cost_per_1m_128k = 5.00
        self.output_token_cost_per_1m_above_128k = 10.00
        self.image_cost_per_image = 0.0025
        
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_images = 0
        self.total_cost = 0.0
        self.requests = []
    
    def add_request(self, input_tokens, output_tokens, images=1, request_type="vision"):
        """Add a request to the cost tracker"""
        # Determine pricing tier based on input token count
        if input_tokens <= 128000:
            input_cost_per_1m = self.input_token_cost_per_1m_128k
            output_cost_per_1m = self.output_token_cost_per_1m_128k
        else:
            input_cost_per_1m = self.input_token_cost_per_1m_above_128k
            output_cost_per_1m = self.output_token_cost_per_1m_above_128k
        
        input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
        image_cost = images * self.image_cost_per_image
        total_request_cost = input_cost + output_cost + image_cost
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_images += images
        self.total_cost += total_request_cost
        
        request_data = {
            'timestamp': datetime.now().isoformat(),
            'request_type': request_type,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'images': images,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'image_cost': image_cost,
            'total_cost': total_request_cost
        }
        self.requests.append(request_data)
        
        return total_request_cost
    
    def get_cost_summary(self):
        """Get a summary of all costs"""
        # Calculate cost breakdown by request type
        cost_by_type = {}
        for req in self.requests:
            req_type = req['request_type']
            if req_type not in cost_by_type:
                cost_by_type[req_type] = {
                    'count': 0,
                    'total_cost': 0,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0
                }
            cost_by_type[req_type]['count'] += 1
            cost_by_type[req_type]['total_cost'] += req['total_cost']
            cost_by_type[req_type]['total_input_tokens'] += req['input_tokens']
            cost_by_type[req_type]['total_output_tokens'] += req['output_tokens']
        
        return {
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_images': self.total_images,
            'total_cost_usd': self.total_cost,
            'average_cost_per_request': self.total_cost / len(self.requests) if self.requests else 0,
            'total_requests': len(self.requests),
            'cost_by_type': cost_by_type
        }
    
    def print_cost_summary(self):
        """Print a formatted cost summary"""
        summary = self.get_cost_summary()
        print("\n💰 Gemini API Cost Analysis")
        print("=" * 40)
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Total Input Tokens: {summary['total_input_tokens']:,}")
        print(f"Total Output Tokens: {summary['total_output_tokens']:,}")
        print(f"Total Images: {summary['total_images']}")
        print(f"Total Cost: ${summary['total_cost_usd']:.4f}")
        print(f"Average Cost per Request: ${summary['average_cost_per_request']:.4f}")
        
        # Show breakdown by request type
        if summary['cost_by_type']:
            print("\n📊 Cost Breakdown by Request Type:")
            for req_type, data in summary['cost_by_type'].items():
                print(f"  {req_type}: {data['count']} requests, ${data['total_cost']:.4f}")
                print(f"    Input: {data['total_input_tokens']:,} tokens, Output: {data['total_output_tokens']:,} tokens")
        
        if self.requests:
            print("\n📊 Recent Requests:")
            for i, req in enumerate(self.requests[-5:], 1):  # Show last 5 requests
                print(f"  {i}. {req['request_type']}: {req['input_tokens']}→{req['output_tokens']} tokens, ${req['total_cost']:.4f}")


# Global cost tracker instance
cost_tracker = GeminiCostTracker()


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


def process_multiple_identifications(extracted_data):
    """Process multiple car identifications and select the best one based on confidence"""
    
    # Start with the primary identification
    primary_identification = {
        'brand': extracted_data.get('brand', 'unknown'),
        'model': extracted_data.get('model', 'unknown'),
        'confidence': float(extracted_data.get('confidence', 0))
    }
    
    # Get alternative identifications
    alternatives = extracted_data.get('alternative_identifications', [])
    
    # Create a list of all identifications including primary
    all_identifications = [primary_identification] + alternatives
    
    # Filter out invalid identifications
    valid_identifications = []
    for identification in all_identifications:
        brand = identification.get('brand', '').strip()
        model = identification.get('model', '').strip()
        confidence = float(identification.get('confidence', 0))
        
        # Skip if brand or model is unknown/empty
        if brand.lower() in ['unknown', ''] or model.lower() in ['unknown', '']:
            continue
            
        # Skip if confidence is too low
        if confidence < 0.1:
            continue
            
        valid_identifications.append({
            'brand': brand,
            'model': model,
            'confidence': confidence
        })
    
    if not valid_identifications:
        # Fallback to primary identification even if low confidence
        return {
            'brand': primary_identification['brand'],
            'model': primary_identification['model'],
            'confidence': primary_identification['confidence'],
            'all_identifications': [primary_identification]
        }
    
    # Sort by confidence (highest first)
    valid_identifications.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Select the best identification
    best_identification = valid_identifications[0]
    
    # Log the selection process
    if len(valid_identifications) > 1:
        print(f"🔍 Multiple identifications found:")
        for i, identification in enumerate(valid_identifications[:3]):  # Show top 3
            print(f"   {i+1}. {identification['brand']} {identification['model']} (confidence: {identification['confidence']:.2f})")
        print(f"✅ Selected: {best_identification['brand']} {best_identification['model']} (highest confidence: {best_identification['confidence']:.2f})")
    else:
        print(f"✅ Single identification: {best_identification['brand']} {best_identification['model']} (confidence: {best_identification['confidence']:.2f})")
    
    return {
        'brand': best_identification['brand'],
        'model': best_identification['model'],
        'confidence': best_identification['confidence'],
        'all_identifications': valid_identifications,
        'selection_method': 'highest_confidence'
    }


def extract_car_info_with_gemini(image_path, form_schema=None, vector_search_results=None, max_retries=3):
    """Extract car information from image using Gemini Vision API with retry logic"""
    
    api_key = setup_gemini_client()
    
    # Encode image
    print("📸 Encoding image for Gemini API...")
    image_base64 = encode_image_to_base64(image_path)
    
    # Enhanced prompt to get multiple possible identifications
    prompt = """
    Analyze this car image and extract the brand and model. Return ONLY a JSON object with this structure:

    {
        "brand": "car brand (e.g., Toyota, Honda, BMW)",
        "model": "car model (e.g., Corolla, Civic, 3 Series)",
        "confidence": "confidence level (0.0 to 1.0)",
        "alternative_identifications": [
            {
                "brand": "alternative brand",
                "model": "alternative model", 
                "confidence": "confidence level (0.0 to 1.0)"
            }
        ]
    }

    Guidelines:
    - Be specific with model names (e.g., "Corolla" not just "Toyota")
    - If you can't identify clearly, use "unknown" for brand or model
    - Focus on the most prominent car in the image
    - Include alternative identifications if you're uncertain between similar models
    - Only include alternatives with confidence > 0.3
    """
    
    # Make API request to Gemini Pro
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
            "temperature": 0.0,
            "topK": 1,
            "topP": 0.95,
            "maxOutputTokens": 2048,
            "candidateCount": 1
        }
    }
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            print(f"🤖 Sending request to Gemini Vision API... (attempt {attempt + 1}/{max_retries})")
            import time
            start_time = time.time()
            
            # Add delay between retries to respect rate limits
            if attempt > 0:
                delay = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                print(f"⏳ Waiting {delay}s before retry...")
                time.sleep(delay)
            
            response = requests.post(url, json=payload, timeout=90)  # Increased timeout to 90 seconds
            response.raise_for_status()
            
            api_duration = time.time() - start_time
            print(f"📊 Processing Gemini API response... (took {api_duration:.1f}s)")
            result = response.json()
            
            if 'candidates' in result and result['candidates']:
                content = result['candidates'][0]['content']['parts'][0]['text']
                
                # Extract token usage from response
                usage_info = result.get('usageMetadata', {})
                input_tokens = usage_info.get('promptTokenCount', 0)
                output_tokens = usage_info.get('candidatesTokenCount', 0)
                
                # Track API costs
                request_cost = cost_tracker.add_request(input_tokens, output_tokens, images=1, request_type="brand_extraction")
                print(f"💰 Request cost: ${request_cost:.4f}")
                
                # Debug: Print actual token counts
                print(f"🔍 Token Usage Debug: input={input_tokens:,}, output={output_tokens:,}")
                
                import re
                json_match = re.search(r'```json\n(\{.*?\})\n```', content, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'(\{.*?\})', content, re.DOTALL)

                if json_match:
                    try:
                        print("✅ Parsing JSON response from Gemini...")
                        extracted_data = json.loads(json_match.group(1))
                        print(f"🎯 Extracted: {extracted_data.get('brand', 'Unknown')} {extracted_data.get('model', 'Unknown')}")
                        
                        # Process multiple identifications and select the best one
                        processed_data = process_multiple_identifications(extracted_data)
                        
                        # Add token usage to the response
                        processed_data['_token_usage'] = {
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'total_tokens': input_tokens + output_tokens,
                            'cost_usd': request_cost
                        }
                        
                        return processed_data
                    except json.JSONDecodeError:
                         print("❌ Failed to parse JSON from Gemini response")
                         return {"error": "Invalid JSON in response", "raw_response": content}
                else:
                    print("❌ No JSON found in Gemini response")
                    return {"error": "No JSON found in response", "raw_response": content}
            else:
                print("❌ No response from Gemini API")
                return {"error": "No response from Gemini", "result": result}
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:  # Last attempt
                return {"error": f"API request failed after {max_retries} attempts: {e}"}
            continue  # Try again
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:  # Last attempt
                return {"error": f"JSON parsing failed after {max_retries} attempts: {e}"}
            continue  # Try again
        except Exception as e:
            print(f"❌ Unexpected error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:  # Last attempt
                return {"error": f"Unexpected error after {max_retries} attempts: {e}"}
            continue  # Try again
    
    # If we get here, all retries failed
    return {"error": f"All {max_retries} attempts failed"}



def get_preliminary_query(image_path):
    """Use a lightweight prompt to get a brand/model query from an image."""
    api_key = setup_gemini_client()
    image_base64 = encode_image_to_base64(image_path)

    prompt = """
    Analyze this car image and identify the brand and model.
    Respond with only the brand and model, separated by a space.
    For example: "Toyota Corolla" or "BMW 3 Series".
    If you are not confident, respond with "unknown".
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
            ]
        }],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 50,
        }
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        if 'candidates' in result and result['candidates']:
            content = result['candidates'][0]['content']['parts'][0]['text'].strip()
            
            # Extract token usage
            usage_info = result.get('usageMetadata', {})
            input_tokens = usage_info.get('promptTokenCount', 0)
            output_tokens = usage_info.get('candidatesTokenCount', 0)
            
            # Track API costs
            request_cost = cost_tracker.add_request(input_tokens, output_tokens, images=1, request_type="preliminary_query")
            print(f"💰 Preliminary query cost: ${request_cost:.4f}")
            
            # Debug: Print actual token counts
            print(f"🔍 Preliminary Query Token Debug: input={input_tokens:,}, output={output_tokens:,}")
            
            if content.lower() != "unknown":
                return {
                    'query': content,
                    'token_usage': {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': input_tokens + output_tokens,
                        'cost_usd': request_cost
                    }
                }
    except Exception:
        return None
    return None


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
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    
    # Clean up extra spaces
    text = ' '.join(text.split())

    return text.strip()


def load_saved_faiss_data():
    """Load saved FAISS index and TF-IDF vectorizer if they exist"""
    index_path = "faiss_index.bin"
    vectorizer_path = "tfidf_vectorizer.pkl"
    metadata_path = "faiss_metadata.pkl"
    
    if (os.path.exists(index_path) and 
        os.path.exists(vectorizer_path) and 
        os.path.exists(metadata_path)):
        try:
            import faiss
            # Load FAISS index
            index = faiss.read_index(index_path)
            
            # Load TF-IDF vectorizer
            with open(vectorizer_path, 'rb') as f:
                tfidf_vectorizer = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            print(f"✅ Loaded saved FAISS index with {index.ntotal} vectors")
            
            def query_embedding_function(query_text):
                """Transform query text to TF-IDF embedding"""
                normalized_query = normalize_text_for_search(query_text)
                query_vector = tfidf_vectorizer.transform([normalized_query]).toarray().astype(np.float32)
                faiss.normalize_L2(query_vector)
                return query_vector
            
            return {
                'index': index,
                'metadata': metadata,
                'embedding_function': query_embedding_function,
                'tfidf_vectorizer': tfidf_vectorizer
            }
        except Exception as e:
            print(f"❌ Error loading saved FAISS data: {e}")
            return None
    return None



def clear_cached_embeddings():
    """Clear cached FAISS embeddings and vectorizer"""
    cache_files = ["faiss_index.bin", "tfidf_vectorizer.pkl", "faiss_metadata.pkl"]
    cleared_count = 0
    
    for file_path in cache_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                cleared_count += 1
                print(f"🗑️ Removed cached file: {file_path}")
            except Exception as e:
                print(f"❌ Error removing {file_path}: {e}")
    
    if cleared_count > 0:
        print(f"✅ Cleared {cleared_count} cached embedding files")
    else:
        print("ℹ️ No cached embedding files found to clear")
    
    return cleared_count

def setup_faiss_vector_search(vector_dataset):
    """Setup FAISS index for vector similarity search with caching"""
    # First try to load saved data
    saved_data = load_saved_faiss_data()
    if saved_data:
        return saved_data
    
    # If no cached data exists, create the index automatically
    print("🔧 No cached vector index found! Creating index on first run...")
    print("⏳ This may take a few minutes for the initial setup...")
    
    # Import the create_vector_index function
    try:
        from create_vector_index import create_vector_index
        success = create_vector_index("car_vector_dataset.json", ".")
        
        if success:
            print("✅ Vector index created successfully!")
            # Try to load the newly created data
            saved_data = load_saved_faiss_data()
            if saved_data:
                return saved_data
            else:
                raise RuntimeError("Failed to load newly created vector index")
        else:
            raise RuntimeError("Failed to create vector index")
            
    except ImportError:
        print("❌ Error: Could not import create_vector_index module")
        print("💡 Please ensure create_vector_index.py is in the same directory")
        raise FileNotFoundError(
            "Vector index creation failed. Please check that create_vector_index.py is available."
        )
    except Exception as e:
        print(f"❌ Error creating vector index: {e}")
        raise RuntimeError(f"Failed to create vector index: {e}")


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
            # Handle case where 'texts' might not be available (when loading from cache)
            text = faiss_data.get('texts', [None] * len(faiss_data['metadata']))[idx] if 'texts' in faiss_data else None
            
            result = {
                'score': float(score),
                'text': text,
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
    
    # Step 1: Extract brand/model with Gemini (with multiple identifications)
    print("📸 Step 1: Extracting brand and model with Gemini Vision...")
    extracted_data = extract_car_info_with_gemini(image_path)
    
    if 'error' in extracted_data:
        return {'error': extracted_data['error']}
    
    print("✅ Extraction completed")
    print(f"   Brand: {extracted_data.get('brand', 'Unknown')}")
    print(f"   Model: {extracted_data.get('model', 'Unknown')}")
    print(f"   Confidence: {extracted_data.get('confidence', 0)}")
    
    # Step 2: Vector search to find exact brand/model keys from our classification
    print("\n🔍 Step 2: Vector search for exact brand/model keys...")
    query = f"{extracted_data.get('brand', '')} {extracted_data.get('model', '')}".strip()
    faiss_data = setup_faiss_vector_search(vector_dataset)
    vector_results = search_vector_database(query, faiss_data, top_k=5)
    
    if vector_results:
        print(f"✅ Found {len(vector_results)} potential matches")
        best_match = vector_results[0]
        print(f"   Best match: {best_match.get('brand_label', 'Unknown')} {best_match.get('model_label', 'Unknown')}")
    else:
        print("⚠️ No vector search matches found")
        best_match = None
    
    # Step 3: Get available field values from form mappings
    print("\n📋 Step 3: Getting available field values...")
    form_mappings = vector_dataset.get('form_field_mappings', {})
    available_values = {}
    
    for field_key, field_info in form_mappings.items():
        if field_info.get('type') == 'enum':
            available_values[field_key] = [v['label'] for v in field_info.get('values', [])]
        elif field_info.get('type') == 'tree':
            available_values[field_key] = [v['label'] for v in field_info.get('values', [])]
    
    print(f"   Available values for {len(available_values)} fields")
    
    # Step 4: Extract additional details with exact available values
    print("\n🔍 Step 4: Extracting additional details with exact values...")
    additional_data = extract_additional_details_with_gemini(image_path, best_match, vector_dataset)
    
    if 'error' in additional_data:
        print(f"❌ Additional details extraction failed: {additional_data['error']}")
        additional_data = {}
    
    # Step 5: Generate form fields using vector search results + additional details
    print("\n📝 Step 5: Generating form fields...")
    ikman_form_json = generate_ikman_form_submission_json(
        {**extracted_data, **additional_data}, 
        vector_dataset, 
        {'match': best_match, 'available_values': available_values}
    )
    
    print("✅ Form JSON generation completed")
    print(f"   Generated {len(ikman_form_json['ai_generated'])} form fields")
    
    # Print cost summary at the end
    cost_tracker.print_cost_summary()
    
    return {
        'extracted_data': extracted_data,
        'additional_data': additional_data,
        'vector_results': vector_results,
        'ikman_form_submission': ikman_form_json
    }

def extract_additional_details_with_gemini(image_path, matched_car_info, vector_dataset, max_retries=3):
    """Extract additional details from image using matched car information and exact available values"""
    api_key = setup_gemini_client()
    image_base64 = encode_image_to_base64(image_path)
    
    # Create prompt with matched car context and exact available values
    brand = matched_car_info.get('brand_label', 'Unknown') if matched_car_info else 'Unknown'
    model = matched_car_info.get('model_label', 'Unknown') if matched_car_info else 'Unknown'
    
    # Get exact available values from form mappings
    form_mappings = vector_dataset.get('form_field_mappings', {})
    condition_values = [v['label'] for v in form_mappings.get('condition', {}).get('values', [])]
    body_values = [v['label'] for v in form_mappings.get('body', {}).get('values', [])]
    fuel_values = [v['label'] for v in form_mappings.get('fuel_type', {}).get('values', [])]
    transmission_values = [v['label'] for v in form_mappings.get('transmission', {}).get('values', [])]
    
    prompt = f"""
    Analyze this car {brand} {model}.
    
    Extract additional details and return ONLY a JSON object using the EXACT provided conditions:

    {{
        "condition": "condition (must be one of: {', '.join(condition_values)})",
        "body": "body type (must be one of: {', '.join(body_values)})",
        "fuel_type": "fuel type (must be one of: {', '.join(fuel_values)})",
        "transmission": "transmission type (must be one of: {', '.join(transmission_values)})",
        "mileage": "mileage in km (ONLY if dashboard/odometer is clearly visible with readable numbers, otherwise null)",
        "mileage_confidence": "confidence for mileage reading (0.0 to 1.0, null if not visible)",
        "color": "primary exterior color",
        "year": "estimated manufacturing year (YYYY format)",
        "price": "estimated price in LKR (provide reasonable estimate based on brand, model, year, and condition)",
        "engine_capacity": "engine capacity in cc (e.g., 1500, 2000, or null for electric)"
    }}

    CRITICAL: You MUST use EXACT values from the provided lists. Do not use synonyms or variations.
    Guidelines:
    - Use EXACT values from the lists provided - no exceptions
    - Only extract mileage if dashboard/odometer is clearly visible and readable
    - Evlauate if car is a hybrid or electric or petrol or diesel or CNG or LPG or hydrogen or other fuel type
    - For price estimation: Provide reasonable estimates based on brand, model, year, and condition
    - Focus on clearly visible information
    - For electric vehicles, set engine_capacity to null, and transmission to automatic
    """
    
    # Make API request to Gemini Pro
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
            "temperature": 0.0,
            "topK": 1,
            "topP": 0.95,
            "maxOutputTokens": 1024,
            "candidateCount": 1
        }
    }
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            print(f"🤖 Extracting additional details... (attempt {attempt + 1}/{max_retries})")
            import time
            start_time = time.time()
            
            # Add delay between retries to respect rate limits
            if attempt > 0:
                delay = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                print(f"⏳ Waiting {delay}s before retry...")
                time.sleep(delay)
            
            response = requests.post(url, json=payload, timeout=90)
            response.raise_for_status()
            
            api_duration = time.time() - start_time
            print(f"📊 Processing Gemini API response... (took {api_duration:.1f}s)")
            result = response.json()
            
            if 'candidates' in result and result['candidates']:
                content = result['candidates'][0]['content']['parts'][0]['text']
                
                # Extract token usage from response
                usage_info = result.get('usageMetadata', {})
                input_tokens = usage_info.get('promptTokenCount', 0)
                output_tokens = usage_info.get('candidatesTokenCount', 0)
                
                # Track API costs
                request_cost = cost_tracker.add_request(input_tokens, output_tokens, images=1, request_type="additional_details")
                print(f"💰 Additional details cost: ${request_cost:.4f}")
                
                # Debug: Print actual token counts
                print(f"🔍 Additional Details Token Debug: input={input_tokens:,}, output={output_tokens:,}")
                
                import re
                json_match = re.search(r'```json\n(\{.*?\})\n```', content, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'(\{.*?\})', content, re.DOTALL)

                if json_match:
                    try:
                        print("✅ Parsing additional details JSON response...")
                        additional_data = json.loads(json_match.group(1))
                        
                        # Add token usage to the response
                        additional_data['_token_usage'] = {
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'total_tokens': input_tokens + output_tokens,
                            'cost_usd': request_cost
                        }
                        
                        return additional_data
                    except json.JSONDecodeError:
                         print("❌ Failed to parse JSON from additional details response")
                         return {"error": "Invalid JSON in response", "raw_response": content}
                else:
                    print("❌ No JSON found in additional details response")
                    return {"error": "No JSON found in response", "raw_response": content}
            else:
                print("❌ No response from Gemini API")
                return {"error": "No response from Gemini", "result": result}
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:  # Last attempt
                return {"error": f"API request failed after {max_retries} attempts: {e}"}
            continue  # Try again
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:  # Last attempt
                return {"error": f"JSON parsing failed after {max_retries} attempts: {e}"}
            continue  # Try again
        except Exception as e:
            print(f"❌ Unexpected error (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:  # Last attempt
                return {"error": f"Unexpected error after {max_retries} attempts: {e}"}
            continue  # Try again
    
    # If we get here, all retries failed
    return {"error": f"All {max_retries} attempts failed"}

def generate_ikman_form_submission_json(extracted_data, vector_dataset, match_info):
    """Generate exact ikman.lk form submission JSON using extracted data and ikman classification"""
    
    # Get the form field mappings from our dataset
    form_mappings = vector_dataset.get('form_field_mappings', {})
    
    # Add model field to form mappings if it doesn't exist
    if 'model' not in form_mappings:
        form_mappings['model'] = {
            'type': 'tree',
            'label': 'Model',
            'values': []  # Will be populated from brand's tree structure
        }
    
    # Initialize the form submission structure with ALL fields from the mapping (except description)
    ai_generated = {field_key: "manual_fill_required" for field_key in form_mappings.keys() if field_key != 'description'}
    
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
            # Extract only the trim level from the model name
            model = extracted_data.get('model', '')
            brand = extracted_data.get('brand', '')
            
            # Common trim levels to extract
            trim_levels = ['Sport', 'Limited', 'Premium', 'SE', 'LE', 'XLE', 'SR', 'SR5', 'TRD', 'GT', 'GTS', 'RS', 'S', 'X', 'LX', 'EX', 'DX', 'GL', 'GLX', 'GLS', 'GLI', 'TDI', 'TSI', 'GTI', 'R', 'AMG', 'M', 'S', 'RS', 'Quattro', '4Matic', 'xDrive', 'AWD', '4WD']
            
            if model:
                # Try to extract trim level from model name
                model_lower = model.lower()
                for trim in trim_levels:
                    if trim.lower() in model_lower:
                        return trim
                
                # If no trim found, try to extract the last word as trim
                model_words = model.split()
                if len(model_words) > 1:
                    last_word = model_words[-1]
                    # Check if last word looks like a trim level
                    if len(last_word) <= 6 and last_word.isupper() or last_word in trim_levels:
                        return last_word
                
                # If still no trim, return a generic edition
                return "Standard"
            elif brand and len(brand) <= max_length:
                return f"{brand} Edition"
            else:
                return "Standard"
        

    
    # Debug: Print available form fields
    print(f"🔍 Available form fields: {list(form_mappings.keys())}")
    print(f"🔍 Extracted model: {extracted_data.get('model')}")
    
    # Map extracted data to form fields using ikman classification
    for field_key, field_info in form_mappings.items():
        field_type = field_info.get('type', 'text')
        
        if field_type == 'enum':
            # Handle enum fields using extracted data and ikman classification
            if field_key == 'condition':
                condition = extracted_data.get('condition')
                if condition:
                    condition_values = field_info.get('values', [])
                    matched_condition = fuzzy_match_string(condition, condition_values, threshold=0.5)
                    if matched_condition:
                        ai_generated[field_key] = matched_condition['key']
                    else:
                        # Condition was extracted but couldn't be matched to form values
                        ai_generated[field_key] = "manual_fill_required"
                else:
                    # No condition extracted
                    ai_generated[field_key] = "manual_fill_required"
            
            elif field_key == 'body':
                body = extracted_data.get('body')
                if body:
                    body_values = field_info.get('values', [])
                    matched_body = fuzzy_match_string(body, body_values, threshold=0.5)
                    if matched_body:
                        ai_generated[field_key] = matched_body['key']
                    else:
                        # Body type was extracted but couldn't be matched to form values
                        ai_generated[field_key] = "manual_fill_required"
                else:
                    # No body type extracted
                    ai_generated[field_key] = "manual_fill_required"
            
            elif field_key == 'fuel_type':
                fuel_type = extracted_data.get('fuel_type')
                if fuel_type:
                    fuel_values = field_info.get('values', [])
                    matched_fuel = fuzzy_match_string(fuel_type, fuel_values, threshold=0.5)
                    if matched_fuel:
                        ai_generated[field_key] = matched_fuel['key']
                    else:
                        # Fuel type was extracted but couldn't be matched to form values
                        ai_generated[field_key] = "manual_fill_required"
                else:
                    # No fuel type extracted
                    ai_generated[field_key] = "manual_fill_required"
            
            elif field_key == 'transmission':
                transmission = extracted_data.get('transmission')
                if transmission:
                    transmission_values = field_info.get('values', [])
                    matched_transmission = fuzzy_match_string(transmission, transmission_values, threshold=0.5)
                    if matched_transmission:
                        ai_generated[field_key] = matched_transmission['key']
                    else:
                        # Transmission was extracted but couldn't be matched to form values
                        ai_generated[field_key] = "manual_fill_required"
                else:
                    # No transmission extracted
                    ai_generated[field_key] = "manual_fill_required"
        
        elif field_type == 'tree':
            # Handle tree fields (brand and model)
            if field_key == 'brand':
                # Use brand from extracted data
                brand = extracted_data.get('brand')
                if brand:
                    brand_values = field_info.get('values', [])
                    matched_brand = fuzzy_match_string(brand, brand_values, threshold=0.5)
                    if matched_brand:
                        ai_generated[field_key] = matched_brand['key']
                    else:
                        # Brand was extracted but couldn't be matched to form values
                        ai_generated[field_key] = "manual_fill_required"
                else:
                    # No brand extracted
                    ai_generated[field_key] = "manual_fill_required"
            
            elif field_key == 'model':
                # Use model from extracted data
                model = extracted_data.get('model')
                brand = extracted_data.get('brand', '')
                print(f"🔍 Processing model field: {model} for brand: {brand}")
                
                if model and brand:
                    # Search vector dataset for brand-model combination
                    search_query = f"{brand.lower()}-{model.lower()}"
                    print(f"🔍 Searching vector dataset for: {search_query}")
                    
                    # Setup FAISS and search
                    faiss_data = setup_faiss_vector_search(vector_dataset)
                    search_results = search_vector_database(search_query, faiss_data, top_k=5)
                    
                    # Look for brand-model match with fuzzy matching
                    matched_model_key = None
                    print(f"🔍 Checking {len(search_results)} search results...")
                    
                    for i, result in enumerate(search_results):
                        metadata = result['metadata']
                        result_brand = metadata.get('brand_label', '').lower()
                        result_model = metadata.get('model_label', '').lower()
                        search_brand = brand.lower()
                        search_model = model.lower()
                        
                        print(f"   {i+1}. {result_brand} {result_model} (type: {metadata.get('type')})")
                        
                        # Check if brand matches exactly
                        if (metadata.get('type') == 'brand_model' and 
                            result_brand == search_brand):
                            
                            # Check for model match (exact or partial)
                            model_match = False
                            if result_model == search_model:
                                # Exact match
                                model_match = True
                                match_type = "exact"
                            elif search_model in result_model or result_model in search_model:
                                # Partial match (e.g., "C3" in "e-c3" or "e-c3" in "C3")
                                model_match = True
                                match_type = "partial"
                            elif any(word in result_model for word in search_model.split()) or any(word in search_model for word in result_model.split()):
                                # Word overlap (e.g., "Pajero Sport" vs "Pajero")
                                model_match = True
                                match_type = "word_overlap"
                            
                            if model_match:
                                matched_model_key = metadata.get('model_key')
                                print(f"✅ Found {match_type} match: {metadata.get('brand_label')} {metadata.get('model_label')} -> key: {matched_model_key}")
                                break
                    
                    if matched_model_key:
                        ai_generated[field_key] = matched_model_key
                        print(f"✅ Using vector DB key for model: {matched_model_key}")
                    else:
                        print(f"❌ No exact match found in vector DB for: {brand} {model}")
                        # Try fuzzy matching with form values if available
                        model_values = field_info.get('values', [])
                        if model_values:
                            matched_model = fuzzy_match_string(model, model_values, threshold=0.5)
                            if matched_model:
                                ai_generated[field_key] = matched_model['key']
                                print(f"✅ Using fuzzy match key for model: {matched_model['key']}")
                            else:
                                ai_generated[field_key] = "manual_fill_required"
                                print(f"❌ No fuzzy match found, setting to manual_fill_required")
                        else:
                            # No predefined model values, set to manual_fill_required
                            ai_generated[field_key] = "manual_fill_required"
                            print(f"❌ No model values available, setting to manual_fill_required")
                else:
                    # No model or brand extracted
                    ai_generated[field_key] = "manual_fill_required"
        
        elif field_type == 'year':
            # Handle year fields
            if field_key == 'model_year':
                year = extracted_data.get('year')
                if year:
                    try:
                        year_int = int(year)
                        min_year = field_info.get('constraints', {}).get('min_year', 1926)
                        max_year = field_info.get('constraints', {}).get('max_year', 2026)
                        
                        if min_year <= year_int <= max_year:
                            ai_generated[field_key] = year_int
                    except (ValueError, TypeError):
                        pass
        
        elif field_type == 'measurement':
            # Handle measurement fields
            if field_key == 'mileage':
                mileage = extracted_data.get('mileage')
                mileage_confidence = extracted_data.get('mileage_confidence', 0)
                
                # Only use mileage if it's provided AND confidence is very high (0.9+)
                if mileage and mileage_confidence and mileage_confidence >= 0.9:
                    try:
                        mileage_int = int(mileage)
                        min_mileage = field_info.get('constraints', {}).get('minimum', 0)
                        max_mileage = field_info.get('constraints', {}).get('maximum', 1000000)
                        
                        if min_mileage <= mileage_int <= max_mileage:
                            ai_generated[field_key] = mileage_int
                    except (ValueError, TypeError):
                        pass
            
            elif field_key == 'engine_capacity':
                engine_capacity = extracted_data.get('engine_capacity')
                if engine_capacity:
                    try:
                        capacity_int = int(engine_capacity)
                        min_capacity = field_info.get('constraints', {}).get('minimum', 0)
                        max_capacity = field_info.get('constraints', {}).get('maximum', 10000)
                        
                        if min_capacity <= capacity_int <= max_capacity:
                            ai_generated[field_key] = capacity_int
                    except (ValueError, TypeError):
                        pass
        
        elif field_type == 'money':
            # Handle money fields
            if field_key == 'price':
                price = extracted_data.get('price')
                if price:
                    try:
                        price_int = int(price)
                        ai_generated[field_key] = price_int
                    except (ValueError, TypeError):
                        pass
                else:
                    # Provide fallback price estimate based on car characteristics
                    brand = extracted_data.get('brand', '').lower()
                    condition = extracted_data.get('condition', '').lower()
                    year = extracted_data.get('year')
                    
                    # Estimate price based on brand and condition
                    if 'bmw' in brand or 'mercedes' in brand or 'audi' in brand:
                        # Luxury brand
                        if 'new' in condition:
                            estimated_price = 12000000  # 12M LKR for new luxury
                        else:
                            estimated_price = 5500000   # 5.5M LKR for used luxury
                    elif 'toyota' in brand or 'honda' in brand or 'nissan' in brand:
                        # Mainstream brand
                        if 'new' in condition:
                            estimated_price = 8500000   # 8.5M LKR for new mainstream
                        else:
                            estimated_price = 3500000   # 3.5M LKR for used mainstream
                    else:
                        # Other brands
                        if 'new' in condition:
                            estimated_price = 7000000   # 7M LKR for new other
                        else:
                            estimated_price = 3000000   # 3M LKR for used other
                    
                    # Adjust for year if available
                    if year and isinstance(year, (int, str)):
                        try:
                            year_int = int(year)
                            current_year = 2024
                            age = current_year - year_int
                            if age > 0:
                                # Reduce price by 10% per year for used cars
                                depreciation = min(age * 0.1, 0.5)  # Max 50% depreciation
                                estimated_price = int(estimated_price * (1 - depreciation))
                        except (ValueError, TypeError):
                            pass
                    
                    ai_generated[field_key] = estimated_price
        
        elif field_type == 'text':
            # Handle text fields (skip description)
            if field_key != 'description':  # Skip description generation
                generated_text = generate_text_field_value(field_key, field_info, extracted_data)
                ai_generated[field_key] = generated_text
    
    # Check if model was extracted but not in form mappings
    manual_required = [k for k, v in ai_generated.items() if v == "manual_fill_required"]
    
    # If model was extracted but not in form mappings, add it to manual_required
    extracted_model = extracted_data.get('model')
    if extracted_model and 'model' not in form_mappings:
        manual_required.append('model')
    
    # Return the form submission JSON
    return {
        'ai_generated': ai_generated,
        'manual_required': manual_required,
        # 'extracted_data': extracted_data,
        # 'match_info': match_info
    }


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


def process_multiple_images(image_paths, vector_dataset):
    """Process multiple car images"""
    results = {}
    successful_count = 0
    failed_count = 0
    
    print(f"🚗 Starting batch processing of {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n📸 Processing image {i}/{len(image_paths)}: {image_path}")
        
        try:
            result = process_car_image_end_to_end(image_path, vector_dataset, add_delay=False)
            
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
    
    print(f"\n📊 Batch processing complete!")
    print(f"✅ Successful: {successful_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📈 Success rate: {(successful_count / len(image_paths)) * 100:.1f}%")
    
    return results

def process_car_images_batch(image_paths, vector_dataset, batch_size=3, max_retries=3):
    """Process multiple car images in batches for cost efficiency"""
    
    if len(image_paths) <= batch_size:
        # Small batch - use individual processing for accuracy
        print(f"📸 Processing {len(image_paths)} images individually...")
        return process_multiple_images(image_paths, vector_dataset)
    
    print(f"🚀 Processing {len(image_paths)} images in batches of {batch_size}...")
    
    api_key = setup_gemini_client()
    all_results = {}
    
    # Process in batches
    for batch_idx in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_idx:batch_idx + batch_size]
        print(f"\n📦 Processing batch {batch_idx//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")
        print(f"   Images: {[os.path.basename(p) for p in batch_paths]}")
        
        # Encode all images in batch
        batch_images = []
        for image_path in batch_paths:
            try:
                image_base64 = encode_image_to_base64(image_path)
                batch_images.append(image_base64)
            except Exception as e:
                print(f"❌ Error encoding {image_path}: {e}")
                all_results[image_path] = {"error": f"Image encoding failed: {e}"}
                continue
        
        if not batch_images:
            continue
        
        # Create batch prompt - ONLY brand and model identification
        batch_prompt = f"""
        Analyze these {len(batch_images)} car images and extract ONLY brand and model for each.
        Return ONLY a JSON array with one object per image in the same order:

        [
            {{
                "image_index": 0,
                "brand": "car brand (e.g., Toyota, Honda, BMW)",
                "model": "car model (e.g., Corolla, Civic, 3 Series)",
                "confidence": "confidence level (0.0 to 1.0)"
            }},
            // ... one object per image in order
        ]

        Guidelines:
        - Be specific with model names (e.g., "Corolla" not just "Toyota")
        - If you can't identify clearly, use "unknown" for brand or model
        - Focus on the most prominent car in each image
        - Return exactly {len(batch_images)} objects in the array
        - ONLY extract brand and model - no other details
        """
        
        # Prepare batch payload
        parts = [{"text": batch_prompt}]
        for image_base64 in batch_images:
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64
                }
            })
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.0,
                "topK": 1,
                "topP": 0.95,
                "maxOutputTokens": 2048,
                "candidateCount": 1
            }
        }
        
        # Make batch API request with retry logic
        batch_success = False
        for attempt in range(max_retries):
            try:
                print(f"🤖 Sending batch request... (attempt {attempt + 1}/{max_retries})")
                start_time = time.time()
                
                if attempt > 0:
                    delay = min(2 ** attempt, 10)
                    print(f"⏳ Waiting {delay}s before retry...")
                    time.sleep(delay)
                
                response = requests.post(url, json=payload, timeout=120)  # Increased timeout for batch
                response.raise_for_status()
                
                api_duration = time.time() - start_time
                print(f"📊 Processing batch response... (took {api_duration:.1f}s)")
                result = response.json()
                
                if 'candidates' in result and result['candidates']:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    
                    # Extract token usage
                    usage_info = result.get('usageMetadata', {})
                    input_tokens = usage_info.get('promptTokenCount', 0)
                    output_tokens = usage_info.get('candidatesTokenCount', 0)
                    
                    # Track API costs
                    request_cost = cost_tracker.add_request(input_tokens, output_tokens, images=len(batch_images), request_type="batch_processing")
                    print(f"💰 Batch cost: ${request_cost:.4f} ({len(batch_images)} images)")
                    
                    # Parse JSON response
                    import re
                    json_match = re.search(r'```json\n(\[.*?\])\n```', content, re.DOTALL)
                    if not json_match:
                        json_match = re.search(r'(\[.*?\])', content, re.DOTALL)
                    
                    if json_match:
                        try:
                            batch_data = json.loads(json_match.group(1))
                            
                            if isinstance(batch_data, list) and len(batch_data) == len(batch_paths):
                                print(f"✅ Successfully processed batch of {len(batch_data)} images")
                                
                                # Distribute results to individual image paths
                                for i, (image_path, car_data) in enumerate(zip(batch_paths, batch_data)):
                                    # Add token usage and cost info
                                    car_data['_token_usage'] = {
                                        'input_tokens': input_tokens // len(batch_images),  # Approximate per image
                                        'output_tokens': output_tokens // len(batch_images),
                                        'total_tokens': (input_tokens + output_tokens) // len(batch_images),
                                        'cost_usd': request_cost / len(batch_images)
                                    }
                                    
                                    # Process with vector search
                                    query = f"{car_data.get('brand', '')} {car_data.get('model', '')}".strip()
                                    faiss_data = setup_faiss_vector_search(vector_dataset)
                                    vector_results = search_vector_database(query, faiss_data, top_k=3)
                                    
                                    # Extract additional details with exact available values
                                    best_match = vector_results[0] if vector_results else None
                                    additional_data = extract_additional_details_with_gemini(image_path, best_match, vector_dataset)
                                    
                                    if 'error' in additional_data:
                                        print(f"❌ Additional details extraction failed for {os.path.basename(image_path)}: {additional_data['error']}")
                                        additional_data = {}
                                    
                                    # Generate form JSON with combined data
                                    combined_data = {**car_data, **additional_data}
                                    ikman_form_json = generate_ikman_form_submission_json(combined_data, vector_dataset, {'match': best_match})
                                    
                                    all_results[image_path] = {
                                        'extracted_data': car_data,
                                        'additional_data': additional_data,
                                        'vector_results': vector_results,
                                        'ikman_form_submission': ikman_form_json
                                    }
                                    
                                    brand = car_data.get('brand', 'Unknown')
                                    model = car_data.get('model', 'Unknown')
                                    confidence = car_data.get('confidence', 0)
                                    print(f"   ✅ {os.path.basename(image_path)}: {brand} {model} (confidence: {confidence:.2f})")
                                
                                batch_success = True
                                break
                            else:
                                print(f"❌ Invalid batch response format: expected {len(batch_paths)} items, got {len(batch_data) if isinstance(batch_data, list) else 'non-list'}")
                                
                        except json.JSONDecodeError as e:
                            print(f"❌ Failed to parse batch JSON: {e}")
                            print(f"Raw response: {content[:200]}...")
                    else:
                        print("❌ No JSON array found in batch response")
                        
                else:
                    print("❌ No response from Gemini API")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Batch API request failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # Fallback to individual processing for this batch
                    print(f"🔄 Falling back to individual processing for batch...")
                    batch_results = process_multiple_images(batch_paths, vector_dataset)
                    all_results.update(batch_results)
            except Exception as e:
                print(f"❌ Unexpected error in batch processing (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    # Fallback to individual processing for this batch
                    print(f"🔄 Falling back to individual processing for batch...")
                    batch_results = process_multiple_images(batch_paths, vector_dataset)
                    all_results.update(batch_results)
        
        if not batch_success:
            print(f"❌ Batch processing failed, using individual processing...")
            batch_results = process_multiple_images(batch_paths, vector_dataset)
            all_results.update(batch_results)
    
    # Print final cost summary
    cost_tracker.print_cost_summary()
    
    return all_results


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
        print(f"✅ FAISS index ready with {faiss_data['index'].ntotal} vectors")
        print("💾 Using cached embeddings (faster startup)")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 To create the vector index, run:")
        print("   python create_vector_index.py")
        exit(1)
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
    print("1. Run 'streamlit run streamlit_app_v2.py' to start the web interface")
    print("2. Or use the functions directly in your code:")
    print("   from main import process_car_image_end_to_end")
    print("   result = process_car_image_end_to_end('your_image.jpg', vector_dataset)")
    print("3. For batch processing (cost efficient):")
    print("   from main import process_car_images_batch")
    print("   results = process_car_images_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'], vector_dataset)")