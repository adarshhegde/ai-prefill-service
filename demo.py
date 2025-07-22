#!/usr/bin/env python3
"""
Demo script for AI Car Autofill Service
This script demonstrates how to use the system programmatically
"""

import json
import os
import sys
from pathlib import Path

# Import our functions
from main import (
    process_car_image_end_to_end,
    setup_faiss_vector_search,
    search_vector_database,
    load_env_vars
)

def demo_vector_search():
    """Demonstrate vector search functionality"""
    print("🔍 Vector Search Demo")
    print("=" * 40)
    
    # Load vector dataset
    try:
        with open('car_vector_dataset.json', 'r', encoding='utf-8') as f:
            vector_dataset = json.load(f)
        print(f"✅ Loaded vector dataset: {vector_dataset['metadata']['total_entries']} entries")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Setup FAISS
    try:
        faiss_data = setup_faiss_vector_search(vector_dataset)
        print(f"✅ FAISS index ready: {faiss_data['index'].ntotal} vectors")
    except Exception as e:
        print(f"❌ FAISS setup failed: {e}")
        return False
    
    # Test searches
    test_queries = [
        "BMW 3 Series 2018",
        "Toyota Camry Hybrid",
        "Honda Civic Type R",
        "Mercedes C-Class AMG"
    ]
    
    print("\n🔍 Testing searches:")
    for query in test_queries:
        results = search_vector_database(query, faiss_data, top_k=2)
        if results:
            best = results[0]
            brand = best['metadata'].get('brand_label', 'Unknown')
            model = best['metadata'].get('model_label', 'Unknown')
            confidence = best['score']
            print(f"   '{query}' → {brand} {model} ({confidence:.3f})")
        else:
            print(f"   '{query}' → No matches")
    
    return True

def demo_form_field_mapping():
    """Demonstrate form field mapping"""
    print("\n📋 Form Field Mapping Demo")
    print("=" * 40)
    
    # Load vector dataset
    with open('car_vector_dataset.json', 'r', encoding='utf-8') as f:
        vector_dataset = json.load(f)
    
    # Show available form fields
    form_mappings = vector_dataset.get('form_field_mappings', {})
    
    print("Available form fields:")
    for field_name, field_info in form_mappings.items():
        field_type = field_info.get('type', 'unknown')
        required = field_info.get('required', False)
        values_count = len(field_info.get('values', []))
        
        status = "✅ Required" if required else "📝 Optional"
        print(f"   {field_name}: {field_type} ({values_count} options) - {status}")
    
    # Show sample values for key fields
    print("\nSample field values:")
    
    # Condition field
    condition_values = form_mappings.get('condition', {}).get('values', [])
    print(f"   Condition: {[v['label'] for v in condition_values]}")
    
    # Body types
    body_values = form_mappings.get('body', {}).get('values', [])
    print(f"   Body types: {[v['label'] for v in body_values]}")
    
    # Fuel types
    fuel_values = form_mappings.get('fuel_type', {}).get('values', [])
    print(f"   Fuel types: {[v['label'] for v in fuel_values]}")
    
    return True

def demo_image_processing(image_path=None):
    """Demonstrate image processing (requires API key)"""
    print("\n📸 Image Processing Demo")
    print("=" * 40)
    
    # Check if API key is available
    load_env_vars()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ No Gemini API key found. Skipping image processing demo.")
        print("   To test image processing:")
        print("   1. Get API key from https://ai.google.dev/")
        print("   2. Create .env file with: GEMINI_API_KEY=your_key")
        print("   3. Run this demo again")
        return False
    
    # Check if image path is provided
    if not image_path:
        print("❌ No image path provided. Skipping image processing demo.")
        print("   To test with an image:")
        print("   python demo.py path/to/car_image.jpg")
        return False
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    print(f"📸 Processing image: {image_path}")
    
    # Load vector dataset
    try:
        with open('car_vector_dataset.json', 'r', encoding='utf-8') as f:
            vector_dataset = json.load(f)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Process image
    try:
        print("🔄 Processing... (this may take a few seconds)")
        result = process_car_image_end_to_end(image_path, vector_dataset)
        
        if 'error' in result:
            print(f"❌ Processing failed: {result['error']}")
            return False
        
        # Display results
        print("\n✅ Processing completed!")
        
        # Show extracted data
        extracted = result['extracted_data']
        print(f"\n📋 Extracted Information:")
        print(f"   Brand: {extracted.get('brand', 'Unknown')}")
        print(f"   Model: {extracted.get('model', 'Unknown')}")
        print(f"   Year: {extracted.get('year', 'Unknown')}")
        print(f"   Condition: {extracted.get('condition', 'Unknown')}")
        print(f"   Body Type: {extracted.get('body_type', 'Unknown')}")
        print(f"   Fuel Type: {extracted.get('fuel_type', 'Unknown')}")
        print(f"   Confidence: {extracted.get('confidence', 0):.2f}")
        
        # Show form submission data
        if 'ikman_form_submission' in result:
            form_data = result['ikman_form_submission']
            print(f"\n📝 Generated Form Data:")
            for field, value in form_data.items():
                if not field.startswith('_'):
                    print(f"   {field}: {value}")
        
        # Show field statistics
        if 'field_statistics' in result:
            stats = result['field_statistics']
            ai_fields = len(stats.get('ai_prefilled', []))
            manual_fields = len(stats.get('manual_required', []))
            total_fields = stats.get('total_fields', ai_fields + manual_fields)
            
            print(f"\n📊 Field Statistics:")
            print(f"   AI Pre-filled: {ai_fields}/{total_fields}")
            print(f"   Manual Required: {manual_fields}/{total_fields}")
            print(f"   Automation Rate: {(ai_fields/total_fields)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return False

def main():
    """Main demo function"""
    print("🚗 AI Car Autofill Service - Demo")
    print("=" * 50)
    
    # Parse command line arguments
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    # Run demos
    success_count = 0
    total_demos = 3
    
    # Demo 1: Vector Search
    if demo_vector_search():
        success_count += 1
    
    # Demo 2: Form Field Mapping
    if demo_form_field_mapping():
        success_count += 1
    
    # Demo 3: Image Processing (if image provided and API key available)
    if demo_image_processing(image_path):
        success_count += 1
    elif image_path:
        # If image was provided but processing failed, don't count as demo
        total_demos = 2
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEMO SUMMARY")
    print("=" * 50)
    
    if success_count == total_demos:
        print("🎉 All demos completed successfully!")
        print("\n🚀 Ready to use the system:")
        print("   1. Web Interface: streamlit run streamlit_app.py")
        print("   2. Programmatic: from main import process_car_image_end_to_end")
        print("   3. Batch Processing: Use process_multiple_images")
    else:
        print(f"⚠️  {success_count}/{total_demos} demos completed successfully")
        if not image_path:
            print("   To test image processing, provide an image path:")
            print("   python demo.py path/to/car_image.jpg")
    
    print("\n📚 For more information:")
    print("   - Setup Guide: SETUP.md")
    print("   - Documentation: README.md")
    print("   - Vector Search Test: python test_vector_search.py")

if __name__ == "__main__":
    main() 