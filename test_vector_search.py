#!/usr/bin/env python3
"""
Test script for vector search functionality
This script tests the core vector search capabilities without requiring API keys
"""

import json
import os
from main import (
    setup_faiss_vector_search,
    search_vector_database,
    normalize_text_for_search
)

def test_vector_search():
    """Test the vector search functionality"""
    print("🔍 Testing Vector Search Functionality")
    print("=" * 50)
    
    # Check if vector dataset exists
    if not os.path.exists('car_vector_dataset.json'):
        print("❌ Vector dataset not found. Please ensure car_vector_dataset.json exists.")
        return False
    
    # Load vector dataset
    try:
        with open('car_vector_dataset.json', 'r', encoding='utf-8') as f:
            vector_dataset = json.load(f)
        print(f"✅ Vector dataset loaded: {vector_dataset['metadata']['total_entries']} entries")
    except Exception as e:
        print(f"❌ Error loading vector dataset: {e}")
        return False
    
    # Test FAISS setup
    try:
        print("🔍 Setting up FAISS vector search...")
        faiss_data = setup_faiss_vector_search(vector_dataset)
        print(f"✅ FAISS index created with {faiss_data['index'].ntotal} vectors")
    except Exception as e:
        print(f"❌ FAISS setup failed: {e}")
        return False
    
    # Test various search queries
    test_queries = [
        "BMW 3 Series",
        "Toyota Camry",
        "Honda Civic",
        "Mercedes C-Class",
        "Audi A4",
        "Nissan Altima",
        "Ford Focus",
        "Volkswagen Golf",
        "Hyundai Elantra",
        "Kia Optima"
    ]
    
    print("\n🔍 Testing search queries...")
    successful_searches = 0
    
    for query in test_queries:
        try:
            results = search_vector_database(query, faiss_data, top_k=3)
            if results:
                best_match = results[0]
                brand = best_match['metadata'].get('brand_label', 'Unknown')
                model = best_match['metadata'].get('model_label', 'Unknown')
                confidence = best_match['score']
                print(f"✅ '{query}' → {brand} {model} (confidence: {confidence:.3f})")
                successful_searches += 1
            else:
                print(f"⚠️  '{query}' → No matches found")
        except Exception as e:
            print(f"❌ '{query}' → Error: {e}")
    
    print(f"\n📊 Search Test Results:")
    print(f"   Successful searches: {successful_searches}/{len(test_queries)}")
    print(f"   Success rate: {(successful_searches/len(test_queries))*100:.1f}%")
    
    # Test text normalization
    print("\n🔍 Testing text normalization...")
    test_texts = [
        "BMW 3-Series",
        "TOYOTA CAMRY",
        "Mercedes-Benz C-Class",
        "Audi A4 2.0T",
        "Honda Civic Type-R"
    ]
    
    for text in test_texts:
        normalized = normalize_text_for_search(text)
        print(f"   '{text}' → '{normalized}'")
    
    return successful_searches >= len(test_queries) * 0.8  # 80% success rate

def test_form_field_mapping():
    """Test form field mapping functionality"""
    print("\n📋 Testing Form Field Mapping")
    print("=" * 50)
    
    # Load vector dataset
    try:
        with open('car_vector_dataset.json', 'r', encoding='utf-8') as f:
            vector_dataset = json.load(f)
    except Exception as e:
        print(f"❌ Error loading vector dataset: {e}")
        return False
    
    # Check form field mappings
    form_mappings = vector_dataset.get('form_field_mappings', {})
    
    required_fields = ['condition', 'brand', 'body', 'fuel_type', 'transmission']
    available_fields = list(form_mappings.keys())
    
    print(f"📋 Form fields available: {len(available_fields)}")
    print(f"   Required fields: {required_fields}")
    print(f"   Available fields: {available_fields}")
    
    # Check if all required fields are present
    missing_fields = [field for field in required_fields if field not in available_fields]
    if missing_fields:
        print(f"❌ Missing required fields: {missing_fields}")
        return False
    else:
        print("✅ All required form fields are present")
    
    # Test field value options
    for field_name in required_fields:
        field_info = form_mappings.get(field_name, {})
        field_type = field_info.get('type', 'unknown')
        values = field_info.get('values', [])
        print(f"   {field_name}: {field_type} ({len(values)} options)")
    
    return True

def main():
    """Main test function"""
    print("🚗 AI Car Autofill Service - Vector Search Test")
    print("=" * 60)
    
    # Test vector search
    vector_search_ok = test_vector_search()
    
    # Test form field mapping
    form_mapping_ok = test_form_field_mapping()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if vector_search_ok and form_mapping_ok:
        print("🎉 All tests passed! The system is ready for AI integration.")
        print("\nNext steps:")
        print("1. Create a .env file with your GEMINI_API_KEY")
        print("2. Run 'python main.py' to test the complete system")
        print("3. Run 'streamlit run streamlit_app.py' to start the web interface")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        if not vector_search_ok:
            print("   - Vector search functionality needs attention")
        if not form_mapping_ok:
            print("   - Form field mapping needs attention")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 