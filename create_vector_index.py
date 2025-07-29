#!/usr/bin/env python3
"""
Standalone script to create and save FAISS vector index
This script can be run independently to generate cached embeddings.
"""

import json
import os
import pickle
import numpy as np
from pathlib import Path

def load_env_vars():
    """Load environment variables from .env file"""
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"').strip("' ")

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

def create_vector_index(vector_dataset_path, output_dir="."):
    """
    Create and save FAISS vector index from vector dataset
    
    Args:
        vector_dataset_path (str): Path to car_vector_dataset.json
        output_dir (str): Directory to save cached files (default: current directory)
    """
    print("🚀 Starting Vector Index Creation")
    print("=" * 50)
    
    # Check if vector dataset exists
    if not os.path.exists(vector_dataset_path):
        print(f"❌ Vector dataset not found at {vector_dataset_path}")
        return False
    
    # Load vector dataset
    try:
        with open(vector_dataset_path, 'r', encoding='utf-8') as f:
            vector_dataset = json.load(f)
        print(f"✅ Vector dataset loaded: {vector_dataset['metadata']['total_entries']} entries")
    except Exception as e:
        print(f"❌ Error loading vector dataset: {e}")
        return False
    
    # Import required libraries
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
    print("\n📊 Preparing text entries for vector indexing...")
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
    print("\n🔍 Creating TF-IDF vectorizer...")
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
    
    # Save the index and vectorizer
    print("\n💾 Saving vector index and vectorizer...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save FAISS index
    index_path = os.path.join(output_dir, "faiss_index.bin")
    faiss.write_index(index, index_path)
    print(f"✅ Saved FAISS index to: {index_path}")
    
    # Save TF-IDF vectorizer
    vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f"✅ Saved TF-IDF vectorizer to: {vectorizer_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "faiss_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✅ Saved metadata to: {metadata_path}")
    
    # Test the saved index
    print("\n🧪 Testing saved index...")
    try:
        # Load the saved index
        test_index = faiss.read_index(index_path)
        
        # Test a simple query
        test_query = "BMW 3 Series"
        normalized_query = normalize_text_for_search(test_query)
        query_vector = tfidf_vectorizer.transform([normalized_query]).toarray().astype(np.float32)
        faiss.normalize_L2(query_vector)
        
        scores, indices = test_index.search(query_vector, 3)
        
        if len(indices[0]) > 0:
            print(f"✅ Index test successful: Found {len(indices[0])} results for '{test_query}'")
            print(f"   Best match score: {scores[0][0]:.3f}")
        else:
            print("⚠️ Index test: No results found")
            
    except Exception as e:
        print(f"❌ Index test failed: {e}")
        return False
    
    # Print summary
    print("\n📊 Summary:")
    print(f"   Total vectors: {index.ntotal}")
    print(f"   Vector dimension: {dimension}")
    print(f"   Output directory: {output_dir}")
    print(f"   Files created:")
    print(f"     - {index_path}")
    print(f"     - {vectorizer_path}")
    print(f"     - {metadata_path}")
    
    print("\n🎉 Vector index creation completed successfully!")
    return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create FAISS vector index from car dataset')
    parser.add_argument('--dataset', '-d', 
                       default='car_vector_dataset.json',
                       help='Path to car_vector_dataset.json (default: car_vector_dataset.json)')
    parser.add_argument('--output', '-o',
                       default='.',
                       help='Output directory for cached files (default: current directory)')
    parser.add_argument('--force', '-f',
                       action='store_true',
                       help='Force recreation even if cache exists')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_env_vars()
    
    # Check if cache already exists
    cache_files = [
        os.path.join(args.output, "faiss_index.bin"),
        os.path.join(args.output, "tfidf_vectorizer.pkl"),
        os.path.join(args.output, "faiss_metadata.pkl")
    ]
    
    existing_files = [f for f in cache_files if os.path.exists(f)]
    
    if existing_files and not args.force:
        print("⚠️ Cache files already exist:")
        for f in existing_files:
            print(f"   - {f}")
        print("\nUse --force to recreate the cache")
        return
    
    if args.force and existing_files:
        print("🗑️ Removing existing cache files...")
        for f in existing_files:
            try:
                os.remove(f)
                print(f"   Removed: {f}")
            except Exception as e:
                print(f"   Error removing {f}: {e}")
    
    # Create the vector index
    success = create_vector_index(args.dataset, args.output)
    
    if success:
        print("\n✅ Vector index creation completed!")
        print(f"💡 You can now use the cached embeddings for faster startup")
    else:
        print("\n❌ Vector index creation failed!")
        exit(1)

if __name__ == "__main__":
    main() 