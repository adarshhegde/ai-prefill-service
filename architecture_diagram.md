# 🚗 AI Car Autofill Service - Architecture Diagram

## System Overview

```mermaid
graph TB
    %% User Interface Layer
    subgraph "🌐 User Interface"
        UI[Streamlit Web App<br/>streamlit_app_v2.py]
        UPLOAD[Image Upload<br/>Multiple Formats]
        URL_EXTRACT[ikman.lk URL<br/>Extraction]
    end

    %% Core Processing Layer
    subgraph "🤖 AI Processing Engine"
        GEMINI[Gemini Vision AI<br/>Image Analysis]
        VECTOR_SEARCH[FAISS Vector Search<br/>Similarity Matching]
        FUZZY_MATCH[Fuzzy Field Mapping<br/>Form Field Matching]
    end

    %% Data Layer
    subgraph "💾 Data Storage"
        VECTOR_DB[Vector Dataset<br/>car_vector_dataset.json]
        CACHE[FAISS Cache<br/>faiss_index.bin]
        METADATA[Metadata Cache<br/>faiss_metadata.pkl]
        VECTORIZER[TF-IDF Vectorizer<br/>tfidf_vectorizer.pkl]
    end

    %% External Services
    subgraph "🌍 External Services"
        GEMINI_API[Gemini Vision API<br/>Google AI]
        IKMAN_API[ikman.lk API<br/>Ad Data Extraction]
    end

    %% Output Layer
    subgraph "📋 Output Generation"
        FORM_JSON[ikman.lk Form JSON<br/>Ready for Submission]
        ANALYSIS_RESULTS[Analysis Results<br/>Confidence Scores]
        DOWNLOAD[Download Options<br/>JSON/Complete Data]
    end

    %% Data Flow
    UI --> UPLOAD
    UI --> URL_EXTRACT
    UPLOAD --> GEMINI
    URL_EXTRACT --> IKMAN_API
    IKMAN_API --> UPLOAD
    GEMINI --> VECTOR_SEARCH
    VECTOR_SEARCH --> VECTOR_DB
    VECTOR_SEARCH --> CACHE
    VECTOR_SEARCH --> METADATA
    VECTOR_SEARCH --> VECTORIZER
    VECTOR_SEARCH --> FUZZY_MATCH
    FUZZY_MATCH --> FORM_JSON
    FUZZY_MATCH --> ANALYSIS_RESULTS
    ANALYSIS_RESULTS --> DOWNLOAD
    FORM_JSON --> DOWNLOAD
```

## Detailed Component Architecture

```mermaid
graph TB
    %% Main Application Components
    subgraph "📱 Main Application"
        MAIN[main.py<br/>Core Processing Logic]
        STREAMLIT[streamlit_app_v2.py<br/>Web Interface]
        RUN[run.py<br/>Startup Script]
        CREATE_INDEX[create_vector_index.py<br/>Vector Index Creation]
    end

    %% Core Functions
    subgraph "🔧 Core Functions"
        EXTRACT[extract_car_info_with_gemini<br/>AI Image Analysis]
        VECTOR_SETUP[setup_faiss_vector_search<br/>Vector Search Setup]
        SEARCH_DB[search_vector_database<br/>FAISS Search]
        MATCH_FORM[match_gemini_extraction_to_form<br/>Form Matching]
        GENERATE_JSON[generate_ikman_form_submission_json<br/>JSON Generation]
        PROCESS_END[process_car_image_end_to_end<br/>End-to-End Processing]
    end

    %% Data Processing
    subgraph "📊 Data Processing"
        NORMALIZE[normalize_text_for_search<br/>Text Normalization]
        ENCODE[encode_image_to_base64<br/>Image Encoding]
        PRELIMINARY[get_preliminary_query<br/>Quick Brand/Model Query]
        MERGE[merge_extracted_data<br/>Multi-Image Merging]
    end

    %% Configuration & Setup
    subgraph "⚙️ Configuration"
        LOAD_ENV[load_env_vars<br/>Environment Variables]
        SETUP_GEMINI[setup_gemini_client<br/>Gemini API Setup]
        CLEAR_CACHE[clear_cached_embeddings<br/>Cache Management]
        AUTO_SETUP[Auto Index Creation<br/>First Run Setup]
    end

    %% Connections
    MAIN --> EXTRACT
    MAIN --> VECTOR_SETUP
    MAIN --> SEARCH_DB
    MAIN --> MATCH_FORM
    MAIN --> GENERATE_JSON
    MAIN --> PROCESS_END
    MAIN --> NORMALIZE
    MAIN --> ENCODE
    MAIN --> PRELIMINARY
    MAIN --> MERGE
    MAIN --> LOAD_ENV
    MAIN --> SETUP_GEMINI
    MAIN --> CLEAR_CACHE
    MAIN --> AUTO_SETUP
    
    STREAMLIT --> MAIN
    RUN --> STREAMLIT
    CREATE_INDEX --> MAIN
```

## Data Flow Sequence

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant Main
    participant Gemini
    participant FAISS
    participant VectorDB
    participant Output

    User->>Streamlit: Upload Car Image
    Streamlit->>Main: process_car_image_end_to_end()
    
    Note over Main: Step 1: Preliminary Query
    Main->>Gemini: get_preliminary_query()
    Gemini-->>Main: Brand/Model Query
    
    Note over Main: Step 2: Vector Search
    Main->>FAISS: search_vector_database()
    FAISS->>VectorDB: Query Similar Entries
    VectorDB-->>FAISS: Search Results
    FAISS-->>Main: Vector Search Results
    
    Note over Main: Step 3: AI Extraction
    Main->>Gemini: extract_car_info_with_gemini()
    Gemini-->>Main: Extracted Car Data
    
    Note over Main: Step 4: Form Matching
    Main->>Main: match_gemini_extraction_to_form()
    Main->>Main: generate_ikman_form_submission_json()
    
    Main-->>Streamlit: Processing Results
    Streamlit-->>User: Analysis Complete
    
    Note over Output: Results Include:
    Note over Output: - AI Extracted Data
    Note over Output: - Form Autofill JSON
    Note over Output: - Confidence Scores
    Note over Output: - Manual Review Fields
```

## Vector Search Architecture

```mermaid
graph TB
    subgraph "🔍 Vector Search Pipeline"
        INPUT[Input Text<br/>Brand/Model Query]
        NORMALIZE[Text Normalization<br/>Unicode, Lowercase, Clean]
        TFIDF[TF-IDF Vectorization<br/>Text to Vector]
        FAISS_SEARCH[FAISS Index Search<br/>Similarity Matching]
        RESULTS[Search Results<br/>Top-K Matches]
    end

    subgraph "💾 Vector Database"
        VECTOR_ENTRIES[Vector Entries<br/>158,560+ Entries]
        BRAND_MODELS[Brand-Model Combinations<br/>1,100+ Models]
        FORM_MAPPINGS[Form Field Mappings<br/>11 Core Fields]
        METADATA[Search Metadata<br/>Type, Labels, Keys]
    end

    subgraph "🎯 Search Types"
        BRAND_SEARCH[Brand-Only Search<br/>Type: 'brand']
        MODEL_SEARCH[Model-Only Search<br/>Type: 'model']
        BRAND_MODEL_SEARCH[Brand-Model Search<br/>Type: 'brand_model']
    end

    INPUT --> NORMALIZE
    NORMALIZE --> TFIDF
    TFIDF --> FAISS_SEARCH
    FAISS_SEARCH --> VECTOR_ENTRIES
    FAISS_SEARCH --> BRAND_MODELS
    FAISS_SEARCH --> FORM_MAPPINGS
    FAISS_SEARCH --> METADATA
    FAISS_SEARCH --> RESULTS
    
    RESULTS --> BRAND_SEARCH
    RESULTS --> MODEL_SEARCH
    RESULTS --> BRAND_MODEL_SEARCH
```

## Form Field Mapping Architecture

```mermaid
graph TB
    subgraph "📝 Form Field Types"
        ENUM[Enum Fields<br/>condition, body, fuel_type, transmission]
        TREE[Tree Fields<br/>brand, model]
        YEAR[Year Fields<br/>model_year]
        MEASUREMENT[Measurement Fields<br/>mileage, engine_capacity]
        MONEY[Money Fields<br/>price]
        TEXT[Text Fields<br/>description, edition]
    end

    subgraph "🔧 Field Processing"
        FUZZY_MATCH[Fuzzy String Matching<br/>Threshold-based]
        EXACT_MATCH[Exact Matching<br/>Direct Comparison]
        RANGE_CHECK[Range Validation<br/>Min/Max Constraints]
        TEXT_GEN[Text Generation<br/>Smart Descriptions]
    end

    subgraph "📊 Field Statistics"
        AI_PREFILLED[AI Prefilled Fields<br/>High Confidence]
        MANUAL_REQUIRED[Manual Review Required<br/>Low Confidence]
        TOTAL_FIELDS[Total Form Fields<br/>11 Core Fields]
    end

    ENUM --> FUZZY_MATCH
    TREE --> EXACT_MATCH
    YEAR --> RANGE_CHECK
    MEASUREMENT --> RANGE_CHECK
    MONEY --> RANGE_CHECK
    TEXT --> TEXT_GEN
    
    FUZZY_MATCH --> AI_PREFILLED
    EXACT_MATCH --> AI_PREFILLED
    RANGE_CHECK --> AI_PREFILLED
    TEXT_GEN --> AI_PREFILLED
    
    FUZZY_MATCH --> MANUAL_REQUIRED
    EXACT_MATCH --> MANUAL_REQUIRED
    RANGE_CHECK --> MANUAL_REQUIRED
    TEXT_GEN --> MANUAL_REQUIRED
    
    AI_PREFILLED --> TOTAL_FIELDS
    MANUAL_REQUIRED --> TOTAL_FIELDS
```

## Cache Management Architecture

```mermaid
graph TB
    subgraph "💾 Cache Files"
        FAISS_INDEX[faiss_index.bin<br/>~440MB - FAISS Index]
        TFIDF_VECTORIZER[tfidf_vectorizer.pkl<br/>~1MB - TF-IDF Vectorizer]
        FAISS_METADATA[faiss_metadata.pkl<br/>~50MB - Search Metadata]
    end

    subgraph "🔄 Cache Operations"
        LOAD_CACHE[load_saved_faiss_data<br/>Load Cached Data]
        CREATE_CACHE[create_vector_index.py<br/>Generate Cache Files]
        CLEAR_CACHE[clear_cached_embeddings<br/>Remove Cache Files]
        SETUP_CACHE[setup_faiss_vector_search<br/>Cache Setup]
    end

    subgraph "⚡ Performance Benefits"
        FAST_STARTUP[16x Faster Startup<br/>Cached Embeddings]
        PARALLEL_PROC[Parallel Processing<br/>Multiple Images]
        BETTER_UX[Better User Experience<br/>Reduced Loading Time]
    end

    CREATE_CACHE --> FAISS_INDEX
    CREATE_CACHE --> TFIDF_VECTORIZER
    CREATE_CACHE --> FAISS_METADATA
    
    LOAD_CACHE --> FAISS_INDEX
    LOAD_CACHE --> TFIDF_VECTORIZER
    LOAD_CACHE --> FAISS_METADATA
    
    SETUP_CACHE --> LOAD_CACHE
    CLEAR_CACHE --> FAISS_INDEX
    CLEAR_CACHE --> TFIDF_VECTORIZER
    CLEAR_CACHE --> FAISS_METADATA
    
    FAISS_INDEX --> FAST_STARTUP
    TFIDF_VECTORIZER --> FAST_STARTUP
    FAISS_METADATA --> FAST_STARTUP
    
    FAST_STARTUP --> PARALLEL_PROC
    PARALLEL_PROC --> BETTER_UX
```

## Error Handling & Validation

```mermaid
graph TB
    subgraph "❌ Error Scenarios"
        API_ERROR[Gemini API Error<br/>Network/Timeout]
        VECTOR_ERROR[Vector Search Error<br/>Index Not Found]
        JSON_ERROR[JSON Parsing Error<br/>Invalid Response]
        FILE_ERROR[File Upload Error<br/>Invalid Format]
    end

    subgraph "✅ Validation Checks"
        PREREQ_CHECK[Prerequisites Check<br/>Files, Dependencies]
        API_KEY_CHECK[API Key Validation<br/>Gemini API Key]
        VECTOR_CHECK[Vector Index Check<br/>FAISS Index Exists]
        DEPENDENCY_CHECK[Dependency Check<br/>Required Packages]
    end

    subgraph "🔄 Recovery Actions"
        RETRY_MECHANISM[Retry Mechanism<br/>Max 3 Attempts]
        FALLBACK_MATCHING[Fallback Matching<br/>Direct String Matching]
        GRACEFUL_DEGRADATION[Graceful Degradation<br/>Partial Results]
        USER_NOTIFICATION[User Notification<br/>Clear Error Messages]
    end

    API_ERROR --> RETRY_MECHANISM
    VECTOR_ERROR --> FALLBACK_MATCHING
    JSON_ERROR --> GRACEFUL_DEGRADATION
    FILE_ERROR --> USER_NOTIFICATION
    
    PREREQ_CHECK --> API_KEY_CHECK
    API_KEY_CHECK --> VECTOR_CHECK
    VECTOR_CHECK --> DEPENDENCY_CHECK
    
    RETRY_MECHANISM --> USER_NOTIFICATION
    FALLBACK_MATCHING --> USER_NOTIFICATION
    GRACEFUL_DEGRADATION --> USER_NOTIFICATION
```

## Performance Metrics & Optimization

```mermaid
graph TB
    subgraph "📊 Performance Metrics"
        PROCESSING_TIME[Processing Time<br/>2-5 seconds per image]
        ACCURACY_RATE[Accuracy Rate<br/>85-95% for clear images]
        FIELD_MAPPING[Field Mapping Accuracy<br/>95%+ with fuzzy matching]
        SUCCESS_RATE[Success Rate<br/>Parallel processing]
    end

    subgraph "⚡ Optimizations"
        CACHE_EMBEDDINGS[Cached Embeddings<br/>16x faster startup]
        PARALLEL_PROCESSING[Parallel Processing<br/>Multiple images]
        VECTOR_SEARCH[Vector Search<br/>FAISS similarity]
        FUZZY_MATCHING[Fuzzy Matching<br/>Threshold-based]
    end

    subgraph "🎯 Quality Assurance"
        CONFIDENCE_SCORING[Confidence Scoring<br/>0-1 scale]
        MANUAL_REVIEW[Manual Review Flags<br/>Low confidence fields]
        VALIDATION_CHECKS[Validation Checks<br/>Range, format, type]
        ERROR_HANDLING[Error Handling<br/>Graceful degradation]
    end

    CACHE_EMBEDDINGS --> PROCESSING_TIME
    PARALLEL_PROCESSING --> PROCESSING_TIME
    VECTOR_SEARCH --> ACCURACY_RATE
    FUZZY_MATCHING --> FIELD_MAPPING
    
    CONFIDENCE_SCORING --> MANUAL_REVIEW
    MANUAL_REVIEW --> VALIDATION_CHECKS
    VALIDATION_CHECKS --> ERROR_HANDLING
    
    PROCESSING_TIME --> SUCCESS_RATE
    ACCURACY_RATE --> SUCCESS_RATE
    FIELD_MAPPING --> SUCCESS_RATE
```

## Technology Stack

```mermaid
graph TB
    subgraph "🤖 AI & ML"
        GEMINI_VISION[Gemini Vision AI<br/>Google AI]
        FAISS[FAISS<br/>Facebook AI Similarity Search]
        TFIDF[TF-IDF Vectorization<br/>scikit-learn]
        FUZZY[Fuzzy String Matching<br/>Custom Algorithm]
    end

    subgraph "🌐 Web Framework"
        STREAMLIT[Streamlit<br/>Web Interface]
        REQUESTS[Requests<br/>HTTP Client]
        PIL[Pillow<br/>Image Processing]
    end

    subgraph "💾 Data Storage"
        JSON[JSON Files<br/>Vector Dataset]
        PICKLE[Pickle Files<br/>Cached Objects]
        BINARY[Binary Files<br/>FAISS Index]
    end

    subgraph "🔧 Development"
        PYTHON[Python 3.8+<br/>Core Language]
        NUMPY[NumPy<br/>Numerical Computing]
        PANDAS[Pandas<br/>Data Manipulation]
    end

    GEMINI_VISION --> FAISS
    FAISS --> TFIDF
    TFIDF --> FUZZY
    
    STREAMLIT --> REQUESTS
    REQUESTS --> PIL
    
    JSON --> PICKLE
    PICKLE --> BINARY
    
    PYTHON --> NUMPY
    NUMPY --> PANDAS
```

This comprehensive architecture diagram shows the complete system design, data flow, and component interactions of the AI Car Autofill Service. The system combines advanced AI vision capabilities with efficient vector search and intelligent form field mapping to provide accurate car information extraction and form autofill functionality. 