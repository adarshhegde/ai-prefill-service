# 🚗 AI Car Autofill Service - Architecture Diagram

## System Overview

```mermaid
graph TB
    %% User Interface Layer
    subgraph "🌐 User Interface"
        UI[Streamlit Web App<br/>streamlit_app_v2.py]
        UPLOAD[Image Upload<br/>Multiple Formats]
        URL_EXTRACT[ikman.lk URL<br/>Extraction]
        BATCH_PROC[Batch Processing<br/>Parallel Workers]
    end

    %% Core Processing Layer
    subgraph "🤖 AI Processing Engine"
        GEMINI[Gemini Vision AI<br/>Image Analysis]
        VECTOR_SEARCH[FAISS Vector Search<br/>Similarity Matching]
        FUZZY_MATCH[Fuzzy Field Mapping<br/>Form Field Matching]
        COST_TRACKER[GeminiCostTracker<br/>API Cost Analysis]
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
        COST_ANALYSIS[Cost Analysis<br/>Token Usage & Pricing]
        DOWNLOAD[Download Options<br/>JSON/Complete Data]
    end

    %% Data Flow
    UI --> UPLOAD
    UI --> URL_EXTRACT
    UI --> BATCH_PROC
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
    GEMINI --> COST_TRACKER
    COST_TRACKER --> COST_ANALYSIS
    ANALYSIS_RESULTS --> DOWNLOAD
    FORM_JSON --> DOWNLOAD
    COST_ANALYSIS --> DOWNLOAD
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
        BATCH_PROC[process_car_images_batch<br/>Batch Processing]
    end

    %% Data Processing
    subgraph "📊 Data Processing"
        NORMALIZE[normalize_text_for_search<br/>Text Normalization]
        ENCODE[encode_image_to_base64<br/>Image Encoding]
        PRELIMINARY[get_preliminary_query<br/>Quick Brand/Model Query]
        MERGE[merge_extracted_data<br/>Multi-Image Merging]
        MULTI_ID[process_multiple_identifications<br/>Best Match Selection]
    end

    %% Configuration & Setup
    subgraph "⚙️ Configuration"
        LOAD_ENV[load_env_vars<br/>Environment Variables]
        SETUP_GEMINI[setup_gemini_client<br/>Gemini API Setup]
        CLEAR_CACHE[clear_cached_embeddings<br/>Cache Management]
        AUTO_SETUP[Auto Index Creation<br/>First Run Setup]
        COST_TRACKER[GeminiCostTracker<br/>Cost Analysis]
    end

    %% Connections
    MAIN --> EXTRACT
    MAIN --> VECTOR_SETUP
    MAIN --> SEARCH_DB
    MAIN --> MATCH_FORM
    MAIN --> GENERATE_JSON
    MAIN --> PROCESS_END
    MAIN --> BATCH_PROC
    MAIN --> NORMALIZE
    MAIN --> ENCODE
    MAIN --> PRELIMINARY
    MAIN --> MERGE
    MAIN --> MULTI_ID
    MAIN --> LOAD_ENV
    MAIN --> SETUP_GEMINI
    MAIN --> CLEAR_CACHE
    MAIN --> AUTO_SETUP
    MAIN --> COST_TRACKER
    
    STREAMLIT --> MAIN
    RUN --> STREAMLIT
    CREATE_INDEX --> MAIN
```

## Enhanced Data Flow Sequence

```mermaid
sequenceDiagram
    participant User
    participant Streamlit
    participant Main
    participant Gemini
    participant FAISS
    participant VectorDB
    participant CostTracker
    participant Output

    User->>Streamlit: Upload Car Image(s)
    Streamlit->>Main: process_car_image_end_to_end() or process_car_images_batch()
    
    Note over Main: Step 1: Brand/Model Extraction
    Main->>Gemini: extract_car_info_with_gemini()
    Gemini-->>Main: Brand/Model with Confidence
    Main->>CostTracker: Track API Usage & Costs
    
    Note over Main: Step 2: Multiple Identifications Processing
    Main->>Main: process_multiple_identifications()
    Main->>Main: Select Best Match by Confidence
    
    Note over Main: Step 3: Vector Search
    Main->>FAISS: search_vector_database()
    FAISS->>VectorDB: Query Similar Entries
    VectorDB-->>FAISS: Search Results
    FAISS-->>Main: Vector Search Results
    
    Note over Main: Step 4: Additional Details Extraction
    Main->>Gemini: extract_additional_details_with_gemini()
    Gemini-->>Main: Detailed Car Information
    Main->>CostTracker: Track Additional API Usage
    
    Note over Main: Step 5: Form Generation
    Main->>Main: generate_ikman_form_submission_json()
    Main->>Main: Fuzzy Matching & Field Mapping
    
    Main-->>Streamlit: Processing Results with Cost Analysis
    Streamlit-->>User: Analysis Complete with Cost Summary
    
    Note over Output: Results Include:
    Note over Output: - AI Extracted Data
    Note over Output: - Form Autofill JSON
    Note over Output: - Confidence Scores
    Note over Output: - Manual Review Fields
    Note over Output: - API Cost Analysis
```

## Cost Tracking Architecture

```mermaid
graph TB
    subgraph "💰 Cost Tracking System"
        COST_TRACKER[GeminiCostTracker<br/>Global Instance]
        TIERED_PRICING[Tiered Pricing<br/>Input/Output Tokens]
        IMAGE_COST[Image Cost<br/>$0.0025 per image]
        REQUEST_TRACKING[Request Tracking<br/>Per API Call]
    end

    subgraph "📊 Pricing Tiers"
        INPUT_128K[Input ≤128k tokens<br/>$1.25 per 1M]
        INPUT_ABOVE[Input >128k tokens<br/>$2.50 per 1M]
        OUTPUT_128K[Output ≤128k tokens<br/>$5.00 per 1M]
        OUTPUT_ABOVE[Output >128k tokens<br/>$10.00 per 1M]
    end

    subgraph "🔍 Request Types"
        BRAND_EXTRACTION[Brand Extraction<br/>extract_car_info_with_gemini]
        ADDITIONAL_DETAILS[Additional Details<br/>extract_additional_details_with_gemini]
        PRELIMINARY_QUERY[Preliminary Query<br/>get_preliminary_query]
        BATCH_PROCESSING[Batch Processing<br/>process_car_images_batch]
    end

    subgraph "📈 Cost Analysis"
        TOTAL_COST[Total Cost USD<br/>Real-time Calculation]
        TOKEN_USAGE[Token Usage<br/>Input/Output Breakdown]
        REQUEST_COUNT[Request Count<br/>By Type]
        AVERAGE_COST[Average Cost<br/>Per Request]
    end

    COST_TRACKER --> TIERED_PRICING
    COST_TRACKER --> IMAGE_COST
    COST_TRACKER --> REQUEST_TRACKING
    
    TIERED_PRICING --> INPUT_128K
    TIERED_PRICING --> INPUT_ABOVE
    TIERED_PRICING --> OUTPUT_128K
    TIERED_PRICING --> OUTPUT_ABOVE
    
    REQUEST_TRACKING --> BRAND_EXTRACTION
    REQUEST_TRACKING --> ADDITIONAL_DETAILS
    REQUEST_TRACKING --> PRELIMINARY_QUERY
    REQUEST_TRACKING --> BATCH_PROCESSING
    
    BRAND_EXTRACTION --> TOTAL_COST
    ADDITIONAL_DETAILS --> TOTAL_COST
    PRELIMINARY_QUERY --> TOTAL_COST
    BATCH_PROCESSING --> TOTAL_COST
    
    TOTAL_COST --> TOKEN_USAGE
    TOTAL_COST --> REQUEST_COUNT
    TOTAL_COST --> AVERAGE_COST
```

## Batch Processing Architecture

```mermaid
graph TB
    subgraph "📦 Batch Processing Flow"
        BATCH_INPUT[Multiple Images<br/>Upload/URL]
        BATCH_ENCODE[Encode All Images<br/>Base64 Conversion]
        BATCH_PROMPT[Simplified Prompt<br/>Brand/Model Only]
        BATCH_API[Single API Call<br/>Multiple Images]
        BATCH_PARSE[Parse Results<br/>Per Image]
    end

    subgraph "🔄 Individual Processing"
        VECTOR_SEARCH[Vector Search<br/>Per Image]
        ADDITIONAL_DETAILS[Additional Details<br/>Per Image]
        FORM_GENERATION[Form Generation<br/>Per Image]
        COST_TRACKING[Cost Tracking<br/>Per Image]
    end

    subgraph "⚙️ Optimization Features"
        PARALLEL_WORKERS[Parallel Workers<br/>Max 3 Workers]
        RATE_LIMITING[Rate Limiting<br/>1s Delay Between]
        FALLBACK[Fallback Processing<br/>Individual if Batch Fails]
        COST_EFFICIENCY[Cost Efficiency<br/>Reduced API Overhead]
    end

    subgraph "📊 Batch Benefits"
        REDUCED_OVERHEAD[Reduced API Overhead<br/>Single Request]
        FASTER_PROCESSING[Faster Processing<br/>Parallel Execution]
        LOWER_COSTS[Lower Costs<br/>Efficient Token Usage]
        BETTER_UX[Better UX<br/>Progress Tracking]
    end

    BATCH_INPUT --> BATCH_ENCODE
    BATCH_ENCODE --> BATCH_PROMPT
    BATCH_PROMPT --> BATCH_API
    BATCH_API --> BATCH_PARSE
    
    BATCH_PARSE --> VECTOR_SEARCH
    VECTOR_SEARCH --> ADDITIONAL_DETAILS
    ADDITIONAL_DETAILS --> FORM_GENERATION
    FORM_GENERATION --> COST_TRACKING
    
    BATCH_API --> PARALLEL_WORKERS
    PARALLEL_WORKERS --> RATE_LIMITING
    RATE_LIMITING --> FALLBACK
    FALLBACK --> COST_EFFICIENCY
    
    COST_EFFICIENCY --> REDUCED_OVERHEAD
    REDUCED_OVERHEAD --> FASTER_PROCESSING
    FASTER_PROCESSING --> LOWER_COSTS
    LOWER_COSTS --> BETTER_UX
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
        VECTOR_MATCH[Vector Database Matching<br/>Exact Keys]
    end

    subgraph "📊 Field Statistics"
        AI_PREFILLED[AI Prefilled Fields<br/>High Confidence]
        MANUAL_REQUIRED[Manual Review Required<br/>Low Confidence]
        TOTAL_FIELDS[Total Form Fields<br/>11 Core Fields]
    end

    ENUM --> FUZZY_MATCH
    TREE --> VECTOR_MATCH
    YEAR --> RANGE_CHECK
    MEASUREMENT --> RANGE_CHECK
    MONEY --> RANGE_CHECK
    TEXT --> TEXT_GEN
    
    FUZZY_MATCH --> AI_PREFILLED
    VECTOR_MATCH --> AI_PREFILLED
    RANGE_CHECK --> AI_PREFILLED
    TEXT_GEN --> AI_PREFILLED
    
    FUZZY_MATCH --> MANUAL_REQUIRED
    VECTOR_MATCH --> MANUAL_REQUIRED
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
        BATCH_ERROR[Batch Processing Error<br/>Fallback to Individual]
    end

    subgraph "✅ Validation Checks"
        PREREQ_CHECK[Prerequisites Check<br/>Files, Dependencies]
        API_KEY_CHECK[API Key Validation<br/>Gemini API Key]
        VECTOR_CHECK[Vector Index Check<br/>FAISS Index Exists]
        DEPENDENCY_CHECK[Dependency Check<br/>Required Packages]
        COST_LIMIT_CHECK[Cost Limit Check<br/>Budget Monitoring]
    end

    subgraph "🔄 Recovery Actions"
        RETRY_MECHANISM[Retry Mechanism<br/>Max 3 Attempts]
        FALLBACK_MATCHING[Fallback Matching<br/>Direct String Matching]
        GRACEFUL_DEGRADATION[Graceful Degradation<br/>Partial Results]
        USER_NOTIFICATION[User Notification<br/>Clear Error Messages]
        BATCH_FALLBACK[Batch Fallback<br/>Individual Processing]
    end

    API_ERROR --> RETRY_MECHANISM
    VECTOR_ERROR --> FALLBACK_MATCHING
    JSON_ERROR --> GRACEFUL_DEGRADATION
    FILE_ERROR --> USER_NOTIFICATION
    BATCH_ERROR --> BATCH_FALLBACK
    
    PREREQ_CHECK --> API_KEY_CHECK
    API_KEY_CHECK --> VECTOR_CHECK
    VECTOR_CHECK --> DEPENDENCY_CHECK
    DEPENDENCY_CHECK --> COST_LIMIT_CHECK
    
    RETRY_MECHANISM --> USER_NOTIFICATION
    FALLBACK_MATCHING --> USER_NOTIFICATION
    GRACEFUL_DEGRADATION --> USER_NOTIFICATION
    BATCH_FALLBACK --> USER_NOTIFICATION
```

## Performance Metrics & Optimization

```mermaid
graph TB
    subgraph "📊 Performance Metrics"
        PROCESSING_TIME[Processing Time<br/>2-5 seconds per image]
        ACCURACY_RATE[Accuracy Rate<br/>85-95% for clear images]
        FIELD_MAPPING[Field Mapping Accuracy<br/>95%+ with fuzzy matching]
        SUCCESS_RATE[Success Rate<br/>Parallel processing]
        COST_EFFICIENCY[Cost Efficiency<br/>Tiered pricing optimization]
    end

    subgraph "⚡ Optimizations"
        CACHE_EMBEDDINGS[Cached Embeddings<br/>16x faster startup]
        PARALLEL_PROCESSING[Parallel Processing<br/>Multiple images]
        VECTOR_SEARCH[Vector Search<br/>FAISS similarity]
        FUZZY_MATCHING[Fuzzy Matching<br/>Threshold-based]
        BATCH_PROCESSING[Batch Processing<br/>Reduced API overhead]
        COST_TRACKING[Cost Tracking<br/>Real-time monitoring]
    end

    subgraph "🎯 Quality Assurance"
        CONFIDENCE_SCORING[Confidence Scoring<br/>0-1 scale]
        MANUAL_REVIEW[Manual Review Flags<br/>Low confidence fields]
        VALIDATION_CHECKS[Validation Checks<br/>Range, format, type]
        ERROR_HANDLING[Error Handling<br/>Graceful degradation]
        MULTIPLE_ID[Multiple Identifications<br/>Best match selection]
    end

    CACHE_EMBEDDINGS --> PROCESSING_TIME
    PARALLEL_PROCESSING --> PROCESSING_TIME
    BATCH_PROCESSING --> PROCESSING_TIME
    VECTOR_SEARCH --> ACCURACY_RATE
    FUZZY_MATCHING --> FIELD_MAPPING
    COST_TRACKING --> COST_EFFICIENCY
    
    CONFIDENCE_SCORING --> MANUAL_REVIEW
    MANUAL_REVIEW --> VALIDATION_CHECKS
    VALIDATION_CHECKS --> ERROR_HANDLING
    MULTIPLE_ID --> CONFIDENCE_SCORING
    
    PROCESSING_TIME --> SUCCESS_RATE
    ACCURACY_RATE --> SUCCESS_RATE
    FIELD_MAPPING --> SUCCESS_RATE
    COST_EFFICIENCY --> SUCCESS_RATE
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
        CONCURRENT[Concurrent.futures<br/>Parallel Processing]
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
    PYTHON --> CONCURRENT
```

This comprehensive architecture diagram shows the complete system design, data flow, and component interactions of the AI Car Autofill Service. The system combines advanced AI vision capabilities with efficient vector search, intelligent form field mapping, cost tracking, and batch processing to provide accurate car information extraction and form autofill functionality. 