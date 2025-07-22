# 🚀 Quick Start Guide

Get the AI Car Autofill Service running in 5 minutes!

## ⚡ Super Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get API Key
- Go to [Google AI Studio](https://ai.google.dev/)
- Create a free API key
- Create `.env` file: `echo "GEMINI_API_KEY=your_key_here" > .env`

### 3. Test Setup
```bash
# Test vector search (no API key needed)
python test_vector_search.py

# Test complete system (needs API key)
python main.py
```

### 4. Launch Web App
```bash
streamlit run streamlit_app.py
```
Open: http://localhost:8501

## 🎯 What You Get

✅ **AI Car Detection** - Identifies brand, model, year from images  
✅ **Smart Form Filling** - Generates ikman.lk compatible JSON  
✅ **Vector Search** - 19,690+ car models in database  
✅ **Confidence Scoring** - Knows when to ask for manual input  
✅ **Web Interface** - Beautiful Streamlit UI  
✅ **API Ready** - Use programmatically in your apps  

## 📸 How to Use

### Web Interface
1. Upload car images (multiple angles work best)
2. View AI analysis results
3. Download generated form JSON
4. Use JSON for ikman.lk submission

### Programmatic Usage
```python
from main import process_car_image_end_to_end
import json

# Load dataset
with open('car_vector_dataset.json', 'r') as f:
    dataset = json.load(f)

# Process image
result = process_car_image_end_to_end('car.jpg', dataset)

# Get form JSON
form_json = result['ikman_form_submission']
```

## 🔧 Troubleshooting

**"API key not found"** → Create `.env` file with your key  
**"Vector dataset not found"** → Ensure `car_vector_dataset.json` exists  
**Import errors** → Run `pip install -r requirements.txt`  
**Port issues** → Use `--server.port 8502` with streamlit  

## 📊 System Status

- ✅ Vector Database: 19,690 entries
- ✅ Form Fields: 11 fields supported
- ✅ AI Models: Gemini Vision + FAISS
- ✅ Web Interface: Streamlit ready
- ✅ API Integration: ikman.lk compatible

## 🎉 You're Ready!

The system is now ready to:
- Detect cars from images
- Generate form data automatically
- Handle confidence-based field filling
- Export ikman.lk compatible JSON

**Next:** Upload some car images and see the magic! 🚗✨ 