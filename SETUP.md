# 🚗 AI Car Autofill Service - Setup Guide

This guide will help you set up and run the AI-powered car brand/model detection and form autofill service.

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A Gemini AI API key (free tier available)

## 🚀 Quick Setup

### 1. Clone or Download the Project

```bash
# If you have the files locally, navigate to the project directory
cd ai-prefill-service
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Get Your Gemini API Key

1. Visit [Google AI Studio](https://ai.google.dev/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

**Important:** Replace `your_actual_api_key_here` with your actual Gemini API key.

### 5. Test the Setup

```bash
# Test vector search functionality (no API key required)
python test_vector_search.py

# Test complete system (requires API key)
python main.py
```

### 6. Launch the Web Interface

```bash
# Start the Streamlit web app
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501` 🎉

## 🔧 Detailed Setup Instructions

### Environment Setup

#### Option A: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Global Installation

```bash
# Install dependencies globally
pip install -r requirements.txt
```

### API Key Configuration

1. **Get API Key:**
   - Go to [Google AI Studio](https://ai.google.dev/)
   - Sign in with your Google account
   - Navigate to "Get API key"
   - Create a new API key
   - Copy the key

2. **Configure Environment:**
   ```bash
   # Create .env file
   cat > .env << EOF
   GEMINI_API_KEY=your_actual_api_key_here
   EOF
   ```

3. **Verify Configuration:**
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', os.getenv('GEMINI_API_KEY')[:10] + '...' if os.getenv('GEMINI_API_KEY') else 'Not found')"
   ```

### Testing the Installation

#### 1. Test Vector Search (No API Key Required)

```bash
python test_vector_search.py
```

Expected output:
```
🎉 All tests passed! The system is ready for AI integration.
```

#### 2. Test Complete System (Requires API Key)

```bash
python main.py
```

Expected output:
```
✅ Gemini API key configured
✅ FAISS index created with X vectors
✅ Vector search test successful
🎉 All tests passed! The system is ready to use.
```

#### 3. Test Web Interface

```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` and test the interface.

## 📁 Project Structure

```
ai-prefill-service/
├── main.py                           # Core processing logic
├── streamlit_app.py                  # Web interface
├── test_vector_search.py             # Vector search testing
├── car_vector_dataset.json          # Vector database (15MB)
├── requirements.txt                  # Python dependencies
├── env_example.txt                   # Environment variables example
├── SETUP.md                          # This setup guide
└── README.md                         # Project documentation
```

## 🧪 Testing with Sample Images

### Using the Web Interface

1. Start the Streamlit app: `streamlit run streamlit_app.py`
2. Upload car images (JPG, PNG, WEBP formats)
3. View AI analysis results
4. Download generated ikman.lk form JSON

### Using Python Code

```python
from main import process_car_image_end_to_end
import json

# Load vector dataset
with open('car_vector_dataset.json', 'r') as f:
    vector_dataset = json.load(f)

# Process a car image
result = process_car_image_end_to_end('path/to/car_image.jpg', vector_dataset)

# Get ikman.lk form submission JSON
if 'ikman_form_submission' in result:
    form_json = result['ikman_form_submission']
    print(json.dumps(form_json, indent=2))
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "GEMINI_API_KEY not found"

**Solution:** Create a `.env` file with your API key:
```bash
echo "GEMINI_API_KEY=your_key_here" > .env
```

#### 2. "Vector dataset not found"

**Solution:** Ensure `car_vector_dataset.json` is in the project root directory.

#### 3. Import Errors

**Solution:** Install missing dependencies:
```bash
pip install -r requirements.txt
```

#### 4. FAISS Installation Issues

**Solution:** Try installing FAISS with conda:
```bash
conda install -c conda-forge faiss-cpu
```

#### 5. Streamlit Port Issues

**Solution:** Use a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### API Quota Issues

The free Gemini API has daily limits:
- **Free Tier:** 50 requests/day
- **Paid Tier:** 1000+ requests/day

**Solutions:**
1. Wait for daily reset (midnight Pacific Time)
2. Upgrade to paid plan
3. Use the system efficiently (batch process images)

### Performance Optimization

1. **Use Multiple Images:** Upload 2-4 images of the same car for better accuracy
2. **Clear Images:** Use well-lit, clear photos
3. **Batch Processing:** Process multiple cars at once to optimize API usage

## 📊 System Requirements

### Minimum Requirements
- **RAM:** 4GB
- **Storage:** 1GB free space
- **CPU:** 2 cores
- **Internet:** Required for API calls

### Recommended Requirements
- **RAM:** 8GB+
- **Storage:** 2GB+ free space
- **CPU:** 4+ cores
- **Internet:** Stable connection

## 🔐 Security Notes

1. **API Key Security:** Never commit your `.env` file to version control
2. **Image Privacy:** Images are processed by Google's servers
3. **Data Storage:** No images are stored locally after processing

## 📞 Support

If you encounter issues:

1. **Check the logs:** Look for error messages in the terminal
2. **Verify setup:** Run `python test_vector_search.py`
3. **Check API key:** Ensure your Gemini API key is valid
4. **Update dependencies:** Run `pip install -r requirements.txt --upgrade`

## 🎯 Next Steps

After successful setup:

1. **Test with your own images:** Upload car photos to test accuracy
2. **Customize form fields:** Modify `car_vector_dataset.json` if needed
3. **Integrate with your system:** Use the generated JSON in your applications
4. **Monitor API usage:** Keep track of your daily API quota

---

**🚗 Ready to go!** Your AI Car Autofill Service is now set up and ready to use. 