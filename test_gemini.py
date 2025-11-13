"""
Quick test script to verify Google Gemini API
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key: {api_key[:10]}...{api_key[-5:]}")

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    
    print("\n🔄 Sending test request...")
    response = model.generate_content("Hello! Please respond with 'API is working!' in Vietnamese.")
    
    print("\n✅ Response:")
    print(response.text)
    print("\n✓ API Key is valid and working!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
