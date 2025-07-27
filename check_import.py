# This is the content for check_import.py
try:
    from my_api_server import app
    print("✅ SUCCESS: The 'app' object was imported correctly by Python.")
    print(f"   Object found is of type: {type(app)}")
except ImportError as e:
    print(f"❌ FAILED to find the file. Error: {e}")
except AttributeError as e:
    print(f"❌ FAILED to find the 'app' attribute inside the file. Error: {e}")