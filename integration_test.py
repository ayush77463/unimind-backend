from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("=== 1. Testing Health Endpoint ===")
health_response = client.get("/health")
print("Status Code:", health_response.status_code)
print("Response JSON:", health_response.json())
assert health_response.status_code == 200
assert health_response.json().get("storage") == "supabase_postgresql"

print("\n=== 2. Testing Document Extraction Endpoint ===")
# Create a dummy text file buffer in memory
file_content = b"This is a dummy text document created to test the UniMind server-side extraction."
files = {"file": ("dummy.txt", file_content, "text/plain")}
data = {"user_id": "test_user"}

upload_response = client.post("/document/upload", data=data, files=files)
print("Status Code:", upload_response.status_code)
print("Response JSON:", upload_response.json())
assert upload_response.status_code == 200
assert upload_response.json().get("success") is True
assert "dummy text document" in upload_response.json().get("extracted_text")

print("\nAll integration tests passed perfectly!")
