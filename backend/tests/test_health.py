from fastapi.testclient import TestClient
from app.main import app

# TestClient lets us send fake HTTP requests to the app without starting a real server.
client = TestClient(app)


def test_health_check():
    # Send a GET request to /health
    response = client.get("/health")

    # Check that the HTTP status code is 200 (means "OK, everything worked")
    assert response.status_code == 200

    # Check that the response body matches exactly what we expect
    assert response.json() == {
        "status": "ok",
        "service": "NarcScan Backend",
    }
