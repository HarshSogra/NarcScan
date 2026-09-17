from fastapi import FastAPI

# Create the FastAPI application instance.
# The title and version show up in the auto-generated docs at /docs.
app = FastAPI(title="NarcScan Backend", version="0.1.0")


# This handles GET requests to the root URL: http://127.0.0.1:8000/
@app.get("/")
def root():
    return {"message": "NarcScan Backend is running."}


# This handles GET requests to: http://127.0.0.1:8000/health
# It is used to quickly check whether the server is alive and responding.
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "NarcScan Backend",
    }
