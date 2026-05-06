import uvicorn

from src.api.app import create_app

# Create the app at module level so uvicorn can import it with
# "main:app" in both dev (--reload) and production.
app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,   # hot-reload on file save — dev only, remove in production
    )
