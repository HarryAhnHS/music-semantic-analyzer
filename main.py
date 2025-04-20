from fastapi import FastAPI
from routes.semantic import router as semantic_router

app = FastAPI()

# Register the route
app.include_router(semantic_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)