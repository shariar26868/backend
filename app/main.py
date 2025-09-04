from fastapi import FastAPI
from app.routes import churn, ltv, fraud, segmentation, engagement
from app.utils.logger import setup_logger

app = FastAPI(
    title="AI-Powered CRM Backend",
    description="FastAPI backend for CRM with ML predictions using Canada777 API",
    version="1.0.0"
)

# Setup logging
logger = setup_logger()

# Include routers
app.include_router(churn.router, prefix="/churn", tags=["Churn Prediction"])
app.include_router(ltv.router, prefix="/ltv", tags=["LTV Prediction"])
app.include_router(fraud.router, prefix="/fraud", tags=["Fraud Detection"])
app.include_router(segmentation.router, prefix="/segmentation", tags=["Segmentation"])
app.include_router(engagement.router, prefix="/engagement", tags=["Engagement Prediction"])

@app.get("/")
async def root():
    return {"message": "Welcome to the AI-Powered CRM Backend"}