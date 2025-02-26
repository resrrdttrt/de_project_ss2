from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
from typing import List
import uvicorn

app = FastAPI(title="Sales Prediction API")

# Load the model from environment variable
model_path = os.getenv('MODEL_PATH', 'app/models/model_randomForest.pkl')

try:
    model = pickle.load(open(f'app/models/{model_path}', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define input data model
class Item(BaseModel):
    Item_Id: int
    Item_Visibility: float
    Store_Id: int
    Item_Fit_Type: int
    Item_Fabric: int
    Item_Fabric_Amount: float
    Item_MRP: float
    Store_Establishment_Year: int
    Store_Size: int
    Store_Location_Type: int
    Store_Type: int

class Items(BaseModel):
    items: List[Item]

# Convert input data to DataFrame
def create_dataframe(items_data: List[Item]) -> pd.DataFrame:
    return pd.DataFrame([item.dict() for item in items_data])

@app.post("/predict")
async def predict(items: Items):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    try:
        # Convert input data to DataFrame
        input_df = create_dataframe(items.items)
        
        # Make predictions
        predictions = model.predict(input_df)
        
        # Create results DataFrame
        predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
        result = input_df.copy()
        result['Sales_Prediction'] = predictions_df['Prediction'].values
        result['Required_Fabric_Prediction'] = predictions_df['Prediction'].values * input_df['Item_Fabric_Amount']
        
        # Convert results to JSON
        results = []
        for idx, row in result.iterrows():
            results.append({
                'Item_Id': row['Item_Id'],
                'Sales_Prediction': float(row['Sales_Prediction']),
                'Required_Fabric_Prediction': float(row['Required_Fabric_Prediction']),
                'Item_Details': {
                    'Item_Visibility': row['Item_Visibility'],
                    'Store_Id': row['Store_Id'],
                    'Item_Fit_Type': row['Item_Fit_Type'],
                    'Item_Fabric': row['Item_Fabric'],
                    'Item_Fabric_Amount': row['Item_Fabric_Amount'],
                    'Item_MRP': row['Item_MRP'],
                    'Store_Establishment_Year': row['Store_Establishment_Year'],
                    'Store_Size': row['Store_Size'],
                    'Store_Location_Type': row['Store_Location_Type'],
                    'Store_Type': row['Store_Type']
                }
            })
        
        return {"status": "success", "predictions": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)