from app.database.crud.inference_input import inference_crud
from app.database.crud.prediction import prediction_crud
from app.services.utils.inference_input import prepare_input_for_prediction

from app.database.crud.processed_feature import processed_crud
from app.services.utils.processed_feature import prepare_dataframe

import pandas as pd
import numpy as np

def get_reference_data() -> pd.DataFrame:
    features = processed_crud.get_all()
    
    trained_df = prepare_dataframe(features)
    
    trained_df = trained_df.drop(columns=['id'], errors='ignore')

    conditions = [
        trained_df['charger_type_Level_1'] == True,
        trained_df['charger_type_Level_2'] == True
    ]
    choices = ['Level_1', 'Level_2']
    trained_df['target'] = np.select(conditions, choices, default='DC_Fast_Charger')
    
    trained_df = trained_df.drop(
        columns=['charger_type_Level_1', 'charger_type_Level_2'], 
        errors='ignore'
    )
    
    return trained_df

def get_current_data(trained_df: pd.DataFrame) -> pd.DataFrame:
    inferences = inference_crud.get_all()
    predictions = prediction_crud.get_all(source='inference')

    processed_dfs = []
    for inference in inferences:
        df = prepare_input_for_prediction(inference, trained_df)
        df['inference_input_id'] = inference.id
        df['timestamp'] = inference.timestamp
        processed_dfs.append(df)
    
    combined_df = pd.concat(processed_dfs, ignore_index=True) if processed_dfs else pd.DataFrame()
    
    predictions_df = pd.DataFrame([
        {
            'inference_input_id': pred.inference_input_id,
            'predicted_class': pred.predicted_class,
            'confidence': pred.confidence,
            'actual_class': pred.actual_class
        }
        for pred in predictions
    ])
    
    combined_df = combined_df.dropna(subset=['inference_input_id'])
    predictions_df = predictions_df.dropna(subset=['inference_input_id'])
    
    combined_df['inference_input_id'] = combined_df['inference_input_id'].astype(int)
    predictions_df['inference_input_id'] = predictions_df['inference_input_id'].astype(int)
    
    combined_df = combined_df.merge(
        predictions_df, 
        on='inference_input_id', 
        how='left'
    )
    
    combined_df['target'] = combined_df['predicted_class'].fillna('DC_Fast_Charger')

    columns_to_drop = ['inference_input_id', 'predicted_class', 'confidence', 'actual_class']
    combined_df = combined_df.drop(columns=columns_to_drop, errors='ignore')

    return combined_df
