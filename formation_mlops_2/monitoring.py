import pandas as pd


def monitor(latest_predictions: pd.DataFrame) -> pd.DataFrame:
    # Start filling function
    monitoring_df = latest_predictions.groupby('predictions_time')['predictions'].mean().reset_index()
 
    # End filling function
    return monitoring_df
