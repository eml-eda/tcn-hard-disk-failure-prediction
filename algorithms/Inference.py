import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

# Define the dataset class for the inference
class InferenceDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

# Function to collate batch for inference
def FPLSTM_collate(batch):
    x_batch = torch.stack(batch).permute(1, 0, 2)
    return x_batch

def load_model(model, model_path):
    """
    Load the saved LSTM model from the specified path.

    Args:
        model (torch.nn.Module): The LSTM model class.
        model_path (str): The path to the saved model file.

    Returns:
        torch.nn.Module: The loaded LSTM model.
    """
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def infer(model, X):
    """
    Use the trained model to make predictions on new data.

    Args:
        model (torch.nn.Module): The trained LSTM model.
        X (np.ndarray): The input data for inference.

    Returns:
        np.ndarray: The predicted labels.
    """
    inference_loader = DataLoader(InferenceDataset(X), batch_size=1, shuffle=False, collate_fn=FPLSTM_collate)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in inference_loader:
            sequences = batch.cuda()
            output = model(sequences)
            predicted_labels = output.argmax(dim=1).cpu().numpy()
            predictions.extend(predicted_labels)
    
    return np.array(predictions)

# Example usage:
if __name__ == "__main__":
    # Define your model architecture (should match the trained model)
    class YourLSTMModel(torch.nn.Module):
        # Define your model here
        pass
    
    # Path to the saved model
    model_path = 'path_to_your_saved_model.pth'

    # Load the model
    model = YourLSTMModel().cuda()
    model = load_model(model, model_path)

    # Load or prepare your input data for inference
    X_inference = np.array(...)  # Replace with your actual input data

    # Make predictions
    predictions = infer(model, X_inference)

    # Print or use the predictions as needed
    print("Predictions:", predictions)