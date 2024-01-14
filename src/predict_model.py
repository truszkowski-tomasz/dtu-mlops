import torch
from torch.utils.data import DataLoader
import pandas as pd 
from train_model import * 

def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, return_probabilities: bool = True) -> torch.Tensor:
    """
    Run prediction for a given model and dataloader.

    Args:
        model: The model to use for prediction.
        dataloader: Dataloader containing the data for prediction.
        device: The device (CPU or GPU) where the model is loaded.
        return_probabilities: If True, returns class probabilities, else returns class labels.

    Returns:
        torch.Tensor: A tensor of predictions. If return_probabilities is True, it's of shape [N, C] where N is the number of samples and C is the number of classes. Otherwise, it's of shape [N] containing class labels.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():  # No gradient calculation
        for batch in dataloader:
            # Adapting to different batch formats, assuming ids, mask, token_type_ids, or single input format
            if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                ids, mask, token_type_ids = [b.to(device) for b in batch[:3]]
                outputs = model(ids, mask, token_type_ids)
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(device)
                outputs = model(batch)
            else:
                raise ValueError("Unsupported batch format for prediction")

            if return_probabilities:
                # Assuming the model outputs logits; apply softmax for probabilities
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predictions.append(probabilities)
            else:
                # Assuming the model outputs logits; taking the argmax for class labels
                labels = torch.argmax(outputs, dim=1)
                predictions.append(labels)

    return torch.cat(predictions, dim=0)

train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)


predictions = predict(model, train_loader, device, return_probabilities=True)
prediction_file_path = './src/Predictions.csv'
def predict_to_file(model, dataloader, device, file_path):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch.to(device)
            outputs = model(inputs)
            # Convert outputs to probabilities using softmax
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # Convert probabilities to numpy array
            probabilities = probabilities.cpu().numpy()
            predictions.append(probabilities)

    # Concatenate all batches
    predictions = np.concatenate(predictions, axis=0)

    # Create a dataframe and save to CSV
    df = pd.DataFrame(predictions, columns=[f'Class_{i}' for i in range(predictions.shape[1])])
    df.to_csv(file_path, index=False)
    # Path where you want to save the prediction file
   

    # Check if the file exists
    if not os.path.exists(prediction_file_path):
        # Create and save the DataFrame to a new CSV file
        df.to_csv(prediction_file_path, index=False)
        print(f"File '{prediction_file_path}' created and data saved.")
    else:
        # Append data to the existing file without including the header
        df.to_csv(prediction_file_path, mode='a', header=False, index=False)
        print(f"Data appended to existing file '{prediction_file_path}'.")


    

# Call the function to make predictions and save to a file
predict_to_file(model, train_loader, device, prediction_file_path)