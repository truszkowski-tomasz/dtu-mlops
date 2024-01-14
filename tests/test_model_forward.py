import torch
from omegaconf import OmegaConf

# from src.models.model import BERTLightning


# def test_forward():
#     # Create a sample configuration for the model with a Hugging Face model hub URL
#     model_config = OmegaConf.create({"model": {"file_path_input": "models/bert-base-uncased"}})

#     # Instantiate the BERTLightning model
#     model = BERTLightning(config=model_config)

#     # Create dummy input tensors
#     ids = torch.randint(0, 100, (5, 10))  # Batch size: 5, Sequence length: 10
#     mask = torch.randint(0, 2, (5, 10))  # Attention mask (binary)
#     token_type_ids = torch.randint(0, 2, (5, 10))  # Token type IDs (binary)

#     # Call the forward method
#     output = model.forward(ids, mask, token_type_ids)

#     # Ensure the output tensor has the correct shape
#     assert output.shape == torch.Size([5, 1])

#     # Ensure the output tensor is on the same device as the input tensors
#     assert output.device == ids.device

#     # assert output.device == token_type_ids.device
#     assert output.device == mask.device
