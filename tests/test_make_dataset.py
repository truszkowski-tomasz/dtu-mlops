import pytest
import pandas as pd


@pytest.fixture
def sample_dataframe():
    # Create a sample dataframe for testing
    df = pd.DataFrame(
        {
            "text": [
                "This is a sample text",
                "Another sample text",
                "This is a sample text",
                "Another sample text",
            ],
            "label": [0, 1, 0, 1],
        }
    )
    return df


# TODO Fix this test (pipeline rises error because it does not have access to the model)
# def test_preprocess_data(sample_dataframe):
#     train_size = 0.8
#     max_len = 100

#     train_set, val_set = preprocess_data(sample_dataframe, train_size, max_len)

#     # Check the shapes of the datasets
#     assert len(train_set) == int(len(sample_dataframe) * train_size)
#     assert len(val_set) == len(sample_dataframe) - int(len(sample_dataframe) * train_size)

#     # Check the shapes of the input tensors
#     assert train_set[0][0].size() == torch.Size([max_len])
#     assert train_set[1][0].size() == torch.Size([max_len])
#     assert train_set[2][0].size() == torch.Size([max_len])

#     assert val_set[0][0].size() == torch.Size([max_len])
