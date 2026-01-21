import torch
from mlops_mnist_classifier.model import CNNModel


def test_model_initialization():
    """Test that the model initializes correctly."""
    model = CNNModel()
    assert model is not None


def test_model_forward_pass():
    """Test that the model can process a batch of MNIST images."""
    model = CNNModel()
    model.eval()

    # Create a batch of dummy MNIST images (batch_size=4, channels=1, height=28, width=28)
    batch = torch.randn(4, 1, 28, 28)

    with torch.no_grad():
        output = model(batch)

    # Check output shape: (batch_size, num_classes)
    assert output.shape == (4, 10)

    # Check output is valid (no NaN or Inf)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_model_output_range():
    """Test that model outputs are reasonable logits."""
    model = CNNModel()
    model.eval()

    batch = torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        output = model(batch)

    # Logits should be in a reasonable range (not exploding)
    assert output.abs().max() < 100


def test_model_gradient_flow():
    """Test that gradients can flow through the model."""
    model = CNNModel()
    model.train()

    batch = torch.randn(2, 1, 28, 28, requires_grad=True)
    targets = torch.tensor([3, 7])

    output = model(batch)
    loss = torch.nn.functional.cross_entropy(output, targets)
    loss.backward()

    # Check that at least some parameters have gradients
    has_gradients = any(p.grad is not None for p in model.parameters())
    assert has_gradients


def test_model_batch_independence():
    """Test that model processes different batch sizes consistently."""
    model = CNNModel()
    model.eval()

    batch_sizes = [1, 4, 16, 32]

    with torch.no_grad():
        for batch_size in batch_sizes:
            batch = torch.randn(batch_size, 1, 28, 28)
            output = model(batch)
            assert output.shape == (batch_size, 10)
