from typing import Optional, Union

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[list[int], tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU(),
        initialization: str = "default",
        dropout_rate: Optional[float] = None,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + hidden_dims
        model = []

        # Example: Initialization for a layer based on activation function
        if activation == nn.ReLU():
            gain = nn.init.calculate_gain("relu")
        elif activation == nn.LeakyReLU():
            gain = nn.init.calculate_gain("leaky_relu")
        elif activation == nn.Tanh():
            gain = nn.init.calculate_gain("tanh")
        elif activation == nn.Sigmoid():
            gain = nn.init.calculate_gain("sigmoid")
        elif activation == nn.ELU():
            gain = nn.init.calculate_gain("elu")
        elif activation == nn.Softplus():
            gain = nn.init.calculate_gain("softplus")
        elif activation == nn.Softsign():
            gain = nn.init.calculate_gain("softsign")
        else:
            gain = 1.0  # Default if no listed activation matches

        # Initialize hidden layers
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            linear_layer = nn.Linear(in_dim, out_dim)
            if initialization == "default":
                nn.init.xavier_uniform_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.1)

            elif initialization == "actor":
                nn.init.orthogonal_(linear_layer.weight, gain=1.414)
                linear_layer.bias.data.fill_(0.0)

            elif initialization == "critic":
                nn.init.orthogonal_(linear_layer.weight, gain=1.414)
                linear_layer.bias.data.fill_(0.0)

            model += (
                [linear_layer, activation] if activation is not None else [linear_layer]
            )

            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]

        # Initialize output layer
        if output_dim is not None:
            linear_layer = nn.Linear(hidden_dims[-1], output_dim)
            if initialization == "default":
                nn.init.xavier_uniform_(linear_layer.weight, gain=gain)
                linear_layer.bias.data.fill_(0.0)

            elif initialization == "actor":
                nn.init.orthogonal_(linear_layer.weight, gain=0.01)
                linear_layer.bias.data.fill_(0.0)

            elif initialization == "critic":
                nn.init.orthogonal_(linear_layer.weight, gain=1)
                linear_layer.bias.data.fill_(0.0)

            model += [linear_layer]
            self.output_dim = output_dim

        self.model = nn.Sequential(*model).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_neural_network(
    inputs: list[list],
    X: list[list],
    Y: list[list],
    test_Y: list[list],
    epochs: int,
    learning_rate: float,
):
    # concat list to matrix
    state_dim = len(X[0][0])
    inputs = torch.tensor(inputs, dtype=torch.float32).view(-1, state_dim)
    X = torch.tensor(X, dtype=torch.float32).view(-1, state_dim)
    Y = torch.tensor(Y, dtype=torch.float32).view(-1, state_dim)
    test_Y = torch.tensor(test_Y, dtype=torch.float32).view(-1, state_dim)

    v_model = MLP(
        input_dim=X.shape[1],
        hidden_dims=[128, 128],
        output_dim=Y.shape[1],
        activation=nn.Tanh(),
        initialization="default",
        dropout_rate=0.2,
    )
    c_model = MLP(
        input_dim=X.shape[1],
        hidden_dims=[128, 128],
        output_dim=Y.shape[1],
        activation=nn.Tanh(),
        initialization="default",
        dropout_rate=0.2,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        [
            {"params": v_model.parameters(), "lr": learning_rate},
            {"params": c_model.parameters(), "lr": learning_rate},
        ]
    )

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v_model.to(device)
    c_model.to(device)
    inputs, X, Y, test_Y = (
        inputs.to(device),
        X.to(device),
        Y.to(device),
        test_Y.to(device),
    )

    v_model.train(), c_model.train()
    val_loss_list = [0.0]
    stack = -1
    for epoch in range(epochs):
        v_outputs, c_outputs = v_model(inputs), c_model(inputs)
        Y_pred = v_outputs * X + c_outputs

        # implement validation loss
        if epoch % 100 == 0:
            val_v_outputs, val_c_outputs = v_model(test_Y), c_model(test_Y)
            val_Y_pred = val_v_outputs * test_Y + val_c_outputs
            val_loss = criterion(val_Y_pred, test_Y)

            if val_loss.item() > val_loss_list[-1]:
                stack += 1

            if stack > 5:
                print("Early stopping...")
                break

            val_loss_list.append(val_loss.item())

            print(f"Validation Loss at Epoch {epoch}: {val_loss.item():.4f}")

        loss = criterion(Y_pred, Y)
        loss += (
            (v_outputs[:, [0, 1, 2, 6, 7, 8, 9]] - 1.0).pow(2).mean()
        )  # regularization
        loss += c_outputs[:, [0, 1, 2, 6, 7, 8, 9]].pow(2).mean()  # regularization

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

    return v_model.cpu().eval(), c_model.cpu().eval()
