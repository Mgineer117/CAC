from base import (
    call_mdp_data,
    compute_nominal_dynamics,
    lstq_regression,
    plot_prediction,
    plot_prediction_error,
    simulate,
)
from neural_netowrk import train_neural_network

if __name__ == "__main__":
    train_states, train_actions, train_next_states, train_terminals, train_x_dots = (
        call_mdp_data(data_type="train")
    )
    test_states, test_actions, test_next_states, test_terminals, test_x_dots = (
        call_mdp_data(data_type="test")
    )
    x_hat_dots = compute_nominal_dynamics(train_states, train_actions)
    v, c = lstq_regression(train_x_dots, x_hat_dots, outlier_removal=True)
    # v, c = train_neural_network(
    #     train_states,
    #     x_hat_dots,
    #     train_x_dots,
    #     test_x_dots,
    #     epochs=10000,
    #     learning_rate=1e-5,
    # )

    true_matrices, pred_matrices, error_matrices = simulate(
        test_states, test_actions, v, c
    )
    plot_prediction(true_matrices, pred_matrices)
    plot_prediction_error(error_matrices)
    # fit_reference_controls(test_actions, test_indices)
