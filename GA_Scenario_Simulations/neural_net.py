"""Neural network delay predictor for the SYNAPSE scheduler.

The model predicts additional scheduling delay. Existing delay is an input
feature and is not incorrectly treated as delay that the model can remove.
The GA or backtracking solver should continue to choose the final order.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

PRIORITY_WEIGHTS = {"express": 3, "passenger": 2, "freight": 1}
MIN_HEADWAY = 2
RANDOM_SEED = 42


@dataclass(frozen=True)
class Train:
    train_id: str
    train_type: str
    arrival: int
    cross_time: int
    platform: int
    priority: int = 0
    delay_minutes: int = 0

    @property
    def weight(self) -> int:
        return self.priority or PRIORITY_WEIGHTS[self.train_type]


def train_features(
    train: Train, platform_load: int, platform_free_in: int
) -> np.ndarray:
    """Create normalized-independent features for one train state."""
    return np.array(
        [
            train.arrival,
            train.delay_minutes,
            train.cross_time,
            train.platform,
            train.weight,
            platform_load,
            platform_free_in,
        ],
        dtype=np.float64,
    )


def scheduling_delay(train: Train, platform_load: int, platform_free_in: int) -> float:
    """Create a training label using the scheduler's delay equation."""
    effective_arrival = train.arrival + train.delay_minutes
    start = max(effective_arrival, effective_arrival + platform_free_in)
    return float(max(0, start - effective_arrival) + platform_load * MIN_HEADWAY)


def create_training_data(
    sample_count: int = 4000, seed: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    train_types = tuple(PRIORITY_WEIGHTS)
    features = []
    labels = []

    for _ in range(sample_count):
        train = Train(
            train_id="sample",
            train_type=rng.choice(train_types),
            arrival=rng.randint(600, 900),
            cross_time=rng.randint(2, 8),
            platform=rng.randint(1, 24),
            delay_minutes=rng.randint(20, 200),
        )
        platform_load = rng.randint(0, 12)
        platform_free_in = rng.randint(0, 30)
        features.append(train_features(train, platform_load, platform_free_in))
        labels.append(scheduling_delay(train, platform_load, platform_free_in))

    return np.asarray(features), np.asarray(labels, dtype=np.float64).reshape(-1, 1)


class DelayNeuralNetwork:
    """Small two-layer regression network trained with NumPy backpropagation."""

    def __init__(
        self, input_size: int = 7, hidden_size: int = 24, seed: int = RANDOM_SEED
    ):
        rng = np.random.default_rng(seed)
        self.mean = np.zeros(input_size)
        self.scale = np.ones(input_size)
        self.weights1 = rng.normal(
            0, np.sqrt(2 / input_size), (input_size, hidden_size)
        )
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = rng.normal(
            0, np.sqrt(2 / hidden_size), (hidden_size, hidden_size)
        )
        self.bias2 = np.zeros((1, hidden_size))
        self.weights3 = rng.normal(0, np.sqrt(2 / hidden_size), (hidden_size, 1))
        self.bias3 = np.zeros((1, 1))

    def _prepare(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.scale

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, 0)

    def _forward(self, features: np.ndarray):
        hidden1_pre = features @ self.weights1 + self.bias1
        hidden1 = self._relu(hidden1_pre)
        hidden2_pre = hidden1 @ self.weights2 + self.bias2
        hidden2 = self._relu(hidden2_pre)
        output = hidden2 @ self.weights3 + self.bias3
        return hidden1_pre, hidden1, hidden2_pre, hidden2, output

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        validation_features: np.ndarray = None,
        validation_labels: np.ndarray = None,
        epochs: int = 1200,
        learning_rate: float = 0.01,
        patience: int = 60,
    ) -> List[float]:
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale[self.scale == 0] = 1
        normalized = self._prepare(features)
        label_mean = labels.mean()
        label_scale = labels.std() or 1
        normalized_labels = (labels - label_mean) / label_scale
        self.label_mean = label_mean
        self.label_scale = label_scale
        history = []
        self.validation_history = []
        best_validation_loss = float("inf")
        best_weights = None
        stale_epochs = 0

        for _ in range(epochs):
            hidden1_pre, hidden1, hidden2_pre, hidden2, output = self._forward(
                normalized
            )
            error = output - normalized_labels
            history.append(float(np.mean(error**2)))
            if validation_features is not None and validation_labels is not None:
                validation_output = self._forward(self._prepare(validation_features))[
                    -1
                ]
                validation_prediction = (
                    validation_output * self.label_scale + self.label_mean
                )
                validation_loss = float(
                    np.mean((validation_prediction - validation_labels) ** 2)
                )
                self.validation_history.append(validation_loss)
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    stale_epochs = 0
                    best_weights = self._copy_weights()
                else:
                    stale_epochs += 1
                    if stale_epochs >= patience:
                        break
            output_gradient = 2 * error / len(features)
            weights3_gradient = hidden2.T @ output_gradient
            bias3_gradient = output_gradient.sum(axis=0, keepdims=True)
            hidden2_gradient = (output_gradient @ self.weights3.T) * (hidden2_pre > 0)
            weights2_gradient = hidden1.T @ hidden2_gradient
            bias2_gradient = hidden2_gradient.sum(axis=0, keepdims=True)
            hidden1_gradient = (hidden2_gradient @ self.weights2.T) * (hidden1_pre > 0)
            weights1_gradient = normalized.T @ hidden1_gradient
            bias1_gradient = hidden1_gradient.sum(axis=0, keepdims=True)

            self.weights3 -= learning_rate * weights3_gradient
            self.bias3 -= learning_rate * bias3_gradient
            self.weights2 -= learning_rate * weights2_gradient
            self.bias2 -= learning_rate * bias2_gradient
            self.weights1 -= learning_rate * weights1_gradient
            self.bias1 -= learning_rate * bias1_gradient

        if best_weights is not None:
            self._restore_weights(best_weights)
        return history

    def _copy_weights(self):
        return tuple(
            array.copy()
            for array in (
                self.weights1,
                self.bias1,
                self.weights2,
                self.bias2,
                self.weights3,
                self.bias3,
            )
        )

    def _restore_weights(self, weights):
        (
            self.weights1,
            self.bias1,
            self.weights2,
            self.bias2,
            self.weights3,
            self.bias3,
        ) = tuple(array.copy() for array in weights)

    def predict(self, features: np.ndarray) -> np.ndarray:
        _, _, _, _, output = self._forward(self._prepare(features))
        return np.maximum(0, output * self.label_scale + self.label_mean).ravel()

    def predict_train(
        self, train: Train, platform_load: int, platform_free_in: int
    ) -> float:
        features = train_features(train, platform_load, platform_free_in).reshape(1, -1)
        return float(self.predict(features)[0])


def create_demo_trains(count: int = 12, seed: int = RANDOM_SEED) -> List[Train]:
    rng = random.Random(seed)
    train_types = tuple(PRIORITY_WEIGHTS)
    return [
        Train(
            train_id=f"T{index}",
            train_type=rng.choice(train_types),
            arrival=rng.randint(600, 660),
            cross_time=rng.randint(2, 8),
            platform=rng.randint(1, 6),
            delay_minutes=rng.randint(20, 200),
        )
        for index in range(1, count + 1)
    ]


def print_prediction_table(model: DelayNeuralNetwork, trains: List[Train]) -> None:
    print()
    print(
        "Train | Type      | Planned | Existing | Effective | Predicted new | Predicted total | Platform"
    )
    print("-" * 96)
    for index, train in enumerate(trains):
        platform_load = index % 4
        platform_free_in = (index * 3) % 15
        prediction = model.predict_train(train, platform_load, platform_free_in)
        effective_arrival = train.arrival + train.delay_minutes
        print(
            f"{train.train_id:5} | {train.train_type:9} | {train.arrival:7} | "
            f"{train.delay_minutes:8} | {effective_arrival:9} | {prediction:13.2f} | "
            f"{train.delay_minutes + prediction:15.2f} | {train.platform:8}"
        )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    errors = predicted - actual
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    total_variance = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = float(1 - np.sum(errors**2) / total_variance) if total_variance else 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "r_squared": r_squared,
        "max_error": float(np.max(np.abs(errors))),
    }


def linear_baseline(
    training_features: np.ndarray,
    training_labels: np.ndarray,
    validation_features: np.ndarray,
) -> np.ndarray:
    training_with_bias = np.column_stack(
        (np.ones(len(training_features)), training_features)
    )
    weights, _, _, _ = np.linalg.lstsq(
        training_with_bias, training_labels.ravel(), rcond=None
    )
    return np.maximum(
        0,
        np.column_stack((np.ones(len(validation_features)), validation_features))
        @ weights,
    )


def plot_neural_network(
    model: DelayNeuralNetwork,
    history: List[float],
    actual: np.ndarray,
    predicted: np.ndarray,
    output: str = "neural_network_training.png",
) -> None:
    """Save training behavior, prediction quality, and network structure."""
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    figure.suptitle("SYNAPSE Neural Network Delay Predictor", fontsize=16, fontweight="bold")

    training_loss = np.asarray(history) * model.label_scale**2
    axes[0, 0].plot(training_loss, label="Training MSE", color="#00798c")
    if model.validation_history:
        axes[0, 0].plot(model.validation_history, label="Validation MSE", color="#d1495b")
    axes[0, 0].set_title("Learning Curve")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Mean squared error (minutes²)")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].scatter(actual, predicted, alpha=0.35, color="#00798c", edgecolors="none")
    limits = [0, max(float(actual.max()), float(predicted.max())) * 1.05]
    axes[0, 1].plot(limits, limits, "--", color="#d1495b", label="Perfect prediction")
    axes[0, 1].set_title("Actual vs Predicted Delay")
    axes[0, 1].set_xlabel("Actual new delay (minutes)")
    axes[0, 1].set_ylabel("Predicted new delay (minutes)")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    residuals = predicted - actual
    axes[1, 0].hist(residuals, bins=30, color="#edae49", edgecolor="white")
    axes[1, 0].axvline(0, color="#333333", linestyle="--")
    axes[1, 0].set_title("Prediction Errors")
    axes[1, 0].set_xlabel("Prediction error (minutes)")
    axes[1, 0].set_ylabel("Validation samples")
    axes[1, 0].grid(axis="y", alpha=0.3)

    layer_sizes = [7, model.weights1.shape[1], model.weights2.shape[1], 1]
    layer_names = ["Inputs", "Hidden 1", "Hidden 2", "Output"]
    x_positions = np.arange(len(layer_sizes))
    for layer_index, (x_position, size) in enumerate(zip(x_positions, layer_sizes)):
        node_count = min(size, 10)
        y_positions = np.linspace(0.2, 0.8, node_count)
        axes[1, 1].scatter(
            np.full(node_count, x_position),
            y_positions,
            s=90,
            color="#00798c" if layer_index < 3 else "#d1495b",
            zorder=2,
        )
        if layer_index < len(layer_sizes) - 1:
            next_count = min(layer_sizes[layer_index + 1], 10)
            next_y_positions = np.linspace(0.2, 0.8, next_count)
            for y_position in y_positions:
                for next_y_position in next_y_positions:
                    axes[1, 1].plot(
                        [x_position, x_position + 1],
                        [y_position, next_y_position],
                        color="#bbbbbb",
                        linewidth=0.35,
                        alpha=0.35,
                        zorder=1,
                    )
    axes[1, 1].set_xticks(x_positions)
    axes[1, 1].set_xticklabels(layer_names)
    axes[1, 1].set_title("Network Architecture (7-24-24-1)")
    axes[1, 1].set_xlim(-0.3, len(layer_sizes) - 0.7)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_yticks([])
    axes[1, 1].grid(axis="x", alpha=0.2)

    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"[Plot] Saved neural-network visualization to {output}")


def main():
    features, labels = create_training_data(sample_count=6000)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(features))
    split_index = int(len(features) * 0.8)
    training_indices = indices[:split_index]
    validation_indices = indices[split_index:]
    training_features = features[training_indices]
    training_labels = labels[training_indices]
    validation_features = features[validation_indices]
    validation_labels = labels[validation_indices]
    model = DelayNeuralNetwork()
    history = model.fit(
        training_features,
        training_labels,
        validation_features,
        validation_labels,
    )
    validation_predictions = model.predict(validation_features)
    actual = validation_labels.ravel()
    model_metrics = regression_metrics(actual, validation_predictions)
    mean_baseline_predictions = np.full_like(actual, np.mean(training_labels))
    mean_baseline_metrics = regression_metrics(actual, mean_baseline_predictions)
    linear_predictions = linear_baseline(
        training_features, training_labels, validation_features
    )
    linear_baseline_metrics = regression_metrics(actual, linear_predictions)
    print("Model comparison (validation set)")
    print("Model           | MAE (min) | RMSE (min) | R^2    | Max error (min)")
    print("-" * 70)
    for name, metrics in (
        ("Mean baseline", mean_baseline_metrics),
        ("Linear baseline", linear_baseline_metrics),
        ("Neural network", model_metrics),
    ):
        print(
            f"{name:15} | {metrics['mae']:9.2f} | {metrics['rmse']:10.2f} | "
            f"{metrics['r_squared']:6.3f} | {metrics['max_error']:15.2f}"
        )
    mean_improvement = 100 * (1 - model_metrics["mae"] / mean_baseline_metrics["mae"])
    print(f"MAE improvement vs mean: {mean_improvement:.1f}%")
    if linear_baseline_metrics["mae"] > 1e-9:
        linear_improvement = 100 * (
            1 - model_metrics["mae"] / linear_baseline_metrics["mae"]
        )
        print(f"MAE improvement vs linear: {linear_improvement:.1f}%")
    else:
        print("MAE improvement vs linear: not applicable (baseline MAE is 0.00)")
    print(
        "Note: the generated scheduling rule is linear, so the linear baseline "
        "is expected to be strongest."
    )
    print(f"Training samples: {len(training_features)}")
    print(f"Validation samples: {len(validation_features)}")
    print(f"Epochs completed: {len(history)}")
    print(f"Neural-network validation MAE: {model_metrics['mae']:.2f} minutes")
    plot_neural_network(model, history, actual, validation_predictions)
    print_prediction_table(model, create_demo_trains())


if __name__ == "__main__":
    main()
