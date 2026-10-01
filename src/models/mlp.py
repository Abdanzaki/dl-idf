"""
DL-IDF: Deep Learning-Based Intrusion Detection Framework for Industrial IoT
Phase 4 — Deep Learning Architectures: Multi-Layer Perceptron (src/models/mlp.py)

B.E. Computer Science & Engineering (Major Project)
--------------------------------------------------------------------------------
This module defines the Deep Multi-Layer Perceptron (MLP) architecture tailored
for industrial IoT network intrusion detection using Keras / TensorFlow.

College Viva Voce Architectural Guide & Technical Defense:
================================================================================
1. Why an MLP for Industrial IoT Network Intrusion Detection?
   Classical algorithms (e.g. Decision Trees, Logistic Regression) rely on linear
   or axis-aligned decision boundaries. An MLP is a feedforward artificial neural
   network capable of learning complex, non-linear relationships across tabular
   network flow features (e.g., correlations between packet sizes, duration,
   protocol flags, and connection states). Under the Universal Approximation
   Theorem (Hornik et al., 1989), a feedforward network with non-linear activation
   functions can approximate any continuous function on compact subsets of R^n.

2. Funnel (Bottleneck) Architecture (128 -> 64 -> 32):
   The decreasing layer widths implement hierarchical feature abstraction:
   - Layer 1 (128 units): Projects the 50 standardized input features into a
     higher-dimensional latent space to decouple complex cross-feature interactions.
   - Layer 2 (64 units): Distills the salient representations into an intermediate
     compact manifold.
   - Layer 3 (32 units): Synthesizes a high-level representation directly preceding
     the classification decision boundary.
   This pyramidal compression forces the network to learn generalized representations
   rather than memorizing spurious feature noise.

3. Activation Function — ReLU (Rectified Linear Unit, f(x) = max(0, x)):
   - Solves the Vanishing Gradient Problem: Unlike Sigmoid or Tanh whose derivatives
     saturate to 0 for large positive/negative activations (d/dx sigmoid <= 0.25),
     ReLU has a constant derivative of 1 for all positive inputs, allowing gradient
     signals to backpropagate effectively through all layers.
   - Computational Efficiency: Involves a simple threshold comparison at zero,
     drastically accelerating forward and backward passes compared to exponentials.
   - Induces Sparsity: Deactivates non-informative features (output 0), mimicking
     biological neural firing and reducing parameter co-dependencies.

4. Regularization — Dropout (rate = 0.3):
   - Proposed by Srivastava et al. (2014).
   - During each training forward-pass, each neuron in the hidden layer has a 30%
     probability of being temporarily zeroed out (dropped).
   - Prevents neuron co-adaptation (where one neuron corrects mistakes of another).
   - Effectively trains an exponential ensemble of 2^k sub-architectures sharing
     weights, providing superior generalization on unseen network traffic attacks.
   - During inference (testing), Dropout is disabled, and activations are scaled
     by (1 - p) to preserve expected values.

5. Output Layer — Dense(1, activation='sigmoid'):
   - Sigmoid: sigma(z) = 1 / (1 + exp(-z)).
   - Squashes the unbounded logit output z into the open interval (0, 1), representing
     the posterior probability P(Y=1 | X), i.e., probability of being an attack.
   - Enables calibrated thresholding (default decision threshold = 0.5).

6. Loss Function — Binary Cross-Entropy (Log Loss):
   - L = - [ y * log(p) + (1 - y) * log(1 - p) ]
   - Strictly derived from Maximum Likelihood Estimation (MLE) for Bernoulli trials.
   - Penalizes confident incorrect predictions exponentially, driving steep
     gradient updates to correct classification mistakes rapidly.

7. Optimizer — Adam (Adaptive Moment Estimation):
   - Combines the advantages of AdaGrad (adaptive learning rates based on historical
     squared gradients) and RMSProp (exponential moving average of squared gradients)
     with classical Momentum (first moment vector m_t).
   - Robust to noisy gradients common in stochastic mini-batch network flow training.
--------------------------------------------------------------------------------
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_mlp(
    input_dim: int = 50,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Constructs and compiles the Keras Sequential Deep Multi-Layer Perceptron (MLP).

    Architecture:
        Input(shape=(input_dim,))
        Dense(128, activation='relu') -> Dropout(0.3)
        Dense(64,  activation='relu') -> Dropout(0.3)
        Dense(32,  activation='relu') -> Dropout(0.3)
        Dense(1,   activation='sigmoid')

    Args:
        input_dim: Number of input features (default: 50 for TON_IoT processed flows).
        learning_rate: Initial learning rate for Adam optimizer (default: 0.001).

    Returns:
        Compiled Keras Sequential model ready for training and evaluation.
    """
    model = keras.Sequential([
        # ----------------------------------------------------------------------
        # INPUT SPECIFICATION
        # Explicitly defines the expected shape of each input sample: (batch_size, 50).
        # This fixes weight initialization and enables immediate model.summary() inspection.
        # ----------------------------------------------------------------------
        layers.Input(shape=(input_dim,), name="input_network_flow_features"),

        # ----------------------------------------------------------------------
        # HIDDEN LAYER 1: Feature Expansion & Non-Linear Mapping
        # 128 hidden units. 50 inputs -> 128 outputs.
        # Trainable parameters = (50 input features * 128 weights) + 128 biases = 6,528 params.
        # ----------------------------------------------------------------------
        layers.Dense(
            units=128,
            activation="relu",
            kernel_initializer="he_normal",  # He-normal is optimal for ReLU activations
            name="dense_layer_1_128"
        ),
        # Regularization: Randomly deactivates 30% of neurons to prevent overfitting
        layers.Dropout(rate=0.3, name="dropout_1_0.3"),

        # ----------------------------------------------------------------------
        # HIDDEN LAYER 2: Intermediate Latent Feature Representation
        # 64 hidden units. 128 inputs -> 64 outputs.
        # Trainable parameters = (128 * 64) + 64 = 8,256 params.
        # ----------------------------------------------------------------------
        layers.Dense(
            units=64,
            activation="relu",
            kernel_initializer="he_normal",
            name="dense_layer_2_64"
        ),
        layers.Dropout(rate=0.3, name="dropout_2_0.3"),

        # ----------------------------------------------------------------------
        # HIDDEN LAYER 3: Low-Dimensional Bottleneck Feature Synthesis
        # 32 hidden units. 64 inputs -> 32 outputs.
        # Trainable parameters = (64 * 32) + 32 = 2,080 params.
        # ----------------------------------------------------------------------
        layers.Dense(
            units=32,
            activation="relu",
            kernel_initializer="he_normal",
            name="dense_layer_3_32"
        ),
        layers.Dropout(rate=0.3, name="dropout_3_0.3"),

        # ----------------------------------------------------------------------
        # OUTPUT LAYER: Binary Classification Decision Unit
        # 1 unit. 32 inputs -> 1 output.
        # Trainable parameters = (32 * 1) + 1 = 33 params.
        # Sigmoid activation: outputs posterior attack probability P(Attack=1 | Flow).
        # ----------------------------------------------------------------------
        layers.Dense(
            units=1,
            activation="sigmoid",
            name="output_attack_probability"
        )
    ], name="DL_IDF_MLP_Classifier")

    # --------------------------------------------------------------------------
    # MODEL COMPILATION
    # - Optimizer: Adam with configurable learning rate.
    # - Loss: binary_crossentropy for 2-class classification.
    # - Metrics: Accuracy tracked per mini-batch and validation split.
    # --------------------------------------------------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    # Smoke-test architecture and print layer-by-layer parameter summary
    print("\n" + "=" * 70)
    print("      DL-IDF PHASE 4: MULTI-LAYER PERCEPTRON (MLP) ARCHITECTURE")
    print("=" * 70)
    mlp_model = build_mlp(input_dim=50)
    mlp_model.summary()
    print("\nTotal Layers:", len(mlp_model.layers))
    print("Compilation Loss:", mlp_model.loss)
    print("Optimizer:", type(mlp_model.optimizer).__name__)
    print("=" * 70 + "\n")
