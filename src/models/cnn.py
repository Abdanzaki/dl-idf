"""
DL-IDF: Deep Learning-Based Intrusion Detection Framework for Industrial IoT
Phase 4 — Deep Learning Architectures: 1D Convolutional Neural Network (src/models/cnn.py)

B.E. Computer Science & Engineering (Major Project)
--------------------------------------------------------------------------------
This module defines the 1D Convolutional Neural Network (1D-CNN) architecture
tailored for industrial IoT network flow intrusion detection using Keras / TensorFlow.

College Viva Voce Architectural Guide & Technical Defense:
================================================================================
1. Why 1D-CNN for Industrial IoT Tabular Network Flows?
   - While 2D-CNNs excel on grid-structured spatial data (images), 1D-CNNs process
     sequential or feature-vector data by sliding 1-dimensional convolution kernels
     across contiguous features.
   - In network flow telemetry (TON_IoT), correlated sub-groups of features exist
     (e.g., source/destination bytes, packet volumes, protocol flags, DNS status codes).
     1D convolutions capture local cross-feature correlation motifs that feedforward
     MLPs treat as isolated coordinates.
   - Weight Sharing & Sparsity of Connections: Instead of connecting every input
     to every hidden neuron, small convolution filters (kernel size = 3) are reused
     across the entire feature vector, dramatically reducing model parameter count
     and preventing overfitting.

2. Reshaping to (n_features, 1):
   - A standard 2D tabular feature vector of shape (batch_size, 50) is reshaped
     into a 3D tensor of shape (batch_size, 50, 1), where:
     * Dimension 1: Batch size (number of network flow samples).
     * Dimension 2: Steps / Sequence length (50 network flow features).
     * Dimension 3: Channels (1 single feature value per step).
   - This provides the temporal/spatial channel representation required by Conv1D.

3. First Convolutional Block — Conv1D(64, kernel_size=3) -> MaxPooling1D(pool_size=2):
   - Conv1D(64, 3): Slides 64 separate filters of length 3 across the 50 features.
     Each filter computes an affine combination of 3 consecutive features:
     y_k = ReLU( sum_{i=0}^{2} W_i * x_{j+i} + b ).
     Parameters: (3 kernel_size * 1 input_channel * 64 filters) + 64 biases = 256 params.
   - MaxPooling1D(2): Takes the maximum activation over non-overlapping windows of
     size 2, halving feature length from 50 to 25. This introduces local translation
     invariance (minor feature scale fluctuations do not affect detection).

4. Second Convolutional Block — Conv1D(32, kernel_size=3) -> GlobalMaxPooling1D():
   - Conv1D(32, 3): Extracts higher-order composite anomaly patterns from the 64
     low-level feature maps.
     Parameters: (3 kernel_size * 64 input_channels * 32 filters) + 32 biases = 6,176 params.
   - GlobalMaxPooling1D(): Takes the single maximum activation across the entire
     remaining sequence for each of the 32 filters, outputting a fixed 1D vector of
     length 32.
     * VIVA DEFENSE: Why GlobalMaxPooling instead of Flatten()?
       Flattening 25 * 32 = 800 activations into Dense(64) would require 51,264 parameters,
       introducing severe risk of overfitting on tabular data. GlobalMaxPooling
       extracts the most prominent anomaly indicator per filter with 0 extra parameters!

5. Classification Head — Dense(64, relu) -> Dropout(0.3) -> Dense(1, sigmoid):
   - Dense(64, relu): Synthesizes the 32 distilled spatial anomaly features into
     a high-level decision manifold.
   - Dropout(0.3): Regularizes the dense representation by deactivating 30% of units
     during training.
   - Dense(1, sigmoid): Squashes the logit into posterior attack probability P(Y=1 | X).

6. Loss & Optimization:
   - Binary Cross-Entropy: Penalizes probabilistic error strictly according to
     information-theoretic entropy.
   - Adam Optimizer: Ensures stable convergence on non-convex loss surfaces.
--------------------------------------------------------------------------------
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn1d(
    input_dim: int = 50,
    learning_rate: float = 0.001
) -> keras.Model:
    """
    Constructs and compiles the Keras Sequential 1D-Convolutional Neural Network (1D-CNN).

    Architecture:
        Input(shape=(input_dim, 1))
        Conv1D(64, kernel_size=3, padding='same', activation='relu')
        MaxPooling1D(pool_size=2)
        Conv1D(32, kernel_size=3, padding='same', activation='relu')
        GlobalMaxPooling1D()
        Dense(64, activation='relu')
        Dropout(0.3)
        Dense(1, activation='sigmoid')

    Args:
        input_dim: Number of input features (default: 50).
        learning_rate: Initial learning rate for Adam optimizer (default: 0.001).

    Returns:
        Compiled Keras Sequential 1D-CNN model ready for training and evaluation.
    """
    model = keras.Sequential([
        # ----------------------------------------------------------------------
        # INPUT SPECIFICATION
        # Explicit shape: (batch_size, input_dim, 1).
        # Treats the 50 standardized network flow features as a 1D sequence of length 50
        # with 1 input channel.
        # ----------------------------------------------------------------------
        layers.Input(shape=(input_dim, 1), name="input_network_flow_sequence"),

        # ----------------------------------------------------------------------
        # CONV1D LAYER 1: Low-Level Feature Cross-Correlation Extraction
        # 64 filters, kernel size = 3, stride = 1, padding = 'same' (preserves length 50).
        # Computes localized dot products across 3 contiguous feature dimensions.
        # Trainable parameters = (3 * 1 * 64) + 64 = 256 params.
        # Output shape = (batch_size, 50, 64)
        # ----------------------------------------------------------------------
        layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name="conv1d_layer_1_64"
        ),

        # ----------------------------------------------------------------------
        # MAXPOOLING1D LAYER 1: Temporal/Spatial Downsampling
        # pool_size = 2, strides = 2.
        # Halves the spatial sequence length from 50 -> 25 while retaining peak activations.
        # Parameter count = 0.
        # Output shape = (batch_size, 25, 64)
        # ----------------------------------------------------------------------
        layers.MaxPooling1D(
            pool_size=2,
            name="maxpooling1d_layer_1"
        ),

        # ----------------------------------------------------------------------
        # CONV1D LAYER 2: Higher-Order Anomaly Pattern Synthesis
        # 32 filters, kernel size = 3, padding = 'same'.
        # Combines the 64 low-level pooled feature maps into 32 higher-order representations.
        # Trainable parameters = (3 * 64 * 32) + 32 = 6,176 params.
        # Output shape = (batch_size, 25, 32)
        # ----------------------------------------------------------------------
        layers.Conv1D(
            filters=32,
            kernel_size=3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            name="conv1d_layer_2_32"
        ),

        # ----------------------------------------------------------------------
        # GLOBAL MAX POOLING 1D: Sequence Aggregation & Spatial Regularization
        # Collapses the temporal dimension (25) into the single strongest activation per filter.
        # Output shape = (batch_size, 32).
        # Avoids parameter explosion of Flatten() (which would be 25 * 32 = 800 units).
        # Parameter count = 0.
        # ----------------------------------------------------------------------
        layers.GlobalMaxPooling1D(name="global_max_pooling_1d"),

        # ----------------------------------------------------------------------
        # DENSE LAYER: High-Level Feature Integration
        # 64 fully connected units with ReLU activation.
        # Trainable parameters = (32 * 64) + 64 = 2,112 params.
        # Output shape = (batch_size, 64)
        # ----------------------------------------------------------------------
        layers.Dense(
            units=64,
            activation="relu",
            kernel_initializer="he_normal",
            name="dense_projection_64"
        ),

        # Regularization: 30% dropout rate prevents overfitting on specific feature combinations
        layers.Dropout(rate=0.3, name="dropout_cnn_0.3"),

        # ----------------------------------------------------------------------
        # OUTPUT LAYER: Binary Attack Probability
        # 1 unit with Sigmoid activation.
        # Trainable parameters = (64 * 1) + 1 = 65 params.
        # Output shape = (batch_size, 1) -> Posterior probability in [0, 1].
        # ----------------------------------------------------------------------
        layers.Dense(
            units=1,
            activation="sigmoid",
            name="output_attack_probability"
        )
    ], name="DL_IDF_1D_CNN_Classifier")

    # --------------------------------------------------------------------------
    # MODEL COMPILATION
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
    print("      DL-IDF PHASE 4: 1D CONVOLUTIONAL NEURAL NETWORK (1D-CNN)")
    print("=" * 70)
    cnn_model = build_cnn1d(input_dim=50)
    cnn_model.summary()
    print("\nTotal Layers:", len(cnn_model.layers))
    print("Compilation Loss:", cnn_model.loss)
    print("Optimizer:", type(cnn_model.optimizer).__name__)
    print("=" * 70 + "\n")
