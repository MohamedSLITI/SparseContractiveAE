import numpy as np
import matplotlib.pyplot as plt
import imageio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# -----------------------------
# Simulate Sensor Data
# -----------------------------
np.random.seed(42)
n_samples = 1000
n_sensors = 5
time = np.linspace(0, 10, n_samples)

sensor_data = np.zeros((n_samples, n_sensors))
sensor_data[:, 0] = np.sin(time) + 0.05*np.random.randn(n_samples)
sensor_data[:, 1] = np.cos(time) + 0.05*np.random.randn(n_samples)
sensor_data[:, 2] = np.sin(2*time) + 0.05*np.random.randn(n_samples)
sensor_data[:, 3] = np.cos(2*time) + 0.05*np.random.randn(n_samples)
sensor_data[:, 4] = np.sin(time)*np.cos(time) + 0.05*np.random.randn(n_samples)

# Standardize instead of min-max scaling for better alignment
X = (sensor_data - np.mean(sensor_data, axis=0)) / np.std(sensor_data, axis=0)
x_train = X[:800]
x_val = X[800:810]  # small validation batch for GIF

# -----------------------------
# Helper: Plot reconstruction
# -----------------------------
def plot_reconstruction(orig, recon, title, fname):
    plt.figure(figsize=(10,4))
    for i in range(len(orig)):
        plt.plot(orig[i], 'k-o', label='Original' if i==0 else "")
        plt.plot(recon[i], 'r--x', label='Reconstruction' if i==0 else "")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# -----------------------------
# Sparse Autoencoder
# -----------------------------
input_dim = n_sensors
latent_dim = 5  # increased latent dimensions
hidden_units = 32  # larger hidden layer
l1 = 1e-4

def build_sparse_ae(input_dim, latent_dim, hidden_units=32, l1=1e-4):
    inp = keras.Input(shape=(input_dim,))
    x = layers.Dense(hidden_units, activation='relu')(inp)
    encoded = layers.Dense(latent_dim, activation='relu',
                           activity_regularizer=regularizers.l1(l1))(x)
    x = layers.Dense(hidden_units, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='linear')(x)  # linear output for standardized data
    model = keras.Model(inp, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model

sparse_ae = build_sparse_ae(input_dim, latent_dim, hidden_units, l1)
frames_sparse = []

epochs = 50  # increased epochs for better convergence
for epoch in range(epochs):
    sparse_ae.fit(x_train, x_train, epochs=1, batch_size=32, verbose=0)
    recon = sparse_ae.predict(x_val)
    fname = f"sparse_epoch_{epoch+1}.png"
    plot_reconstruction(x_val, recon, f"Sparse AE - Epoch {epoch+1}", fname)
    frames_sparse.append(fname)

with imageio.get_writer("sparse_recon.gif", mode='I', duration=0.5) as writer:
    for fname in frames_sparse:
        img = imageio.v2.imread(fname)
        writer.append_data(img)

print("Sparse AE GIF saved as sparse_recon.gif")

# -----------------------------
# Contractive Autoencoder
# -----------------------------
lambda_contractive = 1e-4  # smaller penalty for better alignment

# Encoder
inputs = keras.Input(shape=(input_dim,))
h = layers.Dense(hidden_units, activation='relu')(inputs)
encoded = layers.Dense(latent_dim, activation='linear')(h)
encoder = keras.Model(inputs, encoded)

# Decoder
encoded_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(hidden_units, activation='relu')(encoded_input)
decoded = layers.Dense(input_dim, activation='linear')(x)
decoder = keras.Model(encoded_input, decoded)

# Full contractive AE
inputs_full = keras.Input(shape=(input_dim,))
encoded_full = encoder(inputs_full)
decoded_full = decoder(encoded_full)
contractive_ae = keras.Model(inputs_full, decoded_full)

optimizer = keras.optimizers.Adam()
frames_contractive = []

x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
x_val_tensor = tf.convert_to_tensor(x_val, dtype=tf.float32)

for epoch in range(epochs):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x_train_tensor)
        latent = encoder(x_train_tensor)
        decoded = decoder(latent)
        mse_loss = tf.reduce_mean(tf.square(x_train_tensor - decoded))

        # Contractive penalty using Jacobian
        contractive_penalty = 0
        for j in range(latent_dim):
            grad = tape.gradient(latent[:, j], x_train_tensor)
            contractive_penalty += tf.reduce_sum(tf.square(grad))

        loss = mse_loss + lambda_contractive * contractive_penalty

    grads = tape.gradient(loss, contractive_ae.trainable_variables)
    optimizer.apply_gradients(zip(grads, contractive_ae.trainable_variables))
    del tape  # free memory

    recon = contractive_ae(x_val_tensor).numpy()
    fname = f"contractive_epoch_{epoch+1}.png"
    plot_reconstruction(x_val, recon, f"Contractive AE - Epoch {epoch+1}", fname)
    frames_contractive.append(fname)

with imageio.get_writer("contractive_recon.gif", mode='I', duration=0.5) as writer:
    for fname in frames_contractive:
        img = imageio.v2.imread(fname)
        writer.append_data(img)

print("Contractive AE GIF saved as contractive_recon.gif")
