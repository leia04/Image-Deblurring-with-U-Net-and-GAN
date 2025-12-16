#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from src.preprocessing.paired_dataset import load_image_pairs
from src.models.gan import generator_model, discriminator_model, perceptual_loss


def save_images(epoch, generator, dataset, batch_size, out_dir="artifacts/gan_samples", num_batches=1):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    batch_iter = iter(dataset)
    for batch_idx in range(num_batches):
        blur_images, sharp_images = next(batch_iter)
        gen_images = generator.predict(blur_images)

        num_images = min(batch_size, len(blur_images))
        fig, axs = plt.subplots(num_images, 3, figsize=(10, 4 * num_images))

        for i in range(num_images):
            axs[i, 0].imshow((blur_images[i] + 1) / 2.0, vmin=0, vmax=1)
            axs[i, 0].set_title("Blurred")
            axs[i, 0].axis("off")

            axs[i, 1].imshow((gen_images[i] + 1) / 2.0, vmin=0, vmax=1)
            axs[i, 1].set_title("Generated")
            axs[i, 1].axis("off")

            axs[i, 2].imshow((sharp_images[i] + 1) / 2.0, vmin=0, vmax=1)
            axs[i, 2].set_title("Sharp")
            axs[i, 2].axis("off")

        out_path = os.path.join(out_dir, f"epoch_{epoch+1}_batch_{batch_idx+1}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


@tf.function
def _train_step(g, d, blur_images, sharp_images, g_opt, d_opt, p_weight=0.01):
    # Discriminator update
    with tf.GradientTape() as tape:
        generated_images = g(blur_images, training=True)
        d_real = d(sharp_images, training=True)
        d_fake = d(generated_images, training=True)

        d_loss_real = tf.reduce_mean(d_real)
        d_loss_fake = tf.reduce_mean(d_fake)
        d_loss = d_loss_fake - d_loss_real  # WGAN-style

    d_grads = tape.gradient(d_loss, d.trainable_variables)
    d_opt.apply_gradients(zip(d_grads, d.trainable_variables))

    # Generator update
    with tf.GradientTape() as tape:
        generated_images = g(blur_images, training=True)
        d_fake = d(generated_images, training=True)
        g_loss_fake = -tf.reduce_mean(d_fake)

        p_loss = perceptual_loss(sharp_images, generated_images)
        g_loss = g_loss_fake + p_weight * p_loss

    g_grads = tape.gradient(g_loss, g.trainable_variables)
    g_opt.apply_gradients(zip(g_grads, g.trainable_variables))

    return d_loss, g_loss


def train_gan(
    train_root="data/GOPRO_Large/train",
    max_images=50,
    batch_size=8,
    epochs=3,
    out_path="artifacts/gan_generator.h5",
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # GAN uses [-1,1]
    ds = load_image_pairs(train_root, max_images=max_images, normalize="minus1_1").batch(batch_size)

    g = generator_model()
    d = discriminator_model()

    g_opt = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    d_opt = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for step, (blur_images, sharp_images) in enumerate(ds):
            d_loss, g_loss = _train_step(g, d, blur_images, sharp_images, g_opt, d_opt)
            if (step + 1) % 10 == 0:
                tf.print("Step", step + 1, "| D Loss:", d_loss, "| G Loss:", g_loss)

        save_images(epoch, g, ds, batch_size=batch_size, num_batches=1)

    g.save(out_path)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    train_gan()
