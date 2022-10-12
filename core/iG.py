#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

#############################

# Integrated Gradiants Module refactored from https://www.tensorflow.org/tutorials/interpretability/integrated_gradients

############################



def interpolate_images(baseline,
                       image,
                       alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x +  alphas_x * delta
    return images


def compute_gradients(images, target_class_idx,model):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)


def integral_approximation(gradients):
    # riemann_trapezoidal
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients


def integrated_gradients(baseline,
                         image,
                         target_class_idx,
                         model,
                         m_steps=50,
                         batch_size=32):
    # Generate alphas.
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

    # Collect gradients.    
    gradient_batches = []

    # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(from_ + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        gradient_batch = one_batch(baseline, image, alpha_batch, target_class_idx,model)
        gradient_batches.append(gradient_batch)

    # Concatenate path gradients together row-wise into single tensor.
    total_gradients = tf.concat(gradient_batches, axis=0)

    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # Scale integrated gradients with respect to input.
    integrated_gradients = (image - baseline) * avg_gradients

    return integrated_gradients

@tf.function
def one_batch(baseline, image, alpha_batch, target_class_idx,model):
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                        image=image,
                                                        alphas=alpha_batch)

    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                        target_class_idx=target_class_idx,model=model)
    return gradient_batch




def plot_img_attributions(attributions,baseline,
                          image,
                          cmap=None,
                          overlay_alpha=0.4,figsize=(8,8)):


    # Sum of the attributions across color channels for visualization.
    # The attribution mask shape is a grayscale image with height and width
    # equal to the original image.
    attribution_mask = tf.reduce_sum(tf.math.abs(attributions), axis=-1)

    fig, axs = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=figsize)
    # matplotlib demands a value range between [0,1]
    # if image.max() > 1:
    #     image=image/255.0
    image=tf.cast(image, tf.uint8)

    axs[0,0].set_title('Original image')
    axs[0,0].imshow(image)
    axs[0,0].axis('off')

    axs[0,1].set_title('Attribution mask')
    axs[0,1].imshow(attribution_mask, cmap=cmap)
    axs[0,1].axis('off')

    axs[0,2].set_title('Overlay')
    axs[0,2].imshow(attribution_mask, cmap=cmap)
    axs[0,2].imshow(image, alpha=overlay_alpha)
    axs[0,2].axis('off')

    plt.tight_layout()
    return fig

def path_gradiant(baseline,model,floatIm,alphas,targetLabel_idx):
    interpolated_images = interpolate_images(
        baseline=baseline,
        image=floatIm,
        alphas=alphas)

    path_gradients = compute_gradients(
        images=interpolated_images,
        target_class_idx=targetLabel_idx,model=model)
    
    pred = model.predict(interpolated_images)
    pred_proba = pred[:, targetLabel_idx]
    return path_gradients,pred_proba

def plot_pathGrad(alphas,path_gradients,pred_proba, figsize=(10,4)):
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(alphas, pred_proba)
    ax1.set_title('Target class predicted probability over alpha')
    ax1.set_ylabel('model p(target class)')
    ax1.set_xlabel('alpha')
    ax1.set_ylim([0, 1])

    ax2 = plt.subplot(1, 2, 2)
    # Average across interpolation steps
    average_grads = tf.reduce_mean(path_gradients, axis=[1, 2, 3])
    # Normalize gradients to 0 to 1 scale. E.g. (x - min(x))/(max(x)-min(x))
    average_grads_norm = (average_grads-tf.math.reduce_min(average_grads))/(tf.math.reduce_max(average_grads)-tf.reduce_min(average_grads))
    ax2.plot(alphas, average_grads_norm)
    ax2.set_title('Average pixel gradients (normalized) over alpha')
    ax2.set_ylabel('Average pixel gradients')
    ax2.set_xlabel('alpha')
    ax2.set_ylim([0, 1])
