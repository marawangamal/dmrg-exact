"""Simple example on how to log scalars and images to tensorboard without tensor ops.
License: Copyleft
"""

__author__ = "Michael Gygli"

import tensorflow as tf

from io import StringIO
import matplotlib.pyplot as plt
import numpy as np


class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)

    def log_images(self, tag, images, step):
        """Logs a list of images."""

        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            plt.imsave(s, img, format="png")

            # Create an Image object
            img_sum = tf.compat.v1.Summary.Image(
                encoded_image_string=s.getvalue(),
                height=img.shape[0],
                width=img.shape[1],
            )
            # Create a Summary value
            im_summaries.append(
                tf.compat.v1.Summary.Value(tag="%s/%d" % (tag, nr), image=img_sum)
            )

        # Create and write Summary
        with self.writer.as_default():
            for nr, img in enumerate(images):
                # Convert image to tensor
                img_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
                if len(img_tensor.shape) == 2:
                    img_tensor = tf.expand_dims(img_tensor, -1)
                if img_tensor.shape[-1] == 1:
                    img_tensor = tf.tile(img_tensor, [1, 1, 3])

                # Add batch dimension
                img_tensor = tf.expand_dims(img_tensor, 0)

                tf.summary.image(f"{tag}/{nr}", img_tensor, step=step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.compat.v1.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        with self.writer.as_default():
            tf.summary.histogram(tag, values, step=step)
