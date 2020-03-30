"""Benchmark for the ImageNet example."""
import tempfile

from absl import flags
from absl.testing import absltest
from absl.testing.flagsaver import flagsaver

from flax.testing import Benchmark

import train

import jax
import numpy as np


# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


FLAGS = flags.FLAGS


class ImagenetBenchmark(Benchmark):
  """Benchmarks for the ImageNet Flax example."""

  @flagsaver
  def test_8x_v100_half_precision(self):
    """Run ImageNet on 8x V100 GPUs in half precision for 2 epochs."""
    model_dir = tempfile.mkdtemp()
    FLAGS.batch_size = 2048
    FLAGS.half_precision = True
    FLAGS.loss_scaling = 256.
    FLAGS.num_epochs = 2
    FLAGS.model_dir = model_dir

    train.main([])
    summaries = self.read_summaries(model_dir)

    # Summaries contain all the information necessary for the regression
    # metrics.
    wall_time, _, eval_accuracy = zip(*summaries['eval_accuracy'])
    wall_time = np.array(wall_time)
    sec_per_epoch = np.mean(wall_time[1:] - wall_time[:-1])
    end_accuracy = eval_accuracy[-1]

    print('sec/epoch', sec_per_epoch)
    print('end_accuracy', end_accuracy)

if __name__ == '__main__':
  absltest.main()
