# Lint as: python3

# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark class for Flax regression and integration testing."""

import itertools
from absl.testing import absltest

from tensorboard.backend.event_processing import directory_watcher
from tensorboard.backend.event_processing import event_file_loader


def _make_events_generator(path):
  """Makes a generator yielding TensorBoard events from files in `path`."""
  return directory_watcher.DirectoryWatcher(
      path,
      event_file_loader.EventFileLoader).Load()


def _process_event(event):
  """Parse TensorBoard scalars into a (tag, wall_time, step, scalar) tuple."""
  for value in event.summary.value:
    if value.HasField('simple_value'):
      yield (value.tag, event.wall_time, event.step, value.simple_value)

def _get_tensorboard_scalars(path):
  """Read and parse scalar TensorBoard summaries.

  Args:
    path: str. Path containing TensorBoard event files.

  Returns:
    Dictionary mapping summary tags (str) to lists of
    (wall_time, step, scalar) tuples.
  """
  gen = _make_events_generator(path)
  data = filter(lambda x: x.HasField('summary'), gen)
  data = itertools.chain.from_iterable(map(_process_event, data))

  data_by_key = {}
  for tag, wall_time, step, value in data:
    if not tag in data_by_key:
      data_by_key[tag] = []
    data_by_key[tag].append((wall_time, step, value))
  return data_by_key


class Benchmark(absltest.TestCase):
  """Benchmark class for Flax examples."""

  def read_summaries(self, path):
    """Read TensorBoard summaries."""
    return _get_tensorboard_scalars(path)
