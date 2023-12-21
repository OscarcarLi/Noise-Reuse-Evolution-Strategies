# coding=utf-8
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for learned_optimizers.rnn_mlp_lopt."""
from absl.testing import absltest
from src.learned_optimizers import rnn_mlp_lopt
from src.learned_optimizers import test_utils


class RNNMLPLOptTest(absltest.TestCase):

  def test_rnn_mlp_lopt(self):
    test_utils.smoketest_learned_optimizer(rnn_mlp_lopt.RNNMLPLOpt())


if __name__ == '__main__':
  absltest.main()
