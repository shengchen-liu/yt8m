# Copyright 2017 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 4,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_float(
    "moe_l2", 1e-8,
    "L2 penalty for MoeModel.")
flags.DEFINE_integer(
    "moe_low_rank_gating", -1,
    "Low rank gating for MoeModel.")
flags.DEFINE_bool(
    "moe_prob_gating", False,
    "Prob gating for MoeModel.")
flags.DEFINE_string(
    "moe_prob_gating_input", "prob",
    "input Prob gating for MoeModel.")

import sys
from os.path import dirname
if dirname(__file__) not in sys.path:
  sys.path.append(dirname(__file__))
from all_video_models import *


