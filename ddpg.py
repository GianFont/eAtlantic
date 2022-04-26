"""
ddpg.py
"""
__author__ = "gianluca.fontanesi@ucdconnect.ie"

import numpy as np
import tensorflow as tf
import sys
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
