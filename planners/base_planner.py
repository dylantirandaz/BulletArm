import numpy as np
import numpy.random as npr

class BasePlanner(object):
  def __init__(self, env, config):
    self.env = env
    self.pos_noise = config['pos_noise'] if 'pos_noise' in config else None
    self.rot_noise = config['rot_noise'] if 'rot_noise' in config else None

  def getNextAction(self):
    raise NotImplemented('Planners must implement this function')

  def addNoiseToPos(self, x, y):
    # TODO: Would we ever want to include noise on the z-axis here?
    if self.pos_noise:
      x += npr.uniform(-self.pos_noise, self.pos_noise)
      y += npr.uniform(-self.pos_noise, self.pos_noise)
    return x, y

  def addNoiseToRot(self, rot):
    if self.rot_noise:
      rot += npr.uniform(-self.rot_noise, self.rot_noise)
    return rot
