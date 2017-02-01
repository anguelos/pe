#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Dependencies:
A UNIX like shell must be available
pip install (--user) numpy
pip install (--user) editdistance
pip install (--user) Polygon2
pip install (--user) Pillow (if cv2 is not available)

for information contact:
anguelos.nicolaou@gmail.com
"""
import rrcvisualisation
import rrcmetrics
import rrcio

rrcio=reload(rrcio)
rrcmetrics=reload(rrcmetrics)
rrcvisualisation=reload(rrcvisualisation)

from rrcio import *
from rrcmetrics import *
from rrcvisualisation import *
