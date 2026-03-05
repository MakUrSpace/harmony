import os
import sys

oldPath = os.getcwd()
try:
    mechaCombatDir = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, mechaCombatDir)
    from harmony.MechaCombatSystem import *
finally:
    os.chdir(oldPath)
