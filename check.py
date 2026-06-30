from build123d import *
from ocp_vscode import show

# Create a simple 3D block
box = Box(10, 20, 30)

# Push the shape to the OCP CAD viewer panel
show(box)
