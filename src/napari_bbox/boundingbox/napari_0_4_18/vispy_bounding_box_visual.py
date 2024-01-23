# A copy of napari._vispy.visuals.shapes
from vispy.scene.visuals import Compound, Line, Markers, Mesh, Text

from napari._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin


class BoundingBoxVisual(ClippingPlanesMixin, Compound):
    """
    Compound vispy visual for shapes visualization with
    clipping planes functionality

    Components:
        - Mesh for shape faces (vispy.MeshVisual)
        - Mesh for highlights (vispy.MeshVisual)
        - Lines for highlights (vispy.LineVisual)
        - Vertices for highlights (vispy.MarkersVisual)
        - Text labels (vispy.TextVisual)
    """

    def __init__(self) -> None:
        super().__init__([Mesh(), Mesh(), Line(), Markers(), Text()])
