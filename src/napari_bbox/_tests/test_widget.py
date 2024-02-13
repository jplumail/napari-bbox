import time

import napari
import numpy as np
import pytest
from napari.utils.interactions import (
    mouse_move_callbacks,
    mouse_press_callbacks,
    mouse_release_callbacks,
)

from napari_bbox import BoundingBoxLayer

from ._test_utils import read_only_mouse_event


@pytest.fixture
def create_known_bbox_layer():
    data = [
        [
            [1, 1, 1],
            [10, 1, 1],
            [1, 1, 10],
            [10, 1, 10],
            [1, 10, 10],
            [10, 10, 10],
            [1, 10, 1],
            [10, 10, 1],
        ]
    ]
    known_non_bbox = [20, 20, 20]
    n_bboxs = len(data)

    layer = BoundingBoxLayer(data, ndim=3)
    assert layer.ndim == 3
    assert len(layer.data) == n_bboxs
    assert len(layer.selected_data) == 0

    return layer, n_bboxs, known_non_bbox


def test_add_simple_bbox(
    create_known_bbox_layer: tuple[BoundingBoxLayer, int, list[int]],
):
    """Add simple bbox by clicking in add mode."""
    layer, n_bboxs, known_non_bbox = create_known_bbox_layer

    # Add bbox at location where non exists
    layer.mode = "ADD_BOUNDING_BOX"

    # Simulate click
    event = read_only_mouse_event(
        type="mouse_press",
        position=known_non_bbox[1:],
    )
    mouse_press_callbacks(layer, event)

    known_non_bbox_end = [40, 60]
    # Simulate drag end
    event = read_only_mouse_event(
        type="mouse_move",
        is_dragging=True,
        position=known_non_bbox_end,
    )
    mouse_move_callbacks(layer, event)

    # Simulate release
    event = read_only_mouse_event(
        type="mouse_release",
        position=known_non_bbox_end,
    )
    mouse_release_callbacks(layer, event)

    # Check new bbox added at coordinates
    assert len(layer.data) == n_bboxs + 1
    np.testing.assert_allclose(layer.data[-1][0][1:], known_non_bbox[1:])
    new_bbox_max = np.max(layer.data[-1], axis=0)
    np.testing.assert_allclose(new_bbox_max[1:], known_non_bbox_end)


def test_select_bbox(make_napari_viewer_proxy, create_known_bbox_layer):
    """Select a bbox by clicking on one in select mode."""
    layer, n_bboxs, _ = create_known_bbox_layer
    viewer: napari.Viewer = make_napari_viewer_proxy(show=False)
    viewer.add_layer(layer)

    layer.mode = "select"
    position = layer.data[0][0]

    # Simulate click
    event = read_only_mouse_event(
        type="mouse_press",
        position=position,
    )
    mouse_press_callbacks(layer, event)

    # Simulate release
    event = read_only_mouse_event(
        type="mouse_release",
        position=position,
    )
    mouse_release_callbacks(layer, event)

    # Check clicked bbox selected
    assert len(layer.selected_data) == 1
    assert layer.selected_data == {0}


def test_remove_bbox(
    make_napari_viewer_proxy,
    create_known_bbox_layer: tuple[BoundingBoxLayer, int, list[int]],
):
    """Remove vertex from bbox."""
    layer, n_bboxs, known_non_bbox = create_known_bbox_layer
    viewer: napari.Viewer = make_napari_viewer_proxy(show=False)
    viewer.add_layer(layer)
    old_data = layer.data
    n_bbox = len(old_data)
    layer.mode = "select"

    # select bbox
    select = {0}
    layer.selected_data = select

    # remove selected bbox
    layer.remove_selected()
    assert len(layer.data) == n_bbox - 1

    # wait 1 s to avoid the cursor event throttling
    time.sleep(1)
    # move mouse over canvas
    viewer.mouse_over_canvas = True
    viewer.cursor.position = [1, 1, 1]
    assert viewer.status == viewer.layers.selection.active.get_status(
        viewer.cursor.position, world=True
    )
