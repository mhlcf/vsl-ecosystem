"""Headless tests for the keypoint-driven 3D avatar renderer."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")   # no GUI needed

import numpy as np
import pytest

from p2avatar.renderer import KeypointAvatar, split_frame, transform

N_FRAMES = 10
DIM = 201


@pytest.fixture(scope="module")
def clip() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform(0.0, 1.0, size=(N_FRAMES, DIM)).astype(np.float32)


def test_split_frame_shapes(clip):
    pose, left, right = split_frame(clip)
    assert pose.shape == (N_FRAMES, 75)
    assert left.shape == (N_FRAMES, 63)
    assert right.shape == (N_FRAMES, 63)


def test_split_frame_single_vector(clip):
    pose, left, right = split_frame(clip[-1])
    assert pose.shape == (1, 75)
    assert left.shape == (1, 63)


def test_transform_flips_y(clip):
    t = transform(clip[-1])
    assert t.shape == (67, 3)        # 25 pose + 21 + 21 hand joints
    expected_y = 1.0 - clip[-1].reshape(67, 3)[:, 1]
    assert np.allclose(t[:, 1], expected_y)


def test_keypoint_avatar_render_smoke(clip):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5, 5), dpi=80)
    avatar = KeypointAvatar(fig)
    avatar.rebuild(clip[0])
    avatar.draw_frame(clip[3])        # in-place update must not raise
    avatar.close()


def test_keypoint_avatar_edges_present(clip):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(5, 5), dpi=80)
    avatar = KeypointAvatar(fig)
    avatar.rebuild(clip[0])
    assert len(avatar._hand_meshes) == 2
    assert all(m for m in avatar._hand_meshes)
    avatar.close()


def test_keypoint_avatar_visibility_sizes(clip):
    """Joints/bones must be chunky enough to watch (regression: tiny dots)."""
    from matplotlib.figure import Figure

    from p2avatar import renderer as R

    fig = Figure(figsize=(5, 5), dpi=80)
    avatar = KeypointAvatar(fig)
    avatar.rebuild(clip[0])

    assert np.all(np.asarray(avatar._pose_mesh.get_linewidth())
                  == R._POSE_LINEWIDTH >= 3)
    assert set(avatar._pose_joints.get_sizes()) == {R._POSE_MARKERSIZE ** 2}
    assert len(avatar._pose_joints.get_offsets()) == len(R._BODY_IDX)
    for mesh, joints_mid, joints_tip in avatar._hand_meshes:
        assert np.all(np.asarray(mesh.get_linewidth())
                      == R._HAND_LINEWIDTH >= 3.5)
        assert set(joints_mid.get_sizes()) == {R._HAND_JOINT_SIZE}
        assert set(joints_tip.get_sizes()) == {R._HAND_TIP_SIZE}
        assert R._HAND_TIP_SIZE > R._HAND_JOINT_SIZE >= 100
        assert (len(joints_mid.get_offsets()),
                len(joints_tip.get_offsets())) == (15, len(R._HAND_TIP_IDX))
    avatar.close()


def test_keypoint_avatar_face_declustered(clip):
    """Face points must be small/faded + ringed, not a blob with body joints."""
    from matplotlib.figure import Figure

    from p2avatar import renderer as R

    fig = Figure(figsize=(5, 5), dpi=80)
    avatar = KeypointAvatar(fig)
    avatar.rebuild(clip[0])
    assert set(avatar._face_joints.get_sizes()) == {R._FACE_JOINT_SIZE}
    assert len(avatar._face_joints.get_offsets()) == len(R._FACE_IDX)
    assert R._FACE_JOINT_SIZE < R._POSE_MARKERSIZE ** 2
    assert avatar._head_ring is not None
    avatar.draw_frame(clip[3])    # ring + face update must not raise
    assert avatar._head_ring is not None
    avatar.close()


def test_keypoint_avatar_uses_skeleton_edges(clip):
    """Bones must follow POSE_EDGES/HAND_EDGES, not index-order spaghetti."""
    import io

    from matplotlib.figure import Figure

    from p2avatar import renderer as R

    fig = Figure(figsize=(5, 5), dpi=80)
    avatar = KeypointAvatar(fig)
    avatar.rebuild(clip[0])
    # NOTE: 3D projection (and path building) only happens on a real draw;
    # a bare Figure's default canvas is a no-op, so savefig to force it.
    fig.savefig(io.BytesIO(), format="png")
    assert len(avatar._pose_mesh.get_paths()) == len(R.POSE_EDGES_25)
    for mesh, _mid, _tip in avatar._hand_meshes:
        assert len(mesh.get_paths()) == len(R.HAND_EDGES)
    before = [p.vertices.copy() for p in avatar._pose_mesh.get_paths()]
    avatar.draw_frame(clip[3])    # in-place segment update must refresh bones
    fig.savefig(io.BytesIO(), format="png")
    after = [p.vertices for p in avatar._pose_mesh.get_paths()]
    assert any(not np.array_equal(b, a) for b, a in zip(before, after))
    avatar.close()


def test_keypoint_avatar_default_camera(clip):
    """Default view must be the face-to-face camera, not from below."""
    from matplotlib.figure import Figure

    from p2avatar import renderer as R

    fig = Figure(figsize=(5, 5), dpi=80)
    avatar = KeypointAvatar(fig)
    assert (avatar.ax.elev, avatar.ax.azim) == (R._CAM_ELEV, R._CAM_AZIM)
    avatar.rebuild(clip[0])     # rebuild must restore the same camera
    assert (avatar.ax.elev, avatar.ax.azim) == (R._CAM_ELEV, R._CAM_AZIM)
    avatar.close()