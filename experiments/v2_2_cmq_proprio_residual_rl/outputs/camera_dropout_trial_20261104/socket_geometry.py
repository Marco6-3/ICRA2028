"""Socket-frame geometry for the 00186 asset; no initial-pose error subtraction."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np


def load_plug_tip(asset_root, asset_id, object_scale):
    """Use the +Z distal face center of the actual spawned URDF visual mesh.

    The 00186 mesh is centered on the local Z axis. This is a declared geometric
    endpoint, not an inferred insertion/contact success label.
    """
    urdf = Path(asset_root) / "automate_scaled" / "urdf" / f"{asset_id}_plug.urdf"
    visual = ET.parse(urdf).find(".//visual")
    if visual.find("origin") is not None:
        raise ValueError("Nonidentity mesh origins need an explicit transform")
    mesh = visual.find("geometry/mesh")
    mesh_path = (urdf.parent / mesh.attrib["filename"]).resolve()
    scale = np.fromstring(mesh.attrib.get("scale", "1 1 1"), sep=" ") * object_scale
    vertices = np.array([[float(v) for v in line.split()[1:4]]
                         for line in mesh_path.read_text().splitlines() if line.startswith("v ")]) * scale
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    tip = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), hi[2]])
    return tip, {"asset_id": asset_id, "tip_definition": "+Z mesh distal face center",
                 "plug_tip_offset_b_m": tip.tolist(), "mesh_bounds_m": [lo.tolist(), hi.tolist()],
                 "plug_urdf": str(urdf), "plug_mesh": str(mesh_path),
                 "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
                 "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest()}


def socket_alignment(plug_position, plug_rotation, socket_position, socket_rotation, tip_offset):
    rel_pos = socket_rotation.T @ (plug_position - socket_position)
    rel_rot = socket_rotation.T @ plug_rotation
    tip = rel_pos + rel_rot @ tip_offset
    return {
        "radial_error_mm": float(np.linalg.norm(rel_pos[:2]) * 1000.0),
        "axis_tilt_deg": float(np.degrees(np.arccos(np.clip(abs(rel_rot[2, 2]), 0.0, 1.0)))),
        "tip_lateral_error_mm": float(np.linalg.norm(tip[:2]) * 1000.0),
        "tip_axial_position_mm": float(tip[2] * 1000.0),
    }
