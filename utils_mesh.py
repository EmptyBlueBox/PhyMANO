import numpy as np


def compute_vertex_normals(vertices, faces):
    """
    Calculate vertex normals using vectorized operations.

    Parameters:
    - vertices (np.ndarray, shape=(N, 3)): Array of vertex coordinates.
    - faces (np.ndarray, shape=(M, 3)): Array of vertex indices for each face.

    Returns:
    - np.ndarray, shape=(N, 3): Array of normalized vertex normals.
    """
    # Get the vertices of the triangles
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute the normal vectors for each face
    face_normals = np.cross(v1 - v0, v2 - v0)

    # Initialize vertex normals to zeros
    vertex_normals = np.zeros_like(vertices)

    # Accumulate face normals to corresponding vertices
    for i in range(faces.shape[1]):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    # Normalize the vertex normals
    norm = np.linalg.norm(vertex_normals, axis=1)
    norm[norm == 0] = 1e-8  # Avoid division by zero
    vertex_normals /= norm[:, np.newaxis]

    return vertex_normals
