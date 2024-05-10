import numpy as np
from scipy.sparse import coo_matrix


def ids_from_grid(grid):
    return np.arange(grid.size).reshape(grid.shape)


def ids_from_shape(nrows, ncols):
    return np.arange(nrows * ncols).reshape((nrows, ncols))


def neighbor_ids_to_sparse(neighbor_ids, ids):
    g = coo_matrix(
        (np.ones(neighbor_ids.size), (ids.flat, neighbor_ids.flat)),
        shape=(ids.size, ids.size),
    ).tocsr()
    return g


def id_to_rowcol(idn, nrows, ncols):
    row_inds, col_inds = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
    row = row_inds.flatten()[idn]
    col = col_inds.flatten()[idn]
    return (row, col)


def rowcol_to_id(row, col, nrows, ncols):
    ids = ids_from_shape(nrows, ncols)
    return ids[row, col]
