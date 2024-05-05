import numpy as np
from scipy.sparse import csgraph
from skimage.graph import MCP_Geometric

from pycda.grid_utils import ids_from_grid
from pycda.grid_utils import neighbor_ids_to_sparse


class CostDistanceGrid:

    def __init__(self, cost_grid_array):
        self.cost_grid = cost_grid_array

    def trace_path(self, source, target):
        mcp = MCP_Geometric(self.cost_grid)
        _, _ = mcp.find_costs(starts=[source], ends=[target])
        path = mcp.traceback(target)
        return path

    def cost_accumulation(self, sources):
        mcp = MCP_Geometric(self.cost_grid)
        cumulative_costs, traceback = mcp.find_costs(starts=sources)
        basins = self._basins_from_traceback(traceback, mcp.offsets)
        basins = self._identify_basins(basins, traceback, sources)
        return cumulative_costs, basins

    def _traceback_to_neighbor_ids(self, traceback, offsets):
        offsets = np.append(
            offsets, [[0, 0]], axis=0
        )  # corresponds to the -1 index in traceback (stay in place)

        indices = np.indices(traceback.shape)
        offset_to_neighbor = offsets[traceback]

        neighbor_index = indices - offset_to_neighbor.transpose((2, 0, 1))
        ids = ids_from_grid(traceback)
        neighbor_ids = np.ravel_multi_index(tuple(neighbor_index), traceback.shape)
        return neighbor_ids, ids

    def _basins_from_traceback(self, traceback, offsets):
        """adapted from https://stackoverflow.com/a/62144556"""
        # if -2 in traceback -> -1
        # but keep track of these coordinates as they correspond to unreached places!
        unreached = np.where(traceback < -1)
        traceback[unreached] = -1

        neighbor_ids, ids = self._traceback_to_neighbor_ids(traceback, offsets)
        g = neighbor_ids_to_sparse(neighbor_ids, ids)

        _, components = csgraph.connected_components(g)
        basins = components.reshape(traceback.shape)

        # where unreached set to -2!
        basins[unreached] = -2
        return basins

    def _identify_basins(self, basins, traceback, start_cells):
        # given start cells and walls rename the basins to have the id of the start cell, otherwise -1 for wall and -2 for other
        ids = ids_from_grid(self.cost_grid)
        for outlet in enumerate(start_cells):
            basin_id = basins[outlet]
            start_cell_id = ids[outlet]
            basins[basins == basin_id] = start_cell_id
        basins[traceback == -2] = -2
        basins[self.cost_grid == -1] = -1
        return basins
