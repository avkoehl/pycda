from numba import njit
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

from pycda.grid_utils import rowcol_to_id
from pycda.grid_utils import id_to_rowcol


class CostDistanceDirectional:

    def __init__(self, dem, walls=None, enforce_uphill=False, enforce_downhill=False):
        self.dem = dem
        self.enforce_uphill = enforce_uphill
        self.enforce_downhill = enforce_downhill
        self.walls = walls
        self.nrows, self.ncols = dem.shape

        self.graph = None

    def trace_path(self, source, target):
        if self.graph is None:
            self._construct_graph()

        source_id = rowcol_to_id(source[0], source[1], self.nrows, self.ncols)
        target_id = rowcol_to_id(target[0], target[1], self.nrows, self.ncols)

        _, predecessors, _ = dijkstra(
            csgraph=self.graph,
            directed=False,
            indices=source_id,
            return_predecessors=True,
            min_only=True,
        )

        # unravel path
        path = []
        current = target_id
        while current != source_id:
            path.append(current)
            current = predecessors[current]
            if current == -9999:
                return None

        if len(path):
            path.append(source_id)
            path.reverse()
            path = [id_to_rowcol(p, self.nrows, self.ncols) for p in path]
            return path
        return None

    def cost_accumulation(self, sources):
        if self.graph is None:
            self._construct_graph()

        sources_ids = [
            rowcol_to_id(row, col, self.nrows, self.ncols) for row, col in sources
        ]
        cumulative_costs, _, sources_res = dijkstra(
            csgraph=self.graph,
            directed=False,
            indices=sources_ids,
            return_predecessors=True,
            min_only=True,
        )
        #basins = sources_res.reshape(self.dem.shape)
        #for i, outlet_id in enumerate(sources_ids):
        #    basin_id = basins[sources[i]]
        #    basins[basins == basin_id] = outlet_id

        #return cumulative_costs.reshape(self.dem.shape), basins
        return cumulative_costs, sources

    def _construct_graph(self):
        graphdata = self._create_graph_data_numba(
            self.dem, self.walls, self.enforce_uphill, self.enforce_downhill
        )
        graph = csr_matrix(graphdata)
        self.graph = graph

    @staticmethod
    @njit
    def _create_graph_data_numba(dem, walls, enforce_uphill, enforce_downhill):
        nrows, ncols = dem.shape
        ids = np.arange(dem.size).reshape(dem.shape)
        row_inds = []
        col_inds = []
        data = []
        for row in range(nrows):
            for col in range(ncols):
                if walls is not None:
                    if walls[row, col]:
                        continue

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = row + dx
                        ny = col + dy

                        if walls is not None:
                            if walls[nx, ny]:
                                continue

                        if 0 <= nx < nrows and 0 <= ny < ncols:
                            cost = 1 if dx == 0 or dy == 0 else 1.41
                            cost *= dem[nx, ny] - dem[row, col]

                            if cost < 0 and enforce_uphill:
                                continue

                            if cost > 0 and enforce_downhill:
                                continue

                            cost = np.abs(cost)
                            data.append(np.abs(cost))

                            start = ids[row, col]
                            end = ids[nx, ny]
                            row_inds.append(start)
                            col_inds.append(end)
        return (data, (row_inds, col_inds))
