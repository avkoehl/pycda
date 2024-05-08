from scipy.sparse.csgraph import dijkstra

from pycda.grid_utils import rowcol_to_id
from pycda.grid_utils import id_to_rowcol


class CostDistanceDirectional:

    def __init__(self, graph, shape):
        self.nrows, self.ncols = shape
        self.graph = graph

    def trace_path(self, source, target):
        source_id = rowcol_to_id(source[0], source[1], self.nrows, self.ncols)
        target_id = rowcol_to_id(target[0], target[1], self.nrows, self.ncols)

        _, predecessors, _ = dijkstra(
            csgraph=self.graph,
            directed=True,
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
        sources_ids = [
            rowcol_to_id(row, col, self.nrows, self.ncols) for row, col in sources
        ]
        cumulative_costs, _, sources_res = dijkstra(
            csgraph=self.graph,
            directed=True,
            indices=sources_ids,
            return_predecessors=True,
            min_only=True,
        )
        basins = sources_res.reshape((self.nrows, self.ncols))
        for i, outlet_id in enumerate(sources_ids):
            basin_id = basins[sources[i][0], sources[i][1]]
            basins[basins == basin_id] = outlet_id

        return cumulative_costs.reshape((self.nrows, self.ncols)), basins
