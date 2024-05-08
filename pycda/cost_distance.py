import geopandas
import numpy as np
import rasterio
import rioxarray
from shapely.geometry import Point 
from shapely.geometry import LineString 
import xarray

from pycda.cost_distance_grid import CostDistanceGrid
from pycda.cost_distance_directional import CostDistanceDirectional

class CostDistance:
    """
    A class for calculating cost distance using either a cost raster or a raster and a graph.
    This is a GIS extension for shortest path algorithms implemented on graphs/grids in sklearn and scipy.
    
    Parameters:
        raster (rioxarray.RasterArray): The raster array representing the cost surface.
        graph (scipy.sparse.csr_matrix, optional): The graph representing the cost of moving along the cost surface. Required when using the 'directional' method.
        method (str, optional): The method to use for cost distance calculation. Default is 'omnidirectional'
            - 'omnidirectional': Use a cost raster.
            - 'directional': Use a raster and a graph representing the cost of moving along the raster.
    
    Methods:
        trace_path: Find shortest path between two points
        cost_accumulation: Perform cost accumulation and basin delineation from input source(s)
    """

    def __init__(self, raster, graph=None, method="omnidirectional"):
        if method == "omnidirectional":
            self.cdg = CostDistanceGrid(raster.data)
            self.raster = raster
        elif method == "directional":
            self.cdg = CostDistanceDirectional(graph, raster.data.shape)
            self.raster = raster
        else:
            raise ValueError("Invalid method")

    def trace_path(self, source, target, return_type='vector'):
        # source and target are shapely Points
        source = self._point_to_rowcol(source)
        target = self._point_to_rowcol(target)
        path = self.cdg.trace_path(source, target)

        if return_type == 'vector':
            return self._path_to_linestring(path)
        elif return_type == 'raster':
            return self._path_to_raster(path)
        else:
            raise ValueError("Invalid return_type")


    def cost_accumulation(self, sources, return_basins=True):
        sources_list = self._validate_sources_input(sources)
        cumulative_costs, basins = self.cdg.cost_accumulation(sources_list)

        cumulative_costs = self._np_to_raster(cumulative_costs)

        if return_basins:
            basins = self._np_to_raster(basins)
            return cumulative_costs, basins
        else:
            return cumulative_costs

    def _validate_sources_input(self, sources):
        # can be raster, ndarray, geoseries, or shapely point
        if isinstance(sources, np.ndarray):
            if sources.shape != self.raster.shape:
                raise ValueError("Invalid input shape")
            sources_list = self._raster_to_rowcols(sources)
        elif isinstance(sources, xarray.core.dataarray.DataArray):
            if sources.shape != self.raster.shape:
                raise ValueError("Invalid input shape")
            if hasattr(sources, "rio"):
                if sources.rio.crs != self.raster.rio.crs:
                    raise ValueError("sources crs doesn't match")
                if sources.rio.bounds() != self.raster.rio.bounds():
                    raise ValueError("bounds don't match")
            sources_list = self._raster_to_rowcols(sources.data)
        elif isinstance(sources, geopandas.geoseries.GeoSeries):
            if sources.crs != self.raster.rio.crs:
                raise ValueError("Not the same crs as cost raster")
            sources_list = [self._point_to_rowcol(p) for p in sources]
        else:
            raise ValueError("invalid input")
        return sources_list

    def _np_to_raster(self, arr):
        base = self.raster.copy()
        base.data = arr
        return base

    def _point_to_rowcol(self, point):
        transform = rasterio.transform.AffineTransformer(self.raster.rio.transform())
        row,col = transform.rowcol(point.x, point.y)
        return (row,col)

    def _rowcol_to_point(self, row, col):
        transform = rasterio.transform.AffineTransformer(self.raster.rio.transform())
        x,y = transform.xy(row, col)
        return Point(x,y)

    def _raster_to_rowcols(self, boolean_array):
        return np.transpose((boolean_array).nonzero()).tolist()

    def _path_to_linestring(self, rowcols):
        points = [self._rowcol_to_point(row, col) for row, col in rowcols]
        return LineString(points)

    def _path_to_raster(self, rowcols):
        arr = np.zeros(self.raster.shape, dtype=bool)
        for x,y in rowcols:
            arr[x,y] = True
        ras = self.raster.copy()
        ras.data = arr
        return ras
