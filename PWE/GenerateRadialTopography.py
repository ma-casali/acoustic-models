import numpy as np
import matplotlib.pyplot as plt
import scipy
import rasterio 
import sys
import os

sys.path.append(os.path.abspath('../acoustic-models'))

class TopographyRadials():

    def __init__(self, 
                 topo_file: str = None,
                 center_coords: np.ndarray = None):

        self.topo_file = topo_file
        self.center_coords = center_coords

        with rasterio.open(self.topo_file) as src:
            # topo_data = src.read(1)
            self.box_bounds = src.bounds
            
            self.lat = np.linspace(self.box_bounds.bottom, self.box_bounds.top, src.shape[0])
            self.lon = np.linspace(self.box_bounds.left, self.box_bounds.right, src.shape[1])
            self.LON, self.LAT = np.meshgrid(self.lon, self.lat)

            # convert to m, and +depth in direction of -z
            self.topo_data = (-src.read(1) * 0.3048)[::-1, :] # shape (lat, lon)
            
        # lat, lon in degrees, depth in ft. 
        self.sample_func = scipy.interpolate.RegularGridInterpolator((self.lat, self.lon), self.topo_data)
        self.center_depth = self.sample_func(center_coords)

    def generate_radials(self, 
                         bearings: np.ndarray = None, 
                         ranges: np.ndarray = None,
                         r_max: float = None,
                         dr: float = None):
        
        if ranges is None:
            ranges = np.arange(0, r_max, dr)
        if r_max is None and dr is None:
            r_max = ranges[-1]
            dr = np.diff(ranges)[0]

        self.topo_radials = np.zeros((len(bearings), len(ranges)))
        
        for i, th in enumerate(bearings):
            for j, r in enumerate(ranges):
                new_lat, new_lon = self.get_destination_point(r, th)

                is_inside = (self.box_bounds.left <= new_lon <= self.box_bounds.right) and \
                (self.box_bounds.bottom <= new_lat <= self.box_bounds.top)
                if is_inside:
                    self.topo_radials[i, j] = self.sample_func([new_lat, new_lon])
                else:
                    self.topo_radials[i, j] = np.nan

        if ranges is None:
            return self.topo_radials, ranges
        else:
            return self.topo_radials

    def get_destination_point(self, r, bearing_deg):
        R = 6371000  # Earth radius in meters
        
        lat1 = np.radians(self.center_coords[0])
        lon1 = np.radians(self.center_coords[1])
        bearing = np.radians(bearing_deg)
        
        angular_distance = r / R
        
        lat2 = np.arcsin(np.sin(lat1) * np.cos(angular_distance) + 
                        np.cos(lat1) * np.sin(angular_distance) * np.cos(bearing))
        
        lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(angular_distance) * np.cos(lat1),
                                np.cos(angular_distance) - np.sin(lat1) * np.sin(lat2))
        
        return np.degrees(lat2), np.degrees(lon2)


if __name__ == '__main__':

    topo_file = 'PWE/Data/Topography/MontereyBay.tiff'
    mars_coords = np.array([36 + 42.7481/60, -122 - 11.2139/60])
    topo_radials = TopographyRadials(center_coords=mars_coords, topo_file=topo_file)

    bearings = np.array([0])
    r = np.arange(0, 25e3, 10)
    topography = topo_radials.generate_radials(bearings, ranges=r)

    plt.plot(r, topography[0,:])
    plt.show()
