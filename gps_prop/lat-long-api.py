import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
from sklearn.linear_model import LinearRegression
from celest.satellite import Satellite, Coordinate
from celest.encounter import GroundPosition


class GPSPropogation:


    def __init__(self, time: np.ndarray, eci_positions: np.ndarray,
                 target_coordinates: tuple, linear_fit: tuple):
        """
        Parameters
        ----------
            eci_positions: np.ndarray
                orbit data
            target_coordinates: tuple 
                latitude and langitude coordinates of target location
            linear_fit: tuple
                index of orbit data and maximum error for linear
                approximation of ground track
        """

        self.time = time
        self.eci_positions = eci_positions
        self.target_coordinates = target_coordinates
        self.linear_fit = linear_fit

        # data to be computed
        self.data = None
        self.latitude = None
        self.longitude = None
        self.hull = None


    def load_data_celest(self, eci_positions: np.ndarray) \
            -> "tuple(np.ndarray, np.ndarray, np.ndarray)":
        """Returns an array of the orbit data [latitude, longitude, altitude], 
        an array of the latitude data, and an array of the longitude data

        Parameters
        ----------
            eci_positions: np.ndarray
                orbit data

        Returns
        -------
            data: np.ndarray
                orbit data [latitude, longitude, altitude]
            latitude: np.ndarray
                latitude data
            longitude: np.ndarray
                longitude data
        """

        orbit_data = np.loadtxt(eci_positions, delimiter=',')

        time_seconds = orbit_data[:, 0]  # julian
        position_m = orbit_data[:, 1:4]  # gcrs
        time_julian = time_seconds / 86400

        satellite = Satellite(position=position_m, frame='gcrs',
                              julian=time_julian, offset=0)

        latitude, longitude, altitude = satellite.geo()

        data = np.array([latitude, longitude, altitude])
        data = np.transpose(data)

        return data, latitude, longitude


    def set_parameters(self, start: int, end: int, data: np.ndarray) \
            -> "tuple(np.ndarray, np.ndarray)":
        """Returns truncated latitude and longitude arrays from index = start to index = end

        Parameters
        ----------
            start: int
                start index
            end: int 
                end index
            data: np.ndarray 
                orbit data [latitude, longitude, altitude]

        Returns
        -------
            latitude: np.ndarray
                latitude data
            longitude: np.ndarray
                longitude data
        """

        latitude = data[start:end, 0]  # Degrees north of equator
        longitude = data[start:end, 1]  # Degrees east of prime meridian
        return latitude, longitude


    def get_convex_hull(self, target_coordinates: "tuple(float, float)") -> ConvexHull:
        """Returns convex hull around target coordinates

        Parameters
        ---------- 
            traget_coordinate: tuple
                target latitude and longitude coordinates

        Returns
        -------
            hull
                convex hull
        """

        target = GroundPosition(
            target_coordinates[0], target_coordinates[1])

        # Generate an evenly spaced 2D grid of latitudes and longitudes
        lats = np.linspace(-90, 90, 1000)
        lons = np.linspace(-180, 180, 1000)

        LAT, LON = np.meshgrid(lats, lons)

        # reshape the arrays into 1D arrays
        LAT = LAT.reshape(-1)
        LON = LON.reshape(-1)
        ALT = 550 * np.ones_like(LAT)

        LLA = np.vstack((LAT, LON, ALT)).T

        # make up fictitious times
        times = np.linspace(0, 1, len(LLA))
        coord = Coordinate(position=LLA, frame="geo", julian=times)

        # Initialize ITRS data
        coord.itrs()

        off_nad = coord.off_nadir(target)
        alt, _ = coord.horizontal(target)  # alt is the elevation angle

        # we want an off-nadir angle less than 30 degrees, can try 60
        valid_latitudes = LAT[(off_nad < 60) & (alt > 0)]
        valid_longitudes = LON[(off_nad < 60) & (alt > 0)]

        points = np.vstack((valid_longitudes, valid_latitudes)).T

        return ConvexHull(points)


    def plot_convex_hull_onto_axes(self, hull: ConvexHull, ax: plt.axes):
        # Plot the convex hull
        points = hull.points

        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'k-')

        ax.plot(points[hull.vertices, 0],
                points[hull.vertices, 1], 'r--', lw=2)
        ax.plot(points[hull.vertices[0], 0],
                points[hull.vertices[0], 1], 'r--')


    def inside_hull(self, hull: ConvexHull, latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
        """Returns an array of latitude and longitude points from 
        the orbit data that lies inside the convex hull

        Parameters
        ----------
            hull: 
                convex hull
            latitude: np.ndarray
                latitude data
            longitude: np.ndarray
                longitude data

        Returns
        -------
            np.ndarray
                array of latitude and longitude points that lie inside the convex hull
        """
        points = np.vstack((longitude, latitude)).T

        # This solution was patched together from
        # https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
        hull = Delaunay(hull.points[hull.vertices])

        # Generate a Delauney triangulation of the points in the hull,
        # Then determine which points are inside the hull

        return hull.find_simplex(points) >= 0


    def get_jumps(self, arr: np.ndarray, threshold: int = 10) -> np.ndarray:
        """Returns an array of the indices of the jumps

        Parameters
        ----------
        arr: np.ndarray
            latitude or longitude data
        Returns
        -------
        jumps_arr: np.ndarray
            array of the indices of the jumps
        """
        diff_arr = np.diff(arr)
        jumps_arr = np.where(np.abs(diff_arr) > threshold)
        jumps_arr = np.insert(jumps_arr, 0, -1)
        jumps_arr = np.append(jumps_arr, len(arr))

        return jumps_arr


    def plot_ground_track(self, data: np.ndarray, longitude: np.ndarray, latitude: np.ndarray, hull: ConvexHull, ax: plt.axes) -> None:
        """Plots the ground track, linear approximation of the ground track at a point, and the error of the linear fit

        Parameters
        ----------
            data: np.ndarray
                orbit data
            longitude: np.ndarray
                longitude data
            latitude: np.ndarray
                latitude data
            hull
                convex hull
            ax: plt.axes    
                axes object
        """
        jumps = self.get_jumps(longitude, threshold=50)

        for i in range(len(jumps)-1):
            lat, long = self.set_parameters(jumps[i]+1, jumps[i+1], data)
            inside_ellipse = self.inside_hull(hull, lat, long)

            # plot green
            new_inside_ellipse = np.copy(inside_ellipse)
            updated = False
            for i in range(len(inside_ellipse)-1):
                if inside_ellipse[i] == False and inside_ellipse[i+1] == True:
                    new_inside_ellipse[i] = True
                elif inside_ellipse[i] == True and inside_ellipse[i+1] == False:
                    if updated == False:
                        new_inside_ellipse[i+1] = True
                        updated = True
                    else:
                        updated = False

            lat_inside = lat[new_inside_ellipse]
            long_inside = long[new_inside_ellipse]
            data_new = np.array([np.array(lat_inside), np.array(long_inside)])
            jumps_arr = self.get_jumps(lat_inside)

            for i in range(len(jumps_arr)-1):
                lat_green, long_green = self.set_parameters(
                    jumps_arr[i]+1, jumps_arr[i+1], data_new.T)
                ax.plot(long_green, lat_green, color='tab:green')

            # plot blue
            outside_ellipse = ~inside_ellipse
            new_outside_ellipse = np.copy(outside_ellipse)
            updated = False
            for i in range(len(outside_ellipse)-1):
                if outside_ellipse[i] == False and outside_ellipse[i+1] == True:
                    pass
                elif outside_ellipse[i] == True and outside_ellipse[i+1] == False:
                    if updated == False:
                        new_outside_ellipse[i+1] = True
                        updated = True
                    else:
                        updated = False

            lat_outside = lat[new_outside_ellipse]
            long_outside = long[new_outside_ellipse]
            data_new = np.array(
                [np.array(lat_outside), np.array(long_outside)])
            jumps_arr = self.get_jumps(lat_outside)

            for i in range(len(jumps_arr)-1):
                lat_blue, long_blue = self.set_parameters(
                    jumps_arr[i]+1, jumps_arr[i+1], data_new.T)
                ax.plot(long_blue, lat_blue, color='tab:blue')


    def get_points_for_linear_fit(self, index: int, longitude: np.ndarray,
                        latitude: np.ndarray) -> "tuple(np.ndarray, np.ndarray)":
        """Returns arrays of x and y orbit data points for a linear fit

        Parameters
        ----------
            index: int
                index of the point where a linear fit should be completed
            longitude: np.ndarray
                longitude data
            latitude: np.ndarray
                latitude data

        Returns
        -------
            x, y: np.ndarray
                array of x and y orbit data points for a linear fit
        """
        x = []
        y = []
        for i in range(index-5, index+6):
            x.append(longitude[i])
            y.append(latitude[i])
        x = np.array(x).reshape((-1, 1))
        y = np.array(y)
        
        return x, y


    def get_linear_fit(self, index: int, longitude: np.ndarray,
                  latitude: np.ndarray, acceptable_err: int) -> "tuple(float, float)":
        """Calculates the linear approximation of the ground track at a point

        Parameters
        ----------
            hull
                convex hull
            index: int
                index of the point where a linear fit should be completed
            longitude: np.ndarray
                longitude data
            latitude: np.ndarray
                latitude data
            acceptable_err: int
                maximum allowable absolute error of the linear fit

        Returns
        -------
            m: int
                slope of linear fit
            b: int
                y-intercept of linear fit
            acceptable_x_arr: np.ndarray
                array of x values of the linear approximation that are within the error
            acceptable_y_arr: np.ndarray
                array of y values of the linear approximation that are within the error
        """
        x, y = self.get_points_for_linear_fit(index, longitude, latitude)
        model = LinearRegression().fit(x, y)
        m = model.coef_
        b = model.intercept_

        _, abs_err, ind_array = self.get_linear_fit_error(
            index, longitude, latitude, m, b)
        
        self.plot_linear_fit_error(index, longitude, latitude, m, b)

        # figure out indices to plot
        acceptable_range = abs_err <= acceptable_err
        acceptable_ind = ind_array[acceptable_range]
        line_start = int(acceptable_ind[0])
        line_end = int(acceptable_ind[-1])

        # y = mx + b, x is longitude, y is latitude
        long_start = longitude[line_start]
        long_end = longitude[line_end]
        if long_start > long_end:
            long_start, long_end = long_end, long_start

        x_arr = np.linspace(-180, 180, 500)
        acceptable_x1 = x_arr >= long_start
        acceptable_x2 = x_arr <= long_end
        acceptable_x = acceptable_x1 & acceptable_x2
        acceptable_x_arr = x_arr[acceptable_x]
        acceptable_y_arr = m * acceptable_x_arr + b

        return m, b, acceptable_x_arr, acceptable_y_arr
    

    def plot_linear_fit_onto_axes(self, hull, index: int, longitude: np.ndarray,
                  latitude: np.ndarray, ax: plt.axes, acceptable_err: int) -> "tuple(float, float)":
        """Plots the linear approximation of the ground track at a point

        Parameters
        ----------
            hull
                convex hull
            index: int
                index of the point where a linear fit should be completed
            longitude: np.ndarray
                longitude data
            latitude: np.ndarray
                latitude data
            acceptable_err: int
                maximum allowable absolute error of the linear fit
        """
        _, _, acceptable_x_arr, acceptable_y_arr = self.get_linear_fit(
            index, longitude, latitude, acceptable_err)

        ax.plot(acceptable_x_arr, acceptable_y_arr, color='tab:orange')

        # ax.axline((0, b), slope = m, color='tab:orange')
        ax.set_ylim(-120, 120)

        # plot the inside ellipse region a different colour
        self.plot_line_inside_hull_on_axes(
            hull, acceptable_x_arr, acceptable_y_arr, ax)


    def get_line_points_inside_hull(self, hull, x_arr: np.ndarray, y_arr: np.ndarray) -> "tuple(np.ndarray, np.ndarray)":
        """Returns arrays of x and y points from the linear fit that are inside the convex hull

        Parameters
        ----------
            hull
                convex hull
            x_arr: np.nadarray
                array of x points of the linear fit
            y_arr: np.nadarray
                array of y points of the linear fit

        Returns
        -------
            plot_x, plot_y: np.ndarray
                arrays of x and y points from the linear fit that are inside the convex hull
        """
        inside = self.inside_hull(hull, y_arr, x_arr)
        plot_x = x_arr[inside]
        plot_y = y_arr[inside]

        return plot_x, plot_y


    def plot_line_inside_hull_on_axes(self, hull, x_arr: np.ndarray, y_arr: np.ndarray, ax: plt.axes) -> "tuple(np.ndarray, np.ndarray)":
        """Plots arrays of x and y points from the linear fit that are inside the convex hull onto the ground track plot in a different colour

        Parameters
        ----------
            hull
                convex hull
            x_arr: np.nadarray
                array of x points of the linear fit
            y_arr: np.nadarray
                array of y points of the linear fit
        """
        plot_x, plot_y = self.get_line_points_inside_hull(hull, x_arr, y_arr)

        ax.plot(plot_x, plot_y)


    def get_linear_fit_error(self, index: int, longitude: np.ndarray,
                          latitude: np.ndarray, m: float, b: float) \
            -> "tuple(np.ndarray, np.ndarray)":
        """Returns arrays of the absolute error of the linear fit and the indices of the orbit data where the error was calculated
            Plots the error of the linear fit

        Parameters
        ----------
            index: int
                index of the point where a linear fit should be completed
            longitude: np.ndarray
                longitude data
            latitude: np.ndarray
                latitude data
            m: int
                slope of linear fit
            b: int
                y-intercept of linear fit

        Returns
        -------
            long, abs_err, ind: np.ndarray
                arrays of the longitude points, absolute error of the linear fit, and the indices of the orbit data where the error was calculated
        """
        range = 25  # arbitrarily chosen
        ind = np.linspace(index-range, index+range, range*2+1)
        long = longitude[index-range:index+range+1]
        actual = latitude[index-range:index+range+1]
        pred = long * m + b
        abs_err = abs(actual-pred)

        return long, abs_err, ind
    

    def plot_linear_fit_error(self, index: int, longitude: np.ndarray,
                          latitude: np.ndarray, m: float, b: float) \
            -> "tuple(np.ndarray, np.ndarray)":
        """Plots the error of the linear fit

        Parameters
        ----------
            index: int
                index of the point where a linear fit should be completed
            longitude: np.ndarray
                longitude data
            latitude: np.ndarray
                latitude data
            m: int
                slope of linear fit
            b: int
                y-intercept of linear fit
        """
        long, abs_err, _ = self.get_linear_fit_error(index, longitude, latitude, m, b)

        title = 'Linear Prediction Absolute Error at Index = {}, Point ({}, {})'.format(
            index, round(longitude[index], 2), round(latitude[index], 2))

        fig_long, ax_long = plt.subplots()
        ax_long.plot(long, abs_err)
        ax_long.set_title(title, fontweight='bold')
        ax_long.set_xlabel("Longitude ($^\circ$)")
        ax_long.set_ylabel("Absolute Error ($^\circ$ Latitude)")


    def compute(self):
        data, latitude, longitude = self.load_data_celest(eci_positions)
        # latitude, longitude = set_parameters(0, 500, data) # only get first few points

        hull = self.get_convex_hull(self.target_coordinates)
        self.hull = hull
        self.data = data
        self.latitude = latitude
        self.longitude = longitude


    def plot(self):
        '''
        Generates plots based on precomputed data.
        '''
        # ensure data has been precomputed
        if self.data is None:
            raise Exception("Data has not been precomputed.")

        sns.set_theme()

        fig, ax = plt.subplots()

        # Plot the ground track
        self.plot_ground_track(self.data, self.longitude,
                               self.latitude, self.hull, ax)

        # Plot Toronto
        ax.plot(*list(reversed(self.target_coordinates)),
                marker='o', color='tab:red')

        self.plot_convex_hull_onto_axes(self.hull, ax)

        # Label the plot
        ax.set_title("Latitude vs. Longitude", fontweight='bold')
        ax.set_xlabel("Longitude ($^\circ$)")
        ax.set_ylabel("Latitude ($^\circ$)")

        # Plot a point on the ground track, plot a linear prediction from the point, plot the error of the prediction
        index = linear_fit[0]
        acceptable_err = linear_fit[1]
        ax.plot(self.longitude[index], self.latitude[index],
                marker='o', color='tab:red')
        self.plot_linear_fit_onto_axes(
            self.hull, index, self.longitude, self.latitude, ax, acceptable_err)

        plt.show()


if __name__ == "__main__":
    eci_positions = 'supernova_data.csv'
    target_coordinates = (45.4215, -75.6972) # Toronto
    linear_fit = (100, 2)

    gps = GPSPropogation(None, eci_positions, target_coordinates, linear_fit)

    gps.compute()
    gps.plot()
