import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from celest.satellite import Satellite
from mpl_toolkits.mplot3d import Axes3D
from celest.satellite import Coordinate
from celest.encounter import GroundPosition
from scipy.spatial import ConvexHull, convex_hull_plot_2d

sns.set_theme()

def set_parameters(start, end, data):
    latitude = data[start:end, 0] # Degrees north of equator
    longitude = data[start:end, 1] # Degrees east of prime meridian
    return latitude, longitude

def set_parameters2(start, end, data):
    latitude = data[0, start:end] # Degrees north of equator
    longitude = data[1, start:end] # Degrees east of prime meridian
    return latitude, longitude

def get_convex_hull():
    toronto_lat = 43.6532
    toronto_lon = -79.3832

    toronto = GroundPosition(toronto_lat, toronto_lon)

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

    off_nad = coord.off_nadir(toronto)
    alt, _ = coord.horizontal(toronto) # alt is the elevation angle

    valid_latitudes = LAT[(off_nad < 60) & (alt > 0)] # we want an off-nadir angle less than 30 degrees, can try 60 
    valid_longitudes = LON[(off_nad < 60) & (alt > 0)]

    # Convex hull stuff
    '''
    points = []

    for i in range(len(valid_latitudes)):
        points.append([valid_longitudes[i], valid_latitudes[i]]) # Why YY first then XX?

    points = np.array(points)
    '''
    points = np.vstack((valid_longitudes, valid_latitudes)).T

    hull = ConvexHull(points)

    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'k-')

    ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'r--', lw=2)
    ax.plot(points[hull.vertices[0], 0], points[hull.vertices[0], 1], 'ro')

    return hull

def inside_hull_func(hull, latitude, longitude):
    list = np.vstack((longitude, latitude)).T
    bool_list = []
    '''
    for i in range(len(latitude)):
        list.append((longitude[i], latitude[i]))
    '''
    for point in list:
        bool_list.append(all((np.dot(eq[:-1], point) + eq[-1] <= 0) for eq in hull.equations))
    
    return np.array(bool_list)

def jumps_func(arr):
    diff_arr = np.diff(arr)
    jumps_arr = np.where(np.abs(diff_arr) > 10)
    jumps_arr = np.insert(jumps_arr, 0, -1)
    jumps_arr = np.append(jumps_arr, len(arr))

    return jumps_arr

def plot(longitude, latitude, hull, ax):
    diff_arr = np.diff(longitude)
    jumps = np.where(np.abs(diff_arr) > 50)
    jumps = np.insert(jumps, 0, -1)
    jumps = np.append(jumps, len(longitude))

    for i in range(len(jumps)-1):
        lat, long = set_parameters(jumps[i]+1, jumps[i+1], data)
        inside_ellipse = inside_hull_func(hull, lat, long)
        # plt.plot(long, lat, color='tab:blue')

        # plot green
        new_inside_ellipse = np.copy(inside_ellipse) # extend where the array is True by one on each side of a block of Trues # NP.COPY() TO MAKE A COPY OR ELSE BOTH GET MODIFIED
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
        jumps_arr = jumps_func(lat_inside)
        
        for i in range(len(jumps_arr)-1):
            lat_green, long_green = set_parameters2(jumps_arr[i]+1, jumps_arr[i+1], data_new)
            ax.plot(long_green, lat_green, color='tab:green')

        # plot blue
        outside_ellipse = ~inside_ellipse
        new_outside_ellipse = np.copy(outside_ellipse) # extend where the array is True by one on each side of a block of Trues
        updated = False
        for i in range(len(outside_ellipse)-1):
            if outside_ellipse[i] == False and outside_ellipse[i+1] == True:
                pass # green bleeds out a little
                # new_outside_ellipse[i] = True # blue bleeds in a little
            elif outside_ellipse[i] == True and outside_ellipse[i+1] == False:
                if updated == False:
                    new_outside_ellipse[i+1] = True
                    updated = True
                else:
                    updated = False

        lat_outside = lat[new_outside_ellipse]
        long_outside = long[new_outside_ellipse]
        data_new = np.array([np.array(lat_outside), np.array(long_outside)])
        jumps_arr = jumps_func(lat_outside)
        
        for i in range(len(jumps_arr)-1):
            lat_blue, long_blue = set_parameters2(jumps_arr[i]+1, jumps_arr[i+1], data_new)
            ax.plot(long_blue, lat_blue, color='tab:blue')

def get_line_points(index, longitude, latitude):
    x = []
    y = []
    for i in range(index-5, index+6):
        x.append(longitude[i])
        y.append(latitude[i])
    x = np.array(x).reshape((-1, 1))
    y = np.array(y)
    return x, y

def plot_line(index, longitude, latitude, ax, acceptable_err):
    x, y = get_line_points(index, longitude, latitude)
    model = LinearRegression().fit(x, y)
    m = model.coef_
    b = model.intercept_

    abs_err, ind_array = linear_pred_error(index, longitude, latitude, m, b) # also plots the error graphs
    
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

    ax.plot(acceptable_x_arr, acceptable_y_arr, color='tab:orange')

    # ax.axline((0, b), slope = m, color='tab:orange')
    ax.set_ylim(-120, 120)

    # plot the inside ellipse region a different colour
    ellipse_line_intersect(hull, acceptable_x_arr, acceptable_y_arr, ax)

def ellipse_line_intersect(hull, x_arr, y_arr, ax):
    inside = inside_hull_func(hull, y_arr, x_arr)
    plot_x = x_arr[inside]
    plot_y = y_arr[inside]

    ax.plot(plot_x, plot_y)

def linear_pred_error(index, longitude, latitude, m, b):
    range = 25 # arbitrarily chosen
    ind = np.linspace(index-range, index+range, range*2+1)
    long = longitude[index-range:index+range+1]
    actual = latitude[index-range:index+range+1]
    pred = long * m + b
    abs_err = abs(actual-pred)

    title = 'Linear Prediction Absolute Error at Index = {}, Point ({}, {})'.format(index, round(longitude[index], 2), round(latitude[index], 2))

    fig_long, ax_long = plt.subplots()
    ax_long.plot(long, abs_err)
    ax_long.set_title(title, fontweight='bold')
    ax_long.set_xlabel("Longitude ($^\circ$)")
    ax_long.set_ylabel("Absolute Error ($^\circ$ Latitude)")

    return abs_err, ind

def load_data_celest(file):
    orbit_data = np.loadtxt(file, delimiter=',')

    time_seconds = orbit_data[:, 0] # julian
    position_m = orbit_data[:, 1:4] # gcrs
    time_julian = time_seconds / 86400

    satellite = Satellite(position=position_m, frame='gcrs', 
                        julian=time_julian, offset=0)

    latitude, longitude, altitude = satellite.geo()

    data = np.array([latitude, longitude, altitude])
    data = np.transpose(data)

    return data, latitude, longitude

if __name__ == "__main__":

    data, latitude, longitude = load_data_celest("supernova_data.csv")
    # latitude, longitude = set_parameters(0, 500, data) # only get first few points

    toronto_lat = [43.6532] # Degrees north
    toronto_long = [-79.3832] # Degrees east

    fig, ax = plt.subplots()
    hull = get_convex_hull()

    # Plot the ground track
    plot(longitude, latitude, hull, ax)

    # Plot Toronto
    ax.plot(toronto_long, toronto_lat, marker='o', color='tab:red')
    # plt.annotate("Toronto", (-125, 44))

    # Label the plot
    ax.set_title("Latitude vs. Longitude", fontweight='bold')
    ax.set_xlabel("Longitude ($^\circ$)")
    ax.set_ylabel("Latitude ($^\circ$)")

    # Plot a point on the ground track, plot a linear prediction from the point, plot the error of the prediction 
    index = 150
    acceptable_err = 10
    ax.plot(longitude[index], latitude[index], marker='o', color='tab:red')
    plot_line(index, longitude, latitude, ax, acceptable_err)
    
    plt.show()