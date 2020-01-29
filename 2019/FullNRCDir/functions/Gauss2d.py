def Gauss2D(mesh, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta) -> np.ndarray:
    x = mesh[0]
    y = mesh[1]
    a = (np.cos(theta) ** 2) / (2 * x_stddev ** 2) + (np.sin(theta) ** 2) / (2 * y_stddev ** 2)
    b = -1 * (np.sin(2 * theta) ** 2)/(4 * x_stddev ** 2) + (np.sin(2 * theta) ** 2)/(4 * y_stddev ** 2)
    c = (np.cos(theta) ** 2) / (2 * y_stddev**2) + (np.sin(theta) ** 2) / (2 * x_stddev**2)
    gaussian = amplitude * np.exp(-1 * (1 * a * (x-x_mean) ** 2 +
                                      2 * b * (x - x_mean) * (y - y_mean) +
                                      1 * c * (y - y_mean) ** 2) )
    return gaussian