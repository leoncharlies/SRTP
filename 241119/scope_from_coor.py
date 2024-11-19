import pandas as pd

# POINT (104.063068 30.685989),
def extract_coordinates(point_str):
    coords=point_str.replace("POINT (","").replace(")","").split()
    return float(coords[0]),float(coords[1])

