# -*- coding: utf-8 -*-+
from numpy import pi
import yaml 
from yaml.loader import SafeLoader
from geographiclib.geodesic import Geodesic

def estimacion_epicentro(station,dist,baz,rad=None):
    """
    from numpy import pi
    import yaml 
    from yaml.loader import SafeLoader
    from geographiclib.geodesic import Geodesic
    station(str): nombre estacion
    dist(float): distancia en km
    baz(float): backazimuth
    rad(bool): baz en radianes
    """  
    with open('data/datos_estaciones.yaml') as data_st_file:
        data_st = yaml.load(data_st_file,Loader=SafeLoader)

    if rad == True:
        baz = baz*(180/pi)
    dist_m = dist*1000

    geo_dict = Geodesic.WGS84.Direct(
        lat1=data_st[station][0],
        lon1=data_st[station][1],
        azi1=baz,
        s12=dist_m
        )

    lat = geo_dict['lat2']
    lon = geo_dict['lon2']

    return [lat, lon]