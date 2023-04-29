"""
    refer to https://github.com/emp0ry/solar-calculator/blob/main/main.py for code detail
"""
from math import radians, sin, cos, asin, atan2


def panel_energy(
    long,
    lat,
    date,
    info: dict,
    area: float,
    direction,
    tilt,
    conversion_efficiency: float = 0.2,
) -> float:
    diffuse_energy = info["diffuse"]["actual"] * conversion_efficiency

    rlat = radians(lat)
    rlon = radians(long)
    daynum = (
        367 * date.year
        - 7 * (date.year + (date.month + 9) // 12) // 4
        + 275 * date.month // 9
        + date.day
        - 730531.5
        + 12
    )
    # Mean longitude of the sun
    mean_long = daynum * 0.01720279239 + 4.894967873
    # Mean anomaly of the Sun
    mean_anom = daynum * 0.01720197034 + 6.240040768
    # Ecliptic longitude of the sun
    eclip_long = (
        mean_long
        + 0.03342305518 * sin(mean_anom)
        + 0.0003490658504 * sin(2 * mean_anom)
    )
    # Obliquity of the ecliptic
    obliquity = 0.4090877234 - 0.000000006981317008 * daynum
    # Right ascension of the sun
    rasc = atan2(cos(obliquity) * sin(eclip_long), cos(eclip_long))
    # Declination of the sun
    decl = asin(sin(obliquity) * sin(eclip_long))
    # Local sidereal time
    sidereal = 4.894961213 + 6.300388099 * daynum + rlon
    # Hour angle of the sun
    hour_ang = sidereal - rasc
    # Local elevation of the sun
    elevation = asin(sin(decl) * sin(rlat) + cos(decl) * cos(rlat) * cos(hour_ang))
    # Local azimuth of the sun
    azimuth = atan2(
        -cos(decl) * cos(rlat) * sin(hour_ang), sin(decl) - sin(rlat) * sin(elevation)
    )

    cos_incidence_angle = cos(elevation) * cos(tilt) * sin(direction - azimuth) + sin(
        elevation
    ) * cos(direction - azimuth)
    direct_energy = (
        info["direct"]["actual"] * conversion_efficiency * cos_incidence_angle
    )

    return (diffuse_energy + direct_energy) * area
