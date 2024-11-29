from skyfield.api import load
from skyfield.api import Topos


# Load the timescale and TLE data
ts = load.timescale()
t = ts.utc(2024, 11, 28, 19, 0, 0) 
# Load TLE data
satellites = load.tle_file('https://celestrak.org/NORAD/elements/supplemental/sup-gp.php?FILE=starlink&FORMAT=tle')
satellite_dict = {sat.name: sat for sat in satellites}
"""
# Get position of a specific satellite
ts = load.timescale()
t = ts.utc(2024, 1, 1, 12, 0, 0)  # Example time
satellite = satellite_dict['STARLINK-3000']
geocentric = satellite.at(t)
print(geocentric.position.km)
"""

print(satellite_dict)

# Calculate positions
for sat_name, satellite in satellite_dict.items():
    geocentric = satellite.at(t)
    position = geocentric.position.km  # Position in kilometers (X, Y, Z)
    print(f"{sat_name}: Position (X, Y, Z): {position}")