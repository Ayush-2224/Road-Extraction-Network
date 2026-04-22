import json
import urllib.request
import urllib.parse
import os

OUTPUT_GEOJSON = "indore/real_export.geojson"
BBOX = (22.7000, 75.8300, 22.7500, 75.8900) # (South Lat, West Lon, North Lat, East Lon)

print(f"Downloading real ground truth for BB: {BBOX}...")

query = f"""
[out:json][timeout:50];
(
  way["highway"]({BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]});
);
(._;>;);
out body;
"""

endpoints = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]

data = urllib.parse.urlencode({'data': query}).encode('utf-8')
result = None

for url in endpoints:
    print(f"Trying endpoint: {url}...")
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
            print("Success!")
            break
    except Exception as e:
        print(f"Failed on {url}: {e}")

if not result:
    print("All endpoints failed.")
    exit(1)

# Extract nodes and ways
nodes = {}
ways = []

for element in result.get('elements', []):
    if element['type'] == 'node':
        nodes[element['id']] = (element['lon'], element['lat'])
    elif element['type'] == 'way':
        ways.append(element)

print(f"Fetched {len(ways)} roads (ways) and {len(nodes)} nodes.")

# Convert to GeoJSON FeatureCollection
features = []
for way in ways:
    coords = []
    missing_nodes = False
    for node_id in way.get('nodes', []):
        if node_id in nodes:
            coords.append(nodes[node_id])
        else:
            missing_nodes = True
            break
    
    if not missing_nodes and len(coords) >= 2:
        feature = {
            "type": "Feature",
            "properties": {"id": way.get("id"), **way.get("tags", {})},
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            }
        }
        features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

os.makedirs(os.path.dirname(OUTPUT_GEOJSON), exist_ok=True)
with open(OUTPUT_GEOJSON, "w", encoding="utf-8") as f:
    json.dump(geojson, f, indent=2)

print(f"Successfully saved GeoJSON to {OUTPUT_GEOJSON} with {len(features)} LineString features.")
