"""Download the City of Calgary water main data into data/raw/.

Sources (Open Government Licence - City of Calgary):
  Water Main Breaks:  https://data.calgary.ca/Environment/Water-Main-Breaks/dpcu-jr23
  Public Water Main:  https://data.calgary.ca/Environment/Public-Water-Main/w6h9-w33i

Usage:
    python -m src.pipeline.download_data
"""
from pathlib import Path
from urllib.request import urlretrieve

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
FILES = {
    "breaks.csv": "https://data.calgary.ca/api/views/dpcu-jr23/rows.csv?accessType=DOWNLOAD",
    "pipes.geojson": "https://data.calgary.ca/api/geospatial/w6h9-w33i?method=export&format=GeoJSON",
}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        urlretrieve(url, RAW_DIR / name)
        print(f"{name}: {(RAW_DIR / name).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
