"""Join BasinATLAS v1.0 lev06 sub-basin attributes to HAPI-scored reservoirs by point-in-polygon."""
import time

import geopandas as gpd
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GDB = str(ROOT / 'data/enhancement_probe/basinatlas/BasinATLAS_v10.gdb')
HAPI = str(ROOT / 'data/processed/hapi_scores.csv')
OUT = str(ROOT / 'data/processed/basinatlas_matched.csv')

FIELDS = ['HYBAS_ID', 'SUB_AREA', 'UP_AREA', 'dis_m3_pyr', 'run_mm_syr',
          'ari_ix_uav', 'pre_mm_uyr', 'tmp_dc_uyr', 'sgr_dk_sav',
          'cly_pc_sav', 'for_pc_sse', 'pop_ct_ssu', 'riv_tc_ssu', 'urb_pc_sse', 'ire_pc_sse']

print('loading lev06 polygons...')
t0 = time.time()
basin = gpd.read_file(GDB, layer='BasinATLAS_v10_lev06', columns=FIELDS)
print('loaded in', round(time.time()-t0, 1), 's;', len(basin), 'polygons')

hapi = pd.read_csv(HAPI)
pts = gpd.GeoDataFrame(hapi[['GRAND_ID']], geometry=gpd.points_from_xy(hapi['LON'], hapi['LAT']), crs='EPSG:4326')

print('matching...')
joined = gpd.sjoin(pts, basin, how='left', predicate='intersects')
matched = joined[['GRAND_ID'] + FIELDS]
matched.to_csv(OUT, index=False)
n = matched.dropna(subset=['HYBAS_ID']).shape[0]
print('matched reservoirs:', n, '/', len(hapi))
print('sample:')
print(matched.head(4).to_string())
print('saved:', OUT)
