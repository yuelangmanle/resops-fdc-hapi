"""Compare freshly reproduced outputs against manuscript claims; produce reviewer-facing report."""
import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIRS = (ROOT / 'data/processed', ROOT / 'outputs/submission/source_data', ROOT / 'source_data')
MD_PATHS = (ROOT / 'manuscript/drafts/STANDARD_MANUSCRIPT.md', ROOT / 'STANDARD_MANUSCRIPT.md')
MD_PATH = next((p for p in MD_PATHS if p.exists()), None)
MD = MD_PATH.read_text(encoding='utf-8') if MD_PATH else ''

claims = []

def claim(name, reproduced, manuscript, tol=None, ok=None):
    claims.append({
        'claim': name,
        'reproduced': reproduced,
        'manuscript': manuscript,
        'match': ok if ok is not None else (reproduced is not None and abs(float(reproduced) - float(manuscript)) <= (tol if tol else 0.005))
    })

def read_csv(name):
    for directory in DATA_DIRS:
        p = directory / name
        if p.exists():
            with open(p, encoding='utf-8') as f:
                return list(csv.DictReader(f))
    return None

def read_json(name):
    # Prefer the canonical project output.  The submission package contains a
    # synchronized copy, but it can be older while a manuscript is being
    # revised; using it first can make a local audit report stale values.
    candidates = [ROOT / 'outputs' / name]
    candidates.extend(directory / name for directory in DATA_DIRS)
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text(encoding='utf-8'))
    return None

# --- 1. sample size and reference-inflow strata ---
fra = read_csv('flow_regime_alteration.csv')
claim('US reservoirs (n)', len(fra) if fra else None, 433)

brazil = read_csv('brazil_flow_regime_alteration.csv')
claim('Brazil reservoirs (n)', len(brazil) if brazil else None, 134)

if fra:
    observed = [r for r in fra if r.get('ref_type') == 'observed']
    simulated = [r for r in fra if r.get('ref_type') == 'simulated']
    claim('observed-inflow reservoirs (n)', len(observed), 251)
    claim('simulated-inflow reservoirs (n)', len(simulated), 182)

    def median_field(rows, field):
        values = [float(r[field]) for r in rows if r.get(field) not in (None, '')]
        return round(statistics.median(values), 1) if values else None

    for label, rows, expected in [
        ('observed-inflow', observed, (-23.0, -22.8, -5.6, 4.6)),
        ('simulated-inflow', simulated, (-40.6, -19.9, None, None)),
    ]:
        q5_expected, q10_expected, q50_expected, q75_expected = expected
        claim(f'{label} median Q5 change (%)', median_field(rows, 'alter_q5_pct'), q5_expected, tol=0.15)
        claim(f'{label} median Q10 change (%)', median_field(rows, 'alter_q10_pct'), q10_expected, tol=0.15)
        if q50_expected is not None:
            claim(f'{label} median Q50 change (%)', median_field(rows, 'alter_q50_pct'), q50_expected, tol=0.15)
            claim(f'{label} median Q75 change (%)', median_field(rows, 'alter_q75_pct'), q75_expected, tol=0.15)

hapi = read_csv('hapi_scores.csv')
claim('HAPI scored reservoirs (n)', len(hapi) if hapi else None, 430)

# --- 2. FDC shape median ---
if fra:
    w1 = [float(r['w1_fdc_shape']) for r in fra if r.get('w1_fdc_shape') not in (None, '')]
    med = statistics.median(w1)
    claim('median w1_fdc_shape', round(med, 3), 0.237)
    claim('n with w1_fdc_shape', len(w1), 430, tol=5)

# --- 3. quantile alterations ---
if fra:
    q5 = [float(r['alter_q5_pct']) for r in fra if r.get('alter_q5_pct') not in (None, '')]
    q10 = [float(r['alter_q10_pct']) for r in fra if r.get('alter_q10_pct') not in (None, '')]
    q50 = [float(r['alter_q50_pct']) for r in fra if r.get('alter_q50_pct') not in (None, '')]
    q75 = [float(r['alter_q75_pct']) for r in fra if r.get('alter_q75_pct') not in (None, '')]
    claim('full-sample median Q5 change (%)', round(statistics.median(q5), 1), -31.9, tol=0.15)
    claim('full-sample median Q10 change (%)', round(statistics.median(q10), 1), -21.2, tol=0.15)
    claim('full-sample median Q50 change (%)', round(statistics.median(q50), 1), -3.3, tol=0.15)
    claim('full-sample median Q75 change (%)', round(statistics.median(q75), 1), 4.1, tol=0.15)
    observed_sentence = re.search(
        r'In the observed-inflow subset, median Q5 and Q10 changes were\s+[-−]23\.0% and [-−]22\.8%, while Q50 and Q75 changes were\s+[-−]5\.6% and \+4\.6%',
        MD,
    )
    claim('observed Q50/Q75 manuscript binding', bool(observed_sentence), True, ok=bool(observed_sentence))
    neg = sum(1 for x in q10 if x < 0)
    claim('Q10 negative count', neg, 222, tol=0.5)
    claim('Q10 negative ratio (%)', round(neg/len(q10)*100, 1), 60.7, tol=0.5)

# --- 4. clusters ---
cl = read_csv('fdc_functional_clusters.csv')
if cl:
    from collections import Counter
    cnt = Counter(r['cluster'] for r in cl)
    sizes = [cnt.get(str(i), cnt.get(i, 0)) for i in range(4)]
    claim('cluster sizes (0,1,2,3)', str(sizes), '[254, 27, 92, 60]', ok=sizes == [254, 27, 92, 60])

# --- 5. HAPI sensitivity ---
sens = read_json('hapi_sensitivity_results.json') or read_json('hapi_sensitivity.json')
if sens:
    claim('HAPI Spearman mean', round(sens.get('spearman_mean', 0), 3), 0.89, tol=0.01)
    claim('HAPI Spearman p5', round(sens.get('spearman_p5', 0), 3), 0.57, tol=0.02)
    claim('HAPI top10 overlap mean', round(sens.get('top10_overlap_mean', 0), 3), 0.555, tol=0.02)

# --- 6. heterogeneity ---
hb = read_json('heterogeneity_binary_results.json')
if hb:
    claim('CATE mean', round(hb.get('ate_mean', 0), 3), 0.041, tol=0.005)
    claim('pct positive CATE (%)', round(hb.get('pct_positive_cate', 0)*100, 1), 85.4, tol=0.5)
    ci = hb.get('ate_bootstrap_ci', {})
    if ci:
        claim('bootstrap ATE CI lower', round(ci.get('ci_lower_2.5', 0), 2), -0.04, tol=0.02)
        claim('bootstrap ATE CI upper', round(ci.get('ci_upper_97.5', 0), 2), 0.11, tol=0.02)

cr = read_json('causal_robustness.json')
if cr:
    claim('robust CATE mean', round(cr.get('ate_mean', 0), 3), 0.027, tol=0.005)
    claim('robust pct positive (%)', round(cr.get('pct_positive_cate', 0)*100, 1), 93.7, tol=0.5)

# --- 7. cluster robustness ---
rob = read_json('fdc_cluster_robustness.json')
if rob:
    claim('silhouette k=4', round(rob.get('silhouette_by_k', {}).get('4', 0), 3), 0.329, tol=0.005)
    claim('ARI km/aggl', round(rob.get('ari_kmeans_agglomerative_k4', 0), 3), 0.746, tol=0.01)
    claim('bootstrap ARI mean', round(rob.get('bootstrap_ari_mean', 0), 3), 0.940, tol=0.01)

# --- 8. brazil ---
if brazil:
    bw = [float(r['w1_fdc_shape']) for r in brazil if r.get('w1_fdc_shape') not in (None, '')]
    claim('Brazil median w1_fdc_shape', round(statistics.median(bw), 3), 0.029, tol=0.005)

# --- 9. regional subgroup audit ---
additional = read_json('additional_stats.json')
regional = additional.get('regional_q10_stratification', {}) if additional else {}
legacy_region_markers = (
    '85 of 182', '73 of 251', '48.7%', '37.0%', '+0.5%', '+2.0%',
)
claim(
    'regional Q10 subgroup omitted',
    regional.get('status'),
    'not_reported',
    ok=(regional.get('status') == 'not_reported' and not any(m in MD for m in legacy_region_markers)),
)

# --- 10. GloFAS sensitivity must perturb only simulated-inflow records ---
glofas = read_json('glofas_bias_sensitivity.json')
if glofas:
    expected_glofas = {
        '-0.3': (-32.4, 0.637),
        '-0.2': (-27.5, 0.628),
        '-0.1': (-26.2, 0.623),
        '0.0': (-21.2, 0.607),
        '0.1': (-20.1, 0.596),
        '0.2': (-14.9, 0.593),
        '0.3': (-12.0, 0.577),
    }
    for bias, (median_expected, negative_expected) in expected_glofas.items():
        result = glofas.get(bias, {})
        claim(f'GloFAS {bias} median Q10 (%)', round(result.get('median_alter_q10', 0), 1), median_expected, tol=0.15)
        claim(f'GloFAS {bias} negative share', round(result.get('negative_ratio', 0), 3), negative_expected, tol=0.002)
        claim(f'GloFAS {bias} observed count', result.get('n_observed'), 226, ok=(result.get('n_observed') == 226))
        claim(f'GloFAS {bias} simulated count', result.get('n_simulated'), 140, ok=(result.get('n_simulated') == 140))

# --- 11. remote sensing ---
rs = read_csv('remote_sensing_ndvi_ndwi_all.csv')
if rs:
    import math
    def corr(xs, ys):
        n = len(xs)
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x-mx)**2 for x in xs))
        dy = math.sqrt(sum((y-my)**2 for y in ys))
        return num/(dx*dy)
    score_key = 'HAPI'
    hapi_scores = [float(r[score_key]) for r in rs]
    ndvis = [float(r['ndvi']) for r in rs]
    ndwis = [float(r['ndwi']) for r in rs]
    claim('Pearson HAPI-NDVI', round(corr(hapi_scores, ndvis), 3), -0.315, tol=0.01)
    claim('Pearson HAPI-NDWI', round(corr(hapi_scores, ndwis), 3), 0.044, tol=0.01)

# --- output ---
print('CLAIM CHECK (reproduced vs manuscript):')
print('=' * 70)
passed = 0
for c in claims:
    mark = 'PASS' if c['match'] else 'FAIL'
    if c['match']:
        passed += 1
    print(f"{mark:5s} | {c['claim']:34s} | reproduced={c['reproduced']} | manuscript={c['manuscript']}")
print('=' * 70)
print(f'passed {passed}/{len(claims)}')

# save report
OUT = ROOT / 'outputs' / 'reproducibility'
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'claim_check.json').write_text(json.dumps(claims, ensure_ascii=False, indent=1))
