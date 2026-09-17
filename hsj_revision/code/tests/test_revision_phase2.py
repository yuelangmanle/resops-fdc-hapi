import importlib.util,unittest
from pathlib import Path
import pandas as pd,numpy as np
p=Path(__file__).resolve().parents[1]/'03_analysis/revision_phase2.py';s=importlib.util.spec_from_file_location('phase2',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class ScientificTests(unittest.TestCase):
 def test_no_change_pair(self):
  dates=pd.date_range('1979-10-01','1999-09-30');d=pd.DataFrame({'date':dates,'inflow':1.,'outflow':2.});d['wy']=d.date.dt.year+(d.date.dt.month>=10).astype(int);r=m.pair(d,1980)
  self.assertEqual(r['early_end'],1989);self.assertEqual(r['late_start'],1990);self.assertAlmostEqual(r['abs_shape_change'],0);self.assertAlmostEqual(r['mean_shift'],0)
 def test_window_coverage(self):
  dates=pd.date_range('1979-10-01','1985-09-30');d=pd.DataFrame({'date':dates,'inflow':1.,'outflow':1.});d['wy']=d.date.dt.year+(d.date.dt.month>=10).astype(int);self.assertIsNone(m.pair(d,1980))
 def test_saved_predictions(self):
  q=pd.read_csv(m.O/'chronological_predictions.csv');self.assertTrue(np.isfinite(q[['observed','predicted']]).all().all());self.assertTrue((q.train_max_target_year<q.test_min_target_year).all());self.assertFalse(q.duplicated(['task','sample','model','ID']).any())
 def test_no_future_inputs(self):
  self.assertTrue(all(c.startswith('early_') for c in m.features));self.assertNotIn('season_shift',m.features)
if __name__=='__main__':unittest.main()
