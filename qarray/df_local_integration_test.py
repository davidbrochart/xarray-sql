import unittest
import numpy as np
import xarray as xr
import pandas as pd

from .df import to_dd


class LocalDataframeTest(unittest.TestCase):
  def setUp(self) -> None:
    np.random.seed(42)
    temperature = 15 + 8 * np.random.randn(720, 1440, 240)
    precipitation = 10 * np.random.rand(720, 1440, 240)
    time = pd.date_range('2024-02-28', periods=240)
    reference_time = pd.Timestamp('2024-02-28')
    lat = np.linspace(-90, 90, num=720)
    lon = np.linspace(-180, 180, num=1440)

    self.ds = xr.Dataset(
      data_vars=dict(
        temperature=(['lat', 'lon', 'time'], temperature),
        precipitation=(['lat', 'lon', 'time'], precipitation),
      ),
      coords=dict(
        lat=lat,
        lon=lon,
        time=time,
        reference_time=reference_time,
      ),
      attrs=dict(description='Random weather.')
    )

  def test_compute_perf(self):
    df = to_dd(self.ds, dict(time=12))
    df.compute()
    print(df.head())
    # self.assertEqual(len(df), np.prod(list(self.ds.dims.values())))


if __name__ == '__main__':
  unittest.main()
