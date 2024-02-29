#!/usr/bin/env python3
import numpy as np
import pandas as pd
import xarray as xr
import qarray as qr


np.random.seed(42)
temperature = 15 + 8 * np.random.randn(720, 1440, 240)
precipitation = 10 * np.random.rand(720, 1440, 240)
time = pd.date_range('2024-02-28', periods=240)
reference_time = pd.Timestamp('2024-02-28')
lat = np.linspace(-90, 90, num=720)
lon = np.linspace(-180, 180, num=1440)

ds = xr.Dataset(
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

# 24+ chunks creates memory pressure on my 16 GB M1 Mac.
# time=12 uses about all the memory available on my machine.
df = qr.to_dd(ds, dict(time=12))
df.compute()
print(df.head())
print(len(df))
