"""
Unit Test - Testing units in `pvlib.py`

Author:
-------
Aashish Yadavally
"""
import pandas as pd
from einstein.models.pvlib import PVLibModel
from pvlib.forecast import NAM, GFS, HRRR, RAP, NDFD


PM = PVLibModel()


def test_model_define():
    models = [type(NAM()), type(GFS()), type(HRRR()), type(RAP()),
              type(NDFD())]
    model = PM.model_define()
    assert (type(model) in models) == True


def test_get_processed_data():
    data_frame = PM.get_processed_data()
    assert isinstance(data_frame, pd.DataFrame)
    # Testing that the dataframe is not empty
    assert data_frame.empty == False


def test_get_irradiance():
    data_frame = PM.get_processed_data()
    irrads = PM.forecast_model.cloud_cover_to_irradiance(
        data_frame['total_clouds'])
    # Testing that a Pandas DataFrame is returned
    assert isinstance(irrads, pd.DataFrame)
    # Testing that the Pandas DataFrame is non-empty
    assert irrads.empty == False
