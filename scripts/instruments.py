# -*- coding: utf-8 -*-
import os
import math
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from general.functions import GenericInstrument


class SkinTemperature(GenericInstrument):
    def __init__(self, *args, **kwargs):
        super(SkinTemperature, self).__init__(*args, **kwargs)
        self.general_attributes = {
            "institution": "EPFL",
            "source": "Skin Temperature Measurements",
            "references": "LéXPLORE common instruments camille.minaudo@epfl.ch>",
            "history": "See history on Renku",
            "conventions": "CF 1.7",
            "comment": "Skin temperature measurements collected on Lexplore Platform in Lake Geneva",
            "title": "Lexplore Skin Temperature Measurements"
        }
        self.dimensions = {
            'time': {'dim_name': 'time', 'dim_size': None}
        }
        self.variables = {
            'time': {'var_name': 'time', 'dim': ('time',), 'unit': 'seconds since 1970-01-01 00:00:00',
                     'long_name': 'time'},
            'skintemp': {'var_name': 'skintemp', 'dim': ('time',), 'unit': 'degC', 'long_name': 'skin temperature',},
            'skytemp': {'var_name': 'skytemp', 'dim': ('time',), 'unit': 'degC', 'long_name': 'sky temperature'},
        }

    def read_data(self, file):
        self.log.info("Reading data from {}".format(file), 1)
        try:
            df = pd.read_csv(file, header=None, encoding="ISO-8859-1", skiprows=4)
            df.columns = ["date", "record", "skintemp", "skytemp"]
            df["time"] = df["date"].apply(lambda x: datetime.timestamp(datetime.strptime(x, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)))
            df.drop_duplicates(subset='date', inplace = True)
            df.sort_values("time", inplace=True)
            df.reset_index(inplace=True, drop=True)
            df["datetime"] = pd.to_datetime(df['time'], unit='s')
            df["skintemp"] = pd.to_numeric(df["skintemp"], errors='coerce')
            df["skytemp"] = pd.to_numeric(df["skytemp"], errors='coerce')
            for variable in self.variables:
                self.data[variable] = np.array(df[variable])
        except Exception as e:
            self.log.info("Failed to read data from {}".format(file), indent=1)
            raise e
        return True
