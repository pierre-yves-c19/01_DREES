import os
import requests
import numpy as np
import pandas as pd
import xarray as xr


def calculate_vax_rollout(
    df_effectif,
    list_vac_status=[
        "Non-vaccinés",
        "Vaccinés 1 dose",
        "Vaccinés 2 doses",
        "Vaccinés 3 doses",
        "Vaccinés 4 doses",
    ],
):
    """Calculate vaccination rollout from vaccinal status"""
    df_vax_rollout = pd.DataFrame(index=df_effectif.index)
    n_max = len(list_vac_status) - 1
    df_vax_rollout[n_max] = df_effectif[list_vac_status[n_max]].diff()
    df_vax_rollout[n_max][df_vax_rollout[n_max] < 0] = 0
    n_dose = n_max - 1
    while n_dose > 0:
        print(n_dose)
        df_vax_rollout[n_dose] = (
            df_effectif[list_vac_status[n_dose]].diff() + df_vax_rollout[n_dose + 1]
        )
        df_vax_rollout[n_dose][df_vax_rollout[n_dose] < 0] = 0
        n_dose -= 1
    return df_vax_rollout


class DREES:

    def __init__(self) -> None:
        # Create folders
        for folder in ["data", "graph"]:
            if not os.path.exists(folder):
                os.makedirs(folder)
        for folder in ["age", "region", "national"]:
            if not os.path.exists("graph/" + folder):
                os.makedirs("graph/" + folder)
        # Session for API requests
        self.request_session = requests.Session()

    def get_covid_dataset_list(self):
        r = self.request_session.get(
            "https://data.drees.solidarites-sante.gouv.fr/api/datasets/1.0/search",
            params={"rows": -1},
        )
        dict_dataset = r.json()
        list_datasets = dict_dataset["datasets"]
        list_datasets_id = [dataset["datasetid"] for dataset in list_datasets]
        covid_datasets = [dataset for dataset in list_datasets_id if "covid" in dataset]
        self.covid_datasets = covid_datasets
        return covid_datasets

    def download_data(self, force_reload=False):

        for dataset in self.covid_datasets:
            dataset_file = f"data/df_{dataset}.pkl"
            if not os.path.isfile(dataset_file) or force_reload:
                print(f"Downloading dataset: {dataset}")
                r = self.request_session.get(
                    "https://data.drees.solidarites-sante.gouv.fr/api/records/1.0/search/",
                    params={
                        "dataset": dataset,
                        "rows": -1,
                    },
                )

                df = pd.json_normalize(r.json()["records"])
                dataset_date = pd.to_datetime(df.record_timestamp.unique()[0])
                print(f"Dataset date: {dataset_date:'%Y-%m-%d'}")
                df = df[[col for col in df.columns if "fields." in col]]
                df.columns = [col.split(".")[1] for col in df.columns]
                df.index = pd.to_datetime(df.date)
                df.sort_index(inplace=True)
                df = df.replace({"NA": np.nan})
                df = df.astype(float, errors="ignore")
                df.to_pickle(dataset_file)

    def create_dataset(self, list_dataset, name):
        file = f"data/ds_{name}.nc"
        print(name)
        list_ds = []
        list_version = []
        for dataset in list_dataset:
            # print(dataset)
            version = dataset.split("-")[2]
            list_version.append(version)
            df = pd.read_pickle(f"data/df_{dataset}.pkl")
            list_columns = list(df.columns[df.dtypes == object].drop("date"))
            list_columns = [
                col
                for col in list_columns
                if not "pourcent_" in col and not "pcr_" in col
            ]
            df = pd.pivot_table(
                df,
                columns=list_columns,
                index=df.date,
            )
            df.index = pd.to_datetime(df.index)
            df = pd.concat([df], axis=1, keys=[version])
            list_ds.append(df.unstack().to_xarray())

        ds = xr.concat(list_ds, dim="level_0")
        ds = ds.rename({"level_0": "version", "level_1": "variable"})
        ds = ds.astype(float)
        # df.columns.names = ["version", "indicator", "vax_status"]
        ds.to_netcdf(file)
        return ds

    def create_datasets(self):
        list_dataset = [dataset for dataset in self.covid_datasets if "age" in dataset]
        ds_age = self.create_dataset(list_dataset=list_dataset, name="by_age")

        list_dataset = [
            dataset for dataset in self.covid_datasets if "region" in dataset
        ]
        ds_region = self.create_dataset(list_dataset=list_dataset, name="by_region")

        list_dataset = [
            dataset
            for dataset in self.covid_datasets
            if not "age" in dataset and not "region" in dataset
        ]
        ds = self.create_dataset(list_dataset=list_dataset, name="national")
        return ds_age, ds_region, ds

    def group_vac_statut(self, ds, binary=False):
        list_vaccin_status = ds.vac_statut.values
        if binary:
            dict_vac_groups = {
                "Non-vaccinés": ["Non-vaccinés"],
                "Vaccinés au moins 1 dose": [
                    col for col in list_vaccin_status if col != "Non-vaccinés"
                ],
            }
        else:
            dict_vac_groups = {
                "Non-vaccinés": ["Non-vaccinés"],
                "Vaccinés 1 dose": [
                    col for col in list_vaccin_status if "Primo dose" in col
                ],
                "Vaccinés 2 doses": [
                    col for col in list_vaccin_status if "sans rappel" in col
                ],
                "Vaccinés 3 doses": [
                    col for col in list_vaccin_status if "1 rappel" in col
                ],
                "Vaccinés 4 doses": [
                    col for col in list_vaccin_status if "2 rappel" in col
                ],
            }
        list_vac_groups = [key for key in dict_vac_groups]
        ds = ds.rename({"vac_statut": "Sub_vac_status"})
        ds = xr.concat(
            [
                ds.sel(Sub_vac_status=dict_vac_groups[vac_groups])
                for vac_groups in list_vac_groups
            ],
            dim="vac_statut",
        )
        ds = ds.assign_coords(coords={"vac_statut": list_vac_groups})
        ds = ds.sum(dim="Sub_vac_status")
        return ds
