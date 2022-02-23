import os
import uuid
import numpy as np
import pandas as pd

from .functools import FuncDataFrame
from .geometry import Cartesian3


get_env_var_as_int = lambda var_name: int(os.getenv(var_name))


def single_gamma_hist2d(hits):
    return np.histogram2d(
        hits.localPosX, hits.localPosY, bins=get_env_var_as_int("SIPM_BINS")
    )[0].tolist()


class SipmArray:
    def __init__(self):
        self.detector_size_xy = int(os.getenv("DETECTOR_SIZE_XY"))
        self.detector_size_z = int(os.getenv("DETECTOR_SIZE_Z"))
        self.bins = int(os.getenv("SIPM_BINS"))

    @property
    def sipm_boundaries_coord(self):
        return np.linspace(
            -self.detector_size_xy / 2, self.detector_size_xy / 2, self.bins + 1
        )

    @property
    def sipm_center_coord(self):
        return (
            (np.roll(self.sipm_boundaries_coord, 1) + self.sipm_boundaries_coord) / 2
        )[1:]

    @property
    def sipm_boundaries(self):
        return np.meshgrid(self.sipm_boundaries_coord, self.sipm_boundaries_coord)

    @property
    def sipm_center(self):
        return np.meshgrid(self.sipm_center_coord, self.sipm_center_coord)

    @property
    def sipm_center_3d(self):
        return Cartesian3(
            *self.sipm_center,
            np.ones_like(self.sipm_center[0]) * (self.detector_size_z / 2)
        )


class Hits:
    def __init__(self, raw_hits: FuncDataFrame):
        self.raw_hits = raw_hits

        def _group_n_selector(group_key, filter_func):
            return lambda hits: Hits(
                FuncDataFrame(hits.raw_hits.groupby(group_key).filter(filter_func))
            )

        _fdf_select_process_by_name = lambda process_name: (
            lambda fdf: Hits(fdf.select_where(processName=process_name))
        )

        self._single = _group_n_selector(
            group_key="eventID", filter_func=lambda g: len(g.photonID.unique()) == 1
        )

        self._coincidence = _group_n_selector(
            group_key="eventID", filter_func=lambda g: len(g.photonID.unique()) == 2
        )

        self._single_has_compton = _group_n_selector(
            group_key=["eventID", "photonID"],
            filter_func=lambda g: "Compton" in g.processName.unique(),
        )

        self._coincidence_has_compton = _group_n_selector(
            group_key="eventID",
            filter_func=lambda g: "Compton" in g.processName.unique(),
        )

        self._coincidence_has_no_compton = _group_n_selector(
            group_key="eventID",
            filter_func=lambda g: "Compton" not in g.processName.unique(),
        )

        self._Transportation = _fdf_select_process_by_name("Transportation")
        self._OpticalAbsorption = _fdf_select_process_by_name("OpticalAbsorption")
        self._Compton = _fdf_select_process_by_name("Compton")
        self._PhotoElectric = _fdf_select_process_by_name("PhotoElectric")

        self._event = lambda event_id: (lambda fdf: fdf.select_where(eventID=event_id))

        self._to_cart3_by_key = lambda key: Cartesian3(
            *self.raw_hits.select(key).to_numpy().T
        )

    @property
    def counts(self):
        """
        Caution! counts only works on coincidence.

        Hits(hits_df).coincidence.counts
        """
        return (
            self.raw_hits.groupby(["eventID", "photonID"])
            .apply(
                lambda event_gamma: (
                    FuncDataFrame(event_gamma)
                    # select optical photon by first interaction crystalID
                    .select_where(crystalID=event_gamma.iloc[0].crystalID)
                    # select by Transportation process, which are those transport from back surface of crystal
                    .select_where(processName="Transportation")
                )
            )
            # group again by eventID and photonID
            .reset_index(drop=True)
            .groupby(["eventID", "photonID"])
            # calculate hits2d for each event_gamma
            .apply(single_gamma_hist2d)
            # merge counts by events
            .groupby(["eventID"])
            .apply(lambda e: [np.array(e)[0], np.array(e)[1]])
        )

    def sipm_center_pos(self, crystalRC):
        return (
            self.raw_hits
            # Hits(hits).coincidence.raw_hits
            .groupby(["eventID", "photonID"])
            .apply(
                lambda event_gamma: (
                    SipmArray()
                    .sipm_center_3d.move_by_crystalID(
                        event_gamma.iloc[0].crystalID, crystalRC
                    )
                    .flatten()
                    .to_numpy()
                )
            )
            # merge by events
            .groupby(["eventID"])
            .apply(lambda e: np.array([np.array(e)[0], np.array(e)[1]]))
        )

    @property
    def crystalID(self):
        return (
            self.raw_hits.groupby(["eventID", "photonID"])
            .apply(lambda g: g.iloc[0])
            .crystalID.groupby("eventID")
            .apply(lambda e: [np.array(e)[0], np.array(e)[1]])
        )

    @property
    def sourcePosX(self):
        return self.raw_hits.groupby(["eventID"]).apply(lambda g: g.iloc[0]).sourcePosX

    @property
    def sourcePosY(self):
        return self.raw_hits.groupby(["eventID"]).apply(lambda g: g.iloc[0]).sourcePosY

    @property
    def sourcePosZ(self):
        return self.raw_hits.groupby(["eventID"]).apply(lambda g: g.iloc[0]).sourcePosZ

    @property
    def single(self):
        return self._single(self)

    @property
    def coincidence(self):
        return self._coincidence(self)

    @property
    def coincidence_has_compton(self):
        return self._coincidence_has_compton(self._coincidence(self))

    @property
    def coincidence_has_no_compton(self):
        return self._coincidence_has_no_compton(self._coincidence(self))

    @property
    def num_of_compton_by_event(self):
        return self.raw_hits.groupby("eventID").apply(
            lambda g: g.processName.value_counts()
        )[:, "Compton"]

    @property
    def local_pos(self):
        return self._to_cart3_by_key(["localPosX", "localPosY", "localPosZ"])

    @property
    def global_pos(self):
        return self._to_cart3_by_key(["posX", "posY", "posZ"])

    @property
    def source_pos(self):
        return self._to_cart3_by_key(["localPosX", "localPosY", "localPosZ"])

    def get_event(self, eventID):
        return self._event(eventID)(self.raw_hits)

    def event_sample_hist_2d(self):
        gamma_1, gamma_2 = coincidence_group_by_event_n_gamma(self.raw_hits)
        return [
            single_gamma_hist2d(get_most_hit_crystal_optical_photons(gamma_1)),
            single_gamma_hist2d(get_most_hit_crystal_optical_photons(gamma_2)),
        ]

    def assemble_sample(self, crystalRC):
        sample = pd.DataFrame()
        sample["sourcePosX"] = self.coincidence.sourcePosX
        sample["sourcePosY"] = self.coincidence.sourcePosY
        sample["sourcePosZ"] = self.coincidence.sourcePosZ
        sample["counts"] = self.coincidence.counts
        sample["crystalID"] = self.coincidence.crystalID
        sample["sipm_center_pos"] = self.coincidence.sipm_center_pos(crystalRC)
        return sample

    def assign_to_experiment(self, experiment_id):
        result = pd.DataFrame()

        result["eventID"] = self.coincidence.raw_hits.eventID.unique()
        result["experiment_id"] = experiment_id

        result.to_csv("experiment_coincidence_event.csv", index=False)


def gen_uuid4():
    yield str(uuid.uuid4())


class HitsEventIDMapping:
    def __init__(self, df):
        self.df = df

    def get_by_key(self, key):
        return self.df[key]

    @staticmethod
    def from_file(path="./eventID_mapping.map"):
        return HitsEventIDMapping(dict(pd.read_csv(path).to_records(index=False)))

    @staticmethod
    def build(hits, path="./eventID_mapping.map"):
        try:
            id_map = HitsEventIDMapping.from_file().df
        except FileNotFoundError as e:
            id_map = {
                eventID: next(gen_uuid4()) for eventID in hits["eventID"].unique()
            }
            pd.DataFrame(
                list(id_map.items()), columns=["eventID_num", "eventID_uuid"]
            ).to_csv(path, index=False)
        return HitsEventIDMapping(id_map)

    def to_dict(self):
        return self.df

    def do_replace(self, hits):
        hits["eventID"] = pd.Series([self.df[eventID] for eventID in hits["eventID"]])


gamma_interaction_filters = {
    "PE": (
        lambda g: set([0]) == set(g.nCrystalCompton.unique())
        and "PhotoElectric" in g.processName.unique()
    ),
    "Compton->Escape": (
        lambda g: set([1]) == set(g.nCrystalCompton.unique())
        and "PhotoElectric" not in g.processName.unique()
    ),
    "Compton->PE": (
        lambda g: set([1]) == set(g.nCrystalCompton.unique())
        and "PhotoElectric" in g.processName.unique()
    ),
    "Compton->Compton->Escape": (
        lambda g: set([1, 2]) == set(g.nCrystalCompton.unique())
        and "PhotoElectric" not in g.processName.unique()
    ),
    "Compton->Compton->PE": (
        lambda g: set([1, 2]) == set(g.nCrystalCompton.unique())
        and "PhotoElectric" in g.processName.unique()
    ),
    "Compton->Compton->Compton->Escape": (
        lambda g: set([1, 2, 3]) == set(g.nCrystalCompton.unique())
        and "PhotoElectric" not in g.processName.unique()
    ),
    "Compton->Compton->Compton->PE": (
        lambda g: set([1, 2, 3]) == set(g.nCrystalCompton.unique())
        and "PhotoElectric" in g.processName.unique()
    ),
}


def hist2d(hits, bins=8):
    return np.histogram2d(hits.localPosX, hits.localPosY, bins=bins)[0].tolist()


def get_most_hited_crystal(hits):
    return hits.crystalID.value_counts().idxmax()


def processTransportation(hits):
    """
    Process ONLY Transportation optical photons.
    
    return tuple with 2 fields:
    1. hist2d of Transportation;
    2. time sorted time-vs-transportedPhotoPosition array
    """

    most_hited_crystalID = get_most_hited_crystal(hits)
    hits = hits[hits.crystalID == most_hited_crystalID]

    return {
        "hist2d": hist2d(hits[hits.processName == "Transportation"]),
        "trackLocalTimeVSPos": (
            hits[hits.processName == "Transportation"][
                ["trackLocalTime", "time", "localPosX", "localPosY", "localPosZ"]
            ]
            .to_numpy()
            .tolist()
        ),
    }


def processComptonFollowedByPE(hits):
    PE_index = hits[hits.processName == "PhotoElectric"].index.values

    assert len(PE_index) == 1
    PE_index = PE_index[0]

    compton_hits = hits[hits.index < PE_index]
    PE_hits = hits[hits.index >= PE_index]

    PE_hits_result = interact_step_process(PE_hits)
    PE_hits_result.update(nCrystalCompton=99)
    return {
        "interactionType": "isComptonFollowedByPE",
        "Compton": interact_step_process(compton_hits),
        "PhotoElectric": PE_hits_result,
    }


def interact_step_process(hits):
    hits_processes = set(hits.processName.unique())
    # isJustCompton
    if "Compton" in hits_processes and "PhotoElectric" not in hits_processes:
        return {
            "interactionType": "isJustCompton",
            **hits.iloc[0].to_dict(),
            **processTransportation(hits),
        }

    # isJustPE
    elif "Compton" not in hits_processes and "PhotoElectric" in hits_processes:
        return {
            "interactionType": "isJustPE",
            **hits.iloc[0].to_dict(),
            **processTransportation(hits),
        }

    # isComptonFollowedByPE
    elif "Compton" in hits_processes and "PhotoElectric" in hits_processes:
        result = processComptonFollowedByPE(hits)
        return result


class ComptonHits:
    def __init__(self, hits):
        self.hits = hits

    def group_and_process(self):
        hits_group_and_processed = (
            self.hits.groupby(["eventID", "photonID"])
            .filter(lambda g: "msc" not in g.processName.unique())
            .groupby(["eventID", "photonID", "nCrystalCompton"])
        ).apply(interact_step_process)

        # Just unpack ComptonFollowedByPE events
        hits_group_and_processed_ComptonFollowedByPE = hits_group_and_processed[
            hits_group_and_processed.apply(
                lambda r: r["interactionType"] == "isComptonFollowedByPE"
            )
        ].to_list()
        tmp = []
        for i in hits_group_and_processed_ComptonFollowedByPE:
            tmp.append(i["Compton"])
            tmp.append(i["PhotoElectric"])

        return pd.DataFrame(tmp).append(
            pd.DataFrame(
                hits_group_and_processed[
                    hits_group_and_processed.apply(
                        lambda r: r["interactionType"] != "isComptonFollowedByPE"
                    )
                ].to_list()
            )
        )
