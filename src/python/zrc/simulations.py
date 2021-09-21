import os
import numpy as np

from .functools import FuncDataFrame


get_env_var_as_int = lambda var_name: int(os.getenv(var_name))

single_gamma_hist2d = (
    lambda hits:
    np.histogram2d(
        hits.localPosX,
        hits.localPosY,
        bins=get_env_var_as_int("SIPM_BINS")
    )[0].tolist()
)

def hist_hits_event_group_f(event_hits):
    get_gamma = lambda gamma_id: FuncDataFrame(
        Hits(event_hits)
        .coincidence.raw_hits
        .groupby('eventID')
        .apply(lambda h: h.groupby('photonID').get_group(gamma_id))
    )

    return [get_gamma(1), get_gamma(2)]



class SipmArray:
    def __init__(self):
        self.detector_size_xy = int(os.getenv("DETECTOR_SIZE_XY"))
        self.detector_size_z = int(os.getenv("DETECTOR_SIZE_Z"))
        self.bins = int(os.getenv("SIPM_BINS"))
    
    @property
    def sipm_boundaries_coord(self):
        return np.linspace(
            -self.detector_size_xy/2,
            self.detector_size_xy/2,
            self.bins+1
        )
    
    @property
    def sipm_center_coord(self):
        return (
            (
                np.roll(self.sipm_boundaries_coord, 1)
                + self.sipm_boundaries_coord)/2
        )[1:]

    @property
    def sipm_boundaries(self):
        return np.meshgrid(
            self.sipm_boundaries_coord, 
            self.sipm_boundaries_coord
        )
    
    @property
    def sipm_center(self):
        return np.meshgrid(self.sipm_center_coord, self.sipm_center_coord)


class Hits:
    def __init__(self, raw_hits: FuncDataFrame):
        self.raw_hits = raw_hits
    
        def _group_n_selector(group_key, filter_func):
            return lambda hits: Hits(
                FuncDataFrame(
                    hits
                    .raw_hits
                    .groupby(group_key)
                    .filter(filter_func)
                )
            )
        
        _fdf_select_process_by_name = lambda process_name: (
            lambda fdf: Hits(fdf.select_where(processName=process_name))
        )
        
        self._single = _group_n_selector(
            group_key="eventID",
            filter_func=lambda g: len(g.photonID.unique())==1
        )

        self._coincidence = _group_n_selector(
            group_key="eventID",
            filter_func=lambda g: len(g.photonID.unique())==2
        )

        self._single_has_compton = _group_n_selector(
            group_key=["eventID", "photonID"],
            filter_func=lambda g: 'Compton' in  g.processName.unique()
        )

        self._coincidence_has_compton = _group_n_selector(
            group_key="eventID",
            filter_func=lambda g: 'Compton' in  g.processName.unique()
        )
        
        self._coincidence_has_no_compton = _group_n_selector(
            group_key="eventID",
            filter_func=lambda g: 'Compton' not in  g.processName.unique()
        )

        self._Transportation = _fdf_select_process_by_name("Transportation")
        self._OpticalAbsorption = _fdf_select_process_by_name("OpticalAbsorption")
        self._Compton = _fdf_select_process_by_name("Compton")
        self._PhotoElectric = _fdf_select_process_by_name("PhotoElectric")

        self._event = lambda event_id: (
            lambda fdf: fdf.select_where(eventID=event_id)
        )

        # self._gamma_1 = lambda hits: hits.groupby('photonID').get_group(1)
        # self._gamma_2 = lambda hits: hits.groupby('photonID').get_group(2)

#         self._coincidence_with_one_gamma_compton_one_gamma_photon_electric = lambda hits: FuncDataFrame(
#             hits.loc[
#                 single_has_compton(self._gamma_1(coincidence(hits))).index
#                 .symmetric_difference(
#                     single_has_compton(self._gamma_2(coincidence(hits))).index
#                 )
#             ]
#         )

#         self._coincidence_two_gamma_compton = lambda hits: coincidence(single_has_compton(hits))
        
        self._to_cart3_by_key = lambda key: Cartesian3(*self.raw_hits.select(key).to_numpy().T)
        
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
        return self.raw_hits.groupby("eventID").apply(lambda g: g.processName.value_counts())[:,'Compton']
    
#     @property
#     def coincidence_with_one_gamma_compton_one_gamma_photon_electric(self):
#         return self._coincidence_with_one_gamma_compton_one_gamma_photon_electric(self.raw_hits)
    
#     @property
#     def coincidence_two_gamma_compton(self):
#         return self._coincidence_two_gamma_compton(self.raw_hits)

    @property
    def gamma_1(self):
        return Hits(self.raw_hits.select_where(photonID=1))

    @property
    def gamma_2(self):
        return Hits(self.raw_hits.select_where(photonID=2))

    @property
    def PDG22(self):
        return Hits(self.raw_hits.select_where(PDGEncoding=22))
    
    @property
    def Transportation(self):
        return self._Transportation(self.raw_hits)
    
    @property
    def OpticalAbsorption(self):
        return self._OpticalAbsorption(self.raw_hits)
    
    @property
    def Compton(self):
        return self._Compton(self.raw_hits)
    
    @property
    def PhotoElectric(self):
        return self._PhotoElectric(self.raw_hits)
    
    @property
    def local_pos(self):
        return self._to_cart3_by_key(['localPosX','localPosY','localPosZ'])
    
    @property
    def global_pos(self):
        return self._to_cart3_by_key(['posX','posY','posZ'])
    
    @property
    def source_pos(self):
        return self._to_cart3_by_key(['localPosX','localPosY','localPosZ'])
    
    def get_event(self, eventID):
        return self._event(eventID)(self.raw_hits)
    

    def check(self):
        assert (
            set(self.coincidence_has_no_compton.eventID.unique()) |
            set(self.coincidence_has_compton.eventID.unique()) == 
            set(self.coincidence.eventID.unique())
        )
        
    
