from dataclasses import asdict, dataclass, field
import configparser
import inspect
from enterprise_extensions import models
import collections.abc
import pickle, json

def get_default_args_from_function(func):
    """
    code taken from: https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def update_dictionary_with_subdictionary(d, u):
    """
    code taken from: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dictionary_with_subdictionary(d.get(k, {}), v)
        else:
            d[k] = v
    return d


@dataclass
class RunSettings:
    """Class for keeping track of enterprise model run settings"""
    config_file: str = None
    pulsar_pickle: str = None
    noise_dict_json: str = None

    enterprise_model_params = {}
    enterprise_model_functions = {}

    psrs = None
    noise_dict = None

    def update_from_file(self, config_file: str) -> None:
        """
        Set defaults for functions from file
        """
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(config_file)
        for section in config.sections():
            if section == 'input' or section == 'output':
                # read in input / output files
                mydict = dict(config.items(section))
                for item in mydict.copy():
                    if not mydict[item]:
                        mydict.pop(item)
                self.update_from_dict(**mydict)
            else:
                try:
                    """
                    Get default values for models held in enterprise_extensions
                    """
                    model_function = getattr(models, section)
                    self.enterprise_model_params[section] = get_default_args_from_function(model_function)
                    # Update default args with those held inside of path
                    self.enterprise_model_params[section] = \
                        update_dictionary_with_subdictionary(self.enterprise_model_params[section],
                                                             dict(config.items(section)))
                    self.enterprise_model_functions[section] = model_function
                except AttributeError as e:
                    print(e)
                    print(f"WARNING! there is no {section} in enterprise_extensions.models")
                    print(f"\t ignoring {section} section")

    def update_from_dict(self, **kwargs):
        ann = getattr(self, "__annotations__", {})
        for name, dtype in ann.items():
            if name in kwargs:
                try:
                    kwargs[name] = dtype(kwargs[name])
                except TypeError:
                    pass
                setattr(self, name, kwargs[name])

    def load_pickled_pulsars(self):
        """
        Set self.psrs and self.noise_dict
        """

        try:
            self.psrs = pickle.load(open(self.pulsar_pickle, 'rb'))
            self.noise_dict = json.load(open(self.noise_dict_json))
        except FileNotFoundError as e:
            print(e)
            exit(1)

        for par in list(self.noise_dict.keys()):
            if 'log10_ecorr' in par and 'basis_ecorr' not in par:
                ecorr = par.split('_')[0] + '_basis_ecorr_' + '_'.join(par.split('_')[1:])
                self.noise_dict[ecorr] = self.noise_dict[par]

        # assign noisedict to all enterprise models
        for key in self.enterprise_model_params.keys():
            self.enterprise_model_params[key]['noisedict'] = self.noise_dict

    def create_pta_object(self):
        """Using  enterprise_models, create a PTA object"""
        for key, value in self.enterprise_model_params.items():
            pta = self.enterprise_model_functions[key](self.psrs, **value)

        return pta
