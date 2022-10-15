from dataclasses import asdict, dataclass, field
import configparser
import inspect
import enterprise.signals.parameter # this is used but only implicitly
import enterprise_extensions.models
import collections.abc
import pickle, json
import importlib
import numpy as np


def get_default_args_types_from_function(func):
    """
    Given function, returns two dictionaries with default values and types
    code modified from: https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    """
    signature = inspect.signature(func)
    defaults = {}
    types = {}
    for k, v in signature.parameters.items():
        if v.default is not inspect.Parameter.empty:
            defaults[k] = v.default

        if v.annotation is inspect.Parameter.empty:
            print(f"Warning! {v} does not have an associated type annotation")
        else:
            types[k] = v.annotation
    return defaults, types


def update_dictionary_with_subdictionary(d, u):
    """
    Updates dictionary d with preference for contents of dictionary u
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

    pta_creating_function_parameters = {}
    pta_creating_functions = {}

    custom_classes = {}
    custom_function_returns = {}

    psrs = None
    noise_dict = None

    def update_from_file(self, config_file: str) -> None:
        """
        Set defaults for functions from file
        """
        config = configparser.ConfigParser(comment_prefixes=';', interpolation=configparser.ExtendedInterpolation())
        config.optionxform = str
        config.read(config_file)
        for section in config.sections():
            config_file_items = dict(config.items(section))

            if section == 'input' or section == 'output' or section == 'DEFAULT':
                # read in input / output files
                for item in config_file_items.copy():
                    if not config_file_items[item]:
                        config_file_items.pop(item)
                self.update_from_dict(**config_file_items)

            elif 'class' in config_file_items.keys():
                """
                Get default values for a module defined elsewhere
                """
                # Import a module defined elsewhere
                module = importlib.import_module(config_file_items['module'])

                # import a class from a module
                custom_class = getattr(module, config_file_items['class'])

                class_parameters, types = get_default_args_types_from_function(custom_class.__init__)
                class_parameters_from_file = self.apply_types(config_file_items, types,
                                                         exclude_keys=['module', 'class'])
                class_parameters = update_dictionary_with_subdictionary(class_parameters, class_parameters_from_file)
                self.custom_classes[section] = custom_class(**class_parameters)
            elif 'function' in config_file_items.keys():
                # import a module defined elsewhere
                module = importlib.import_module(config_file_items['module'])
                # import a function from a module
                custom_function = getattr(module, config_file_items['function'])
                function_parameters, types = get_default_args_types_from_function(custom_function)
                function_parameters_from_file = self.apply_types(config_file_items, types,
                                                            exclude_keys=['module', 'function'])

                if 'returns' in config_file_items.keys():
                    self.custom_function_returns[config_file_items['returns']] = custom_function(function_parameters_from_file)
                else:
                    # TODO not sure what to do if function is not PTA creating...
                    self.pta_creating_functions[section] = custom_function
                    self.pta_creating_function_parameters[section] = update_dictionary_with_subdictionary(
                                                                                function_parameters,
                                                                               function_parameters_from_file)
            else:
                try:
                    """
                    Get default values for models held in enterprise_extensions
                    """
                    model_function = getattr(enterprise_extensions.models, section)
                    self.pta_creating_function_parameters[section], types = get_default_args_types_from_function(model_function)
                    # Update default args with those held inside of path
                    config_file_items = {k: types[k](config_file_items[k]) for k in config_file_items}
                    self.pta_creating_function_parameters[section] = \
                        update_dictionary_with_subdictionary(self.pta_creating_function_parameters[section],
                                                             config_file_items)
                    self.pta_creating_functions[section] = model_function
                except AttributeError as e:
                    # TODO this should probably exit
                    print(e)
                    print(f"WARNING! there is no {section} in enterprise_extensions.models")
                    raise AttributeError

    def apply_types(self, dictionary, type_dictionary, exclude_keys=[]):
        """
        Given dictionary (from config_file) and dictionary containing types
        apply type to dictionary

        Note: if CUSTOM_CLASS:your_class is in dictionary[key],
        then instead of applying type,
        """
        out_dictionary = {}
        for key, value in dictionary.items():
            if key in exclude_keys:
                continue
            if 'CUSTOM_CLASS:' in value:
                # Apply custom class instance stored in custom_classes
                out_dictionary[key] = self.custom_classes[value.replace('CUSTOM_CLASS:', '')]
                continue
            if 'FUNCTION_CALL:' in value:
                function_call = value.replace('FUNCTION_CALL:', '')
                out_dictionary[key] = eval(function_call)
                continue
            if key not in type_dictionary.keys():
                print(f"WARNING! {key} is not within type dictionary!")
                print(f"Object value is {value} and type is {type(value)}")
                print(f"Continuing")
                continue
            # special comprehension for (1d) numpy arrays
            if type_dictionary[key] == np.ndarray:
                out_dictionary[key] = np.array([np.float(x) for x in value.split(',')])
            # Special comprehension for bool because otherwise bool('False') == True
            elif type_dictionary[key] == bool:
                out_dictionary[key] = dictionary[key].lower().capitalize() == "True"
            else:
                out_dictionary[key] = type_dictionary[key](value)

        return out_dictionary



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
        for key in self.pta_creating_function_parameters.keys():
            if 'noisedict' in self.pta_creating_function_parameters[key].keys():
                self.pta_creating_function_parameters[key]['noisedict'] = self.noise_dict

    def create_pta_object_from_signals(self):
        raise NotImplementedError

    def create_pta_object(self):
        """
        Using enterprise_models, create a PTA object
        """
        if self.psrs is None:
            self.load_pickled_pulsars()

        keys = list(self.pta_creating_function_parameters.keys())
        if len(keys) == 0:
            print("WARNING: create_pta_object, there are no pta_creating_functions!")
            raise IndexError

        pta = self.pta_creating_functions[keys[0]](psrs=self.psrs, **self.pta_creating_function_parameters[keys[0]])
        for key in keys[1:]:
            pta = pta + self.pta_creating_functions[key](psrs=self.psrs,
                                                             **self.pta_creating_function_parameters[key])

        return pta
