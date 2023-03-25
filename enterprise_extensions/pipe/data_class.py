import collections.abc
import configparser
import importlib
import inspect
import json
import pickle
import re
from dataclasses import dataclass, field

import numpy as np
from enterprise.signals import signal_base
import enterprise.signals.parameter  # this is used but only implicitly
import enterprise_extensions.models


def get_default_args_types_from_function(func):
    """
    Given function, returns two dictionaries with default values and types
    code modified from: https://stackoverflow.com/questions/12627118/get-a-function-arguments-default-value
    """
    signature = inspect.signature(func)
    defaults = {}
    types = {}
    for key, value in signature.parameters.items():
        # get default kwarg value from function
        if value.default is not inspect.Parameter.empty:
            defaults[key] = value.default

        # get type annotation from function
        if value.annotation is inspect.Parameter.empty:
            print(f"Warning! in {func} {value} does not have an associated type annotation")
        else:
            types[key] = value.annotation
    return defaults, types


def update_dictionary_with_subdictionary(d, u):
    """
    Updates dictionary d with preference for contents of dictionary u
    code taken from: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
    if d is None:
        return u
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dictionary_with_subdictionary(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_module_globally(package_dict):
    # import modules globablly from dictionary import key as item
    for import_as, package_name in package_dict.items():
        mod = importlib.import_module(package_name)
        globals()[import_as] = mod
    return


@dataclass()
class RunSettings:
    """
    Class for keeping track of enterprise model run settings
    TODO: link to examples of how to use
    """
    pulsar_pickle: str = None
    noise_dict_json: str = None

    # dictionary of functions that create signals
    signal_creating_function_keys: list = field(default_factory=list)

    # dictionary of functions that create signals depending on parameters of each pulsar
    per_pulsar_signal_creating_function_keys: dict = field(default_factory=dict)

    # dictionary of functions that create pta objects
    pta_creating_function_keys: list = field(default_factory=list)

    custom_classes: dict = field(default_factory=dict)
    custom_class_parameters: dict = field(default_factory=dict)

    function_parameters: dict = field(default_factory=dict)
    functions: dict = field(default_factory=dict)
    custom_return: dict = field(default_factory=dict)

    psrs: list = field(default_factory=list)
    noise_dict: dict = field(default_factory=dict)
    # stores config file as dictionary
    config_file_items: dict = field(default_factory=dict)

    def update_from_file(self, config_file: str) -> None:
        """
        Set defaults for functions from file

        [modules]: example np=numpy will load numpy as np globally
        """
        config = configparser.ConfigParser(comment_prefixes=';',
                                           interpolation=configparser.ExtendedInterpolation())
        config.optionxform = str
        config.read(config_file)
        exclude_keys = ['function', 'module', 'class', 'signal_return', 'pta_return',
                        'custom_return', 'per_pulsar_signal', 'singular_pulsar_signal']
        for section in config.sections():
            config_file_items = dict(config.items(section))
            self.config_file_items[section] = config_file_items
            if section == 'modules':
                load_module_globally(config_file_items)
            elif 'class' in config_file_items.keys():
                """
                Initialize a class given in a config file 
                """
                if 'module' in config_file_items.keys():
                    # get a class defined from a module
                    module = importlib.import_module(config_file_items['module'])
                    custom_class = getattr(module, config_file_items['class'])
                else:
                    # or if class module has already been imported, do this!
                    custom_class = eval(config_file_items['class'])

                default_class_parameters, types = get_default_args_types_from_function(custom_class.__init__)
                class_parameters_from_file = self.apply_types(config_file_items, types,
                                                              exclude_keys=exclude_keys)
                class_parameters = update_dictionary_with_subdictionary(default_class_parameters,
                                                                        class_parameters_from_file)
                self.custom_classes[section] = custom_class(**class_parameters)
                self.custom_class_parameters[section] = class_parameters

            elif 'function' in config_file_items.keys():
                if 'module' in config_file_items.keys():
                    # get a class defined from a module
                    module = importlib.import_module(config_file_items['module'])
                    custom_function = getattr(module, config_file_items['function'])
                else:
                    # or if class module has already been imported, do this!
                    custom_function = eval(config_file_items['function'])

                try:
                    default_function_parameters, types = get_default_args_types_from_function(custom_function)
                except ValueError:
                    # builtin functions like dictionary and list don't have default types
                    default_function_parameters, types = {}, {}
                function_parameters_from_file = self.apply_types(config_file_items, types,
                                                                 exclude_keys=exclude_keys)
                self.functions[section] = custom_function
                self.function_parameters[section] = update_dictionary_with_subdictionary(
                    default_function_parameters,
                    function_parameters_from_file)

                if 'custom_return' in config_file_items.keys():
                    # custom_return means to store the return value of this function in self.custom_function_return
                    self.custom_return[config_file_items['custom_return']] = \
                        self.functions[section](**self.function_parameters[section])
                elif 'signal_return' in config_file_items.keys():
                    # label this function as something that returns signal models
                    self.signal_creating_function_keys.append(section)
                elif 'per_pulsar_signal' in config_file_items.keys():
                    # Per pulsar can either be used as a function applied to every pulsar, or specify one by name
                    if (config_file_items['per_pulsar_signal'] == 'EACH_PULSAR') \
                            or (config_file_items['per_pulsar_signal'] == 'True'):
                        try:
                            self.per_pulsar_signal_creating_function_keys['EACH_PULSAR']
                        except KeyError:
                            self.per_pulsar_signal_creating_function_keys['EACH_PULSAR'] = []
                        self.per_pulsar_signal_creating_function_keys['EACH_PULSAR'].append(section)
                    else:
                        pulsar_name = config_file_items['per_pulsar_signal']
                        try:
                            self.per_pulsar_signal_creating_function_keys[pulsar_name]
                        except KeyError:
                            self.per_pulsar_signal_creating_function_keys[pulsar_name] = []
                        self.per_pulsar_signal_creating_function_keys[pulsar_name].append(section)

                elif 'pta_return' in config_file_items.keys():
                    # label this function as something that returns ptas
                    self.pta_creating_function_keys.append(section)
                if 'singular_pulsar' in config_file_items.keys():
                    # Only apply to given pulsar
                    pulsar_name = config_file_items['singular_pulsar']
                    try:
                        self.singular_pulsar_function_keys[pulsar_name]
                    except KeyError:
                        self.singular_pulsar_function_keys[pulsar_name] = []
                    self.singular_pulsar_function_keys[pulsar_name].append(section)

            else:
                # If not a class or function or module
                # it must be something specified in the RunSettings class
                # now read those in from the file
                for item in config_file_items.copy():
                    if not config_file_items[item]:
                        config_file_items.pop(item)
                self.update_from_dict(**config_file_items)

    def apply_types(self, dictionary, type_dictionary, exclude_keys=None):
        """
        Given dictionary (usually created from config_file) and dictionary containing types
        apply type to dictionary

        if CUSTOM_CLASS:your_class is in dictionary[key],
            instead of applying type it assigns from self.custom_classes
        if CUSTOM_FUNCTION_RETURN:whatever is in dictionary[key]
            instead of applying type it assigns from self.custom_returns[whatever]
        if EVAL:whatever is in dictionary[key]
            will call eval("whatever") and assign that
        """
        if exclude_keys is None:
            exclude_keys = []
        out_dictionary = {}
        for key, value in dictionary.items():
            if key in exclude_keys:
                continue
            if 'CUSTOM_FUNCTION_RETURN:' in value or 'CUSTOM_RETURN' in value:
                if 'CUSTOM_FUNCTION_RETURN:' in value:
                    value = value.replace('CUSTOM_FUNCTION_RETURN', 'CUSTOM_RETURN')
                    print("CUSTOM_FUNCTION_RETURN has been renamed CUSTOM_RETURN, please use that instead")
                out_dictionary[key] = self.custom_return[value.replace('CUSTOM_RETURN:', '')]
                continue
            # Apply custom class instance stored in custom_classes
            if 'CUSTOM_CLASS:' in value:
                # Apply custom class instance stored in custom_classes
                out_dictionary[key] = self.custom_classes[value.replace('CUSTOM_CLASS:', '')]
                continue
            if 'EVAL:' in value:
                function_call = value.replace('EVAL:', '')
                out_dictionary[key] = eval(function_call)
                continue
            if key not in type_dictionary.keys():
                print(f"WARNING! apply_types: {key} is not within type dictionary!")
                print(f"\tObject value is {value} and type is {type(value)}")
                print("\tIgnoring value and continuing")
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
        print(f'WARNING: {set(kwargs.keys()) - set(ann.keys())} arguments are getting ignored!')

    def load_pickled_pulsars(self):
        """
        Set self.psrs and self.noise_dict
        """

        try:
            self.psrs = self.get_pulsars()
            self.noise_dict = self.get_noise_dict()
        except FileNotFoundError as e:
            print(e)
            exit(1)

        for par in list(self.noise_dict.keys()):
            if 'log10_ecorr' in par and 'basis_ecorr' not in par:
                ecorr = par.split('_')[0] + '_basis_ecorr_' + '_'.join(par.split('_')[1:])
                self.noise_dict[ecorr] = self.noise_dict[par]

        # assign noisedict to all enterprise models
        for key in self.pta_creating_function_keys:
            if 'noisedict' in self.function_parameters[key].keys():
                self.function_parameters[key]['noisedict'] = self.noise_dict

    def get_pulsars(self):
        if len(self.psrs) == 0:
            self.psrs = pickle.load(open(self.pulsar_pickle, 'rb'))
        return self.psrs

    def get_noise_dict(self):
        if len(self.noise_dict.keys()) == 0:
            self.noise_dict = json.load(open(self.noise_dict_json))

            for par in list(self.noise_dict.keys()):
                if 'log10_equad' in par:
                    efac = re.sub('log10_equad', 'efac', par)
                    equad = re.sub('log10_equad', 'log10_t2equad', par)
                    self.noise_dict[equad] = np.log10(10 ** self.noise_dict[par] / self.noise_dict[efac])
                elif 'log10_ecorr' in par and 'basis_ecorr' not in par:
                    ecorr = par.split('_')[0] + '_basis_ecorr_' + '_'.join(par.split('_')[1:])
                    self.noise_dict[ecorr] = self.noise_dict[par]
        return self.noise_dict

    def create_pta_object_from_signals(self):
        """
        Using both signals from pta objects and signals from self.signal_creating_functions
        Create a pta object
        """
        self.load_pickled_pulsars()

        pta_list = self.get_pta_objects()
        signal_collections = [self.get_signal_collection_from_pta_object(pta) for pta in pta_list]
        for key in self.signal_creating_function_keys:
            func = self.functions[key]
            signal_collections.append(func(**self.function_parameters[key]))

        signal_collection = sum(signal_collections[1:], signal_collections[0])

        model_list = []
        # get list of functions to apply to each pulsar
        try:
            function_keys_for_every_pulsar = self.per_pulsar_signal_creating_function_keys['EVERY_PULSAR']
        except KeyError:
            function_keys_for_every_pulsar = []

        for psr in self.psrs:
            # get list of functions to only apply to this pulsar
            try:
                keys_for_this_pulsar = self.per_pulsar_signal_creating_function_keys[psr.name]
            except KeyError:
                keys_for_this_pulsar = []
            print(psr.name, keys_for_this_pulsar)
            keys_for_this_pulsar.extend(function_keys_for_every_pulsar)

            per_pulsar_signal = []
            for key in keys_for_this_pulsar:
                # TODO this will only work if the parameter is named psr or pulsar
                if 'psr' in self.function_parameters[key]:
                    self.function_parameters[key]['psr'] = psr
                if 'pulsar' in self.function_parameters[key]:
                    self.function_parameters[key]['pulsar'] = psr
                # this allows each pulsar to have signals applied to them
                per_pulsar_signal.append(self.functions[key](**self.function_parameters[key]))

            # just sums to signal_collection if additional_models is empty
            model_list.append(sum(per_pulsar_signal, signal_collection)(psr))

        pta = signal_base.PTA(model_list)

        # apply noise dictionary to pta
        pta.set_default_params(self.noise_dict)
        # return pta object
        return pta

    def get_signal_collection_from_pta_object(self, pta):
        """
        Under assumption that same model has been applied to ALL pulsars
        and that there are pulsars inside of this pta,
        get signal collection from this pta object
        """
        return type(pta.pulsarmodels[0])

    def get_pta_objects(self):
        """
        Using pta creating functions specified in config, get list of pta objects
        """
        pta_list = []
        if len(self.psrs) == 0:
            print("Loading pulsars")
            self.load_pickled_pulsars()

        for key in self.pta_creating_function_keys:
            pta_list.append(self.functions[key](psrs=self.psrs, **self.function_parameters[key]))
        return pta_list
