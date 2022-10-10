import data_class

params = data_class.RunSettings()
params.update_from_file('example.ini')

params.load_pickled_pulsars()
pta = params.create_pta_object()
print(pta)