from astrolab.data.manager import DataManager, ModeDataManager

dm: DataManager = DataManager.instance()
mdm: ModeDataManager = dm.mode_data_manager

mdm.model_dims = 16
mdm.subsample = 5

mdm.prepare_inputs()