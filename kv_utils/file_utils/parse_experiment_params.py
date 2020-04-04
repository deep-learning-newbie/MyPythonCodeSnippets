import configparser

class ParseExpParams:
    def __init__(self, cfg_file):
        self.config = configparser.ConfigParser()
        self.config.read(cfg_file)
        ## Hyper params
        self.learning_rate = self.config['Hyper_Params']['Learning_Rate']
        self.batch_size = self.config['Hyper_Params']['Batch_Size']
        self.max_no_epochs = self.config['Hyper_Params']['Max_Epochs']
        self.img_dim = self.config['Hyper_Params']['Img_Dim']
        self.img_channels = self.config['Hyper_Params']['Img_Channels']
        self.optimizer_name = self.config['Hyper_Params']['Optimizer']
        self.base_architecture = self.config['Hyper_Params']['Base_Architecture']
        ## Experiment Params
        self.project_name = self.config['Experiment_Params']['Project_Name']
        self.checkpoints_path = self.config['Experiment_Params']['CheckPoints_Path']
        self.base_data_path = self.config['Experiment_Params']['Base_Data_Path']
        self.unstructured_data_path = self.config['Experiment_Params']['Unstructured_Data_Path']
        self.img_ext = self.config['Experiment_Params']['Img_Ext']
        self.num_classes = self.config['Experiment_Params']['Num_Classes']
        self.validation_frac = self.config['Experiment_Params']['Validation_Fraction']
        self.dataset_shuffle = self.config['Experiment_Params']['Dataset_Shuffle']
        self.no_system_threads = self.config['Experiment_Params']['No_System_Threads']
        self.cyclic_lr_policy = self.config['Experiment_Params']['Cyclic_LR']

        ## init the vars from cfg file
        #self.init_hyperparams()
        #self.init_TrainPaths()

    def get_learning_rate(self):
        return float(self.learning_rate)

    def get_batch_size(self):
        return int(self.batch_size)

    def get_max_no_epochs(self):
        return int(self.max_no_epochs)

    def get_img_dim(self):
        return int(self.img_dim)

    def get_img_channels(self):
        return int(self.img_channels)

    def get_img_ext(self):
        return self.img_ext

    def get_num_classes(self):
        return int(self.num_classes)

    def get_base_architecture(self):
        return self.base_architecture

    def get_optimizer_name(self):
        return self.optimizer_name

    def get_checkpoints_path(self):
        return self.checkpoints_path

    def get_project_name(self):
        return self.project_name

    def get_base_data_path(self):
        return self.base_data_path

    def get_unstructured_data_path(self):
        return self.unstructured_data_path

    def get_validation_fraction(self):
        return float(self.validation_frac)

    def get_dataset_shuffle(self):
        if self.dataset_shuffle == 'True':
            return True
        else:
            return False
    def get_cyclic_lr_policy(self):
        if self.cyclic_lr_policy == 'True':
            return True
        else:
            return False

    def get_no_system_threads(self):
        return int(self.no_system_threads)
# ob = ParseTuneParam('params.cfg')
#
# print(ob.get_learning_rate())
# print(ob.get_batch_size())
# print(ob.get_max_no_epochs())
# print(ob.get_optimizer_name())
# print(ob.get_base_architecture())
# print(ob.get_checkpoints_path())
# print(ob.get_base_data_path())

