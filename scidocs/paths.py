import os


PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.getcwd()))
    
    
class DataPaths:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = os.path.join(PROJECT_ROOT_PATH, 'data')
        self.base_path = base_path
        
        self.cite_val = os.path.join(base_path, 'cite', 'val.qrel')
        self.cite_test = os.path.join(base_path, 'cite', 'test.qrel')
        
        self.cocite_val = os.path.join(base_path, 'cocite', 'val.qrel')
        self.cocite_test = os.path.join(base_path, 'cocite', 'test.qrel')

        self.coread_val = os.path.join(base_path, 'coread', 'val.qrel')
        self.coread_test = os.path.join(base_path, 'coread', 'test.qrel')
        
        self.coview_val = os.path.join(base_path, 'coview', 'val.qrel')
        self.coview_test = os.path.join(base_path, 'coview', 'test.qrel')
        
        self.mag_train = os.path.join(base_path, 'mag', 'train.csv')
        self.mag_val = os.path.join(base_path, 'mag', 'val.csv')
        self.mag_test = os.path.join(base_path, 'mag', 'test.csv')
        
        self.mesh_train = os.path.join(base_path, 'mesh', 'train.csv')
        self.mesh_val = os.path.join(base_path, 'mesh', 'val.csv')
        self.mesh_test = os.path.join(base_path, 'mesh', 'test.csv')
        
        self.recomm_train = os.path.join(base_path, 'recomm', 'train.csv')
        self.recomm_val = os.path.join(base_path, 'recomm', 'val.csv')
        self.recomm_test = os.path.join(base_path, 'recomm', 'test.csv')
        self.recomm_config = os.path.join(base_path, 'recomm', 'train_similar_papers_model.json')
        self.recomm_propensity_scores = os.path.join(base_path, 'recomm', 'propensity_scores.json')
        
        self.paper_metadata_view_cite_read = os.path.join(base_path, 'paper_metadata_view_cite_read.json')
        self.paper_metadata_mag_mesh = os.path.join(base_path, 'paper_metadata_mag_mesh.json')
        self.paper_metadata_recomm = os.path.join(base_path, 'paper_metadata_recomm.json')
