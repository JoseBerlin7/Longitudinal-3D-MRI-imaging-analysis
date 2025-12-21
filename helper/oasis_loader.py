# Making the data accessible just by specifying the Subj ID and Visit Num
from torch.utils.data import Dataset
import numpy as np
import os
import nibabel as nib

class OasisDataset(Dataset):
    '''
    This is a custom Dataset class for the OASIS MRI dataset.
    '''
    def __init__(self, df, transform=None):
        self.groups = df.groupby("Subject ID")

        self.index = {}

        for _, row in df.iterrows():
            subj = row["Subject ID"]
            visit = row["Visit"]
            
            if subj not in self.index:
                self.index[subj] = {}

            self.index[subj][visit] = {
                "Path": row["Path"],
                "File": row["Best"],
                "metadata": row.to_dict()
            }

        # for stable list of subject ID's for indexing
        self.subjects = list(self.groups.groups.keys())

        self.transform = transform


    def _load_analyze(self, img_path):
        img = nib.load(img_path)
        img_data = img.get_fdata(dtype=np.float32)

        if img_data.shape[-1]==1:
            img_data = img_data.squeeze(-1)
        return img_data
        
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, subj_id):
        if subj_id not in self.index:
            raise KeyError(f"Subject {subj_id} not found")
        
        return SubjectView(self, subj_id)

class SubjectView:
    def __init__(self, dataset, subj_id):
        self.dataset = dataset
        self.subj_id = subj_id
        self.visits = dataset.index[subj_id]

    def visits_available(self):
        return list(self.visits.keys())
    
    def __getitem__(self, visit_id):
        if visit_id not in self.visits:
            raise KeyError(f"Visit {visit_id} not found")
        
        entry = self.visits[visit_id]

        path = os.path.join(entry["Path"], entry["File"])

        image = self.dataset._load_analyze(path)

        if self.dataset.transform:
            image = self.dataset.transform(image)

        return {
            "subject_id": self.subj_id,
            "visit": visit_id,
            "image": image,
            "metadata": entry["metadata"]
        }
        