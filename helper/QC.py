import os
import glob
from tqdm import tqdm
import numpy as np
import nibabel as nib

class Quality_Check:
    def __init__(self, df):
        self.df = df
    
    def load_nifti(self, path):
        img = nib.load(path).get_fdata(dtype=np.float32)

        img = (img - np.mean(img)) / (np.std(img) + 1e-8)

        img = img.squeeze(-1)

        return img

    # Simple signal to noise ratio calc
    def compute_snr(self, img):
        signal = np.mean(img)        
        noise = np.std(img)
        return np.abs(signal) / (noise + 1e-8)

    # Approximate edges using finite differences
    def compute_sharpness(self, img):
        gx = img[1: ,: ,:] - img[:-1 ,: ,:]
        gy = img[:, 1:, :] - img[:, :-1, :]
        gz = img[:, :, 1:] - img[:, :, :-1]

        grad_mag = (
            gx[:, :-1, :-1]**2 +
            gy[:-1, :, :-1]**2 +
            gz[:-1, :-1, :]**2
        )

        return np.mean(grad_mag)

    # Manual entropy calc
    def compute_entropy(self, img, bins=256):
        hist, _ = np.histogram(img, bins=bins, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def get_QS(self, img):
        snr = self.compute_snr(img)
        sharpness = self.compute_sharpness(img)
        entropy = self.compute_entropy(img)

        entropy_penality = 1/ (entropy + 1e-8)

        score = (
            0.5 * snr +
            0.4 * sharpness +
            0.1 * entropy_penality
        )

        return score, {"snr": snr.item(), "sharpness": sharpness.item(), "entropy": entropy.item()}
    
    def pick_best(self):
        results = {}

        for row in tqdm(self.df.itertuples(), total=len(self.df), desc="Processing Visits.."):
            dir = row.Path
            if not os.path.exists(dir):
                raise FileNotFoundError(f"Path {dir} does not exist.")
            img_files = sorted(glob.glob(os.path.join(dir, "*.img")))

            best_score = -np.inf
            best_scan = None
            report = {}
            
            for path in img_files:
                img = self.load_nifti(path)
                score, metrics = self.get_QS(img)
                
                report[path] = metrics

                if score > best_score:
                    best_score = score
                    best_scan = path
                    
            results[row.Index] = {"Best_scan": best_scan, "metrices": report}
            self.df.loc[row.Index, "Best"] = best_scan.split("/")[-1]

        return self.df, results