# ---
# agi_prototype/src/hdc.py

import torch
import torch.nn.functional as F
from config import HD_DIM, DEVICE, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(RANDOM_SEED)

class HDC:
    """Implements core Hyperdimensional Computing operations using bipolar vectors."""
    def __init__(self, dim=HD_DIM):
        self.dim = dim
        self.basis_vectors = {}

    def create_random_hypervector(self):
        """Generates a single random bipolar {-1, 1} hypervector."""
        return torch.randint(0, 2, (self.dim,), device=DEVICE, dtype=torch.float32) * 2 - 1

    def init_basis_vectors(self, num_entities: int, prefix: str = ""):
        """Initializes a set of unique basis vectors for specific items."""
        for i in range(num_entities):
            key = f"{prefix}_{i}"
            if key not in self.basis_vectors:
                self.basis_vectors[key] = self.create_random_hypervector()

    def get_basis_vector(self, key: str):
        """Retrieves a pre-generated basis vector."""
        return self.basis_vectors.get(key)

    def bind(self, hv1, hv2):
        """Binds two hypervectors using element-wise multiplication (XOR)."""
        return hv1 * hv2

    def bundle(self, hvs_list):
        """Bundles a list of hypervectors using element-wise sum and thresholding."""
        if not hvs_list:
            return torch.zeros(self.dim, device=DEVICE)
        bundled = torch.sum(torch.stack(hvs_list), dim=0)
        return torch.sign(bundled).float() # Ensure output is float

    def permute(self, hv, shift=1):
        """Permutes a hypervector using a circular shift."""
        return torch.roll(hv, shifts=shift, dims=0)

    def cosine_similarity(self, hv1, hv2):
        """
        Calculates cosine similarity between two hypervectors or batches of them.
        --- FIX: Replaced torch.dot with F.cosine_similarity for robustness.
        This correctly handles 1D and 2D tensors, resolving the error.
        """
        return F.cosine_similarity(hv1, hv2, dim=-1)
