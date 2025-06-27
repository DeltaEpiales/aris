# src/memory.py

import torch
from collections import deque
import config

class HippocampalModule:
    def __init__(self, hdc_system, data_encoder):
        self.hdc = hdc_system
        self.data_encoder = data_encoder
        self.episodic_memory = deque(maxlen=config.HIPPOCAMPUS_CAPACITY)
        
        # --- FIX: Pre-generate basis vectors for roles AND values ---
        self.hdc.init_basis_vectors(len(config.CONTEXT_PERMUTATION_SHIFTS), prefix="role")
        self.hdc.init_basis_vectors(1, prefix="time_value_base")
        self.hdc.init_basis_vectors(1, prefix="reward_value_base")


    def encode_pattern(self, spike_pattern, context_data):
        """
        Encodes a new episodic memory using a more structured, "quantum-inspired"
        binding process. It creates explicit role-filler pairs for context.
        """
        core_pattern_hv = self.data_encoder.encode_spike_pattern_to_hypervector(spike_pattern)
        if torch.all(core_pattern_hv == 0):
            return

        context_chunks = []

        # --- Create a "Time" context chunk: bind(ROLE_time, VALUE_time) ---
        if 'time_val' in context_data:
            # Role HV for "time"
            role_time_hv = self.hdc.get_basis_vector("role_0") # Assumes 'time' is the first context role
            # Value HV for the specific time, created by permuting a base value vector
            time_value_base_hv = self.hdc.get_basis_vector("time_value_base_0")
            value_time_hv = self.hdc.permute(time_value_base_hv, shift=int(context_data['time_val']))
            # Bind them to create the chunk
            time_chunk = self.hdc.bind(role_time_hv, value_time_hv)
            context_chunks.append(time_chunk)

        # --- Create a "Reward" context chunk: bind(ROLE_reward, VALUE_reward) ---
        if 'reward_val' in context_data:
            # Role HV for "reward"
            role_reward_hv = self.hdc.get_basis_vector("role_1") # Assumes 'reward' is the second context role
            # Value HV for the specific reward, created by permuting a base value vector
            reward_value_base_hv = self.hdc.get_basis_vector("reward_value_base_0")
            # Quantize reward to get an integer shift amount
            reward_shift = int(context_data['reward_val'] * 10)
            value_reward_hv = self.hdc.permute(reward_value_base_hv, shift=reward_shift)
            # Bind them to create the chunk
            reward_chunk = self.hdc.bind(role_reward_hv, value_reward_hv)
            context_chunks.append(reward_chunk)
            
        # --- Create the final episodic memory ---
        if context_chunks:
            # Bundle all context chunks into a single composite context vector
            composite_context_hv = self.hdc.bundle(context_chunks)
            # Bind the core pattern with the composite context
            episodic_hv = self.hdc.bind(core_pattern_hv, composite_context_hv)
        else:
            # If no context, the memory is just the core pattern
            episodic_hv = core_pattern_hv
        
        episode = {
            'hv': episodic_hv,
            'core_hv': core_pattern_hv,
            'reward': context_data.get('reward_val', 0.1),
            'recency': 1.0
        }
        self.episodic_memory.append(episode)

    def trigger_replay(self, neocortex, neuromodulators):
        if not self.episodic_memory:
            return
        
        priorities = self._calculate_priorities(neocortex, neuromodulators)
        if torch.sum(priorities) <= 0:
            return
            
        probabilities = priorities / torch.sum(priorities)
        num_to_replay = min(config.REPLAY_BATCH_SIZE, len(self.episodic_memory))
        if probabilities.numel() > 0 and torch.any(probabilities > 0):
            indices_to_replay = torch.multinomial(probabilities, num_samples=num_to_replay, replacement=False)

            for idx in indices_to_replay:
                episode = self.episodic_memory[idx]
                neocortex.consolidate_pattern(episode['core_hv'])

    def _calculate_priorities(self, neocortex, neuromodulators):
        if not self.episodic_memory:
            return torch.tensor([], device=config.DEVICE)

        priorities = []
        serotonin_level = neuromodulators.serotonin_level.item()
        for episode in self.episodic_memory:
            episode['recency'] *= config.RECENCY_DECAY
            similarity = neocortex.get_similarity_to_memory(episode['core_hv'])
            novelty = 1.0 - similarity
            
            total_priority = (episode['recency'] * config.RECENCY_WEIGHT +
                              episode['reward'] * config.REWARD_WEIGHT +
                              novelty * (config.NOVELTY_WEIGHT + serotonin_level * config.SEROTONIN_INFLUENCE))
            priorities.append(total_priority)
        
        return torch.clamp(torch.tensor(priorities, device=config.DEVICE), min=0)

    def reset(self):
        self.episodic_memory.clear()

class NeocorticalModule:
    def __init__(self, hdc_system):
        self.hdc = hdc_system
        self.semantic_memory_bank = []

    def consolidate_pattern(self, replayed_hv):
        if not self.semantic_memory_bank:
            self.semantic_memory_bank.append(replayed_hv)
            return

        similarities = self.get_similarity_to_memory(replayed_hv, return_tensor=True)
        if similarities.numel() > 0:
            max_sim, best_match_idx = torch.max(similarities, dim=0)

            if max_sim >= config.NEOCORTEX_CONSOLIDATION_THRESHOLD:
                existing_concept = self.semantic_memory_bank[best_match_idx]
                updated_concept = self.hdc.bundle([existing_concept, replayed_hv])
                self.semantic_memory_bank[best_match_idx] = updated_concept
            else:
                self.semantic_memory_bank.append(replayed_hv)
        else:
             self.semantic_memory_bank.append(replayed_hv)


    def get_similarity_to_memory(self, query_hv, return_tensor=False):
        if not self.semantic_memory_bank:
            return 0.0 if not return_tensor else torch.tensor([], device=config.DEVICE)

        concepts_tensor = torch.stack(self.semantic_memory_bank)
        similarities = self.hdc.cosine_similarity(query_hv.unsqueeze(0), concepts_tensor)

        if return_tensor:
            return similarities.squeeze(0)
        else:
            return torch.max(similarities).item() if similarities.numel() > 0 else 0.0
        
    def reset(self):
        self.semantic_memory_bank = []
