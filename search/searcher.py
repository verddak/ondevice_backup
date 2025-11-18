# training/pba_search.py (updated)
from __future__ import annotations
import random
from typing import List, Tuple, Type, Optional, Dict, Any
from pathlib import Path
import numpy as np
import torch
import json
from datetime import datetime


class AugmentationPolicy:
    """
    Single augmentation policy consisting of two operations.
    Each operation has (augmentation_class, probability, magnitude).
    """
    def __init__(
        self,
        op1: Tuple[Type, int, int],
        op2: Tuple[Type, int, int]
    ):
        """
        Args:
            op1: (Augmentation class, prob (0-10), mag (0-9))
            op2: (Augmentation class, prob (0-10), mag (0-9))
        """
        self.op1 = op1  # (AugClass, prob, mag)
        self.op2 = op2
        self.fitness = 0.0  # validation accuracy
        
    def get_transforms(self) -> torch.nn.Sequential:
        """
        Convert policy to actual transform pipeline.
        Returns nn.Sequential that can be applied to batches.
        """
        transforms = []
        
        for aug_class, prob, mag in [self.op1, self.op2]:
            if prob > 0:  # prob을 0~1 범위로 변환하여 사용
                transforms.append(aug_class(prob=prob/10, magnitude=mag))
        
        # 빈 리스트면 Identity 반환
        if not transforms:
            return torch.nn.Identity()
        
        return torch.nn.Sequential(*transforms)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary for logging"""
        return {
            'op1': {
                'method': self.op1[0].__name__,
                'prob': self.op1[1],
                'mag': self.op1[2]
            },
            'op2': {
                'method': self.op2[0].__name__,
                'prob': self.op2[1],
                'mag': self.op2[2]
            },
            'fitness': float(self.fitness)
        }
    
    def __repr__(self) -> str:
        return (f"Policy(op1={self.op1[0].__name__},"
                f"prob={self.op1[1]},mag={self.op1[2]} | "
                f"op2={self.op2[0].__name__},"
                f"prob={self.op2[1]},mag={self.op2[2]} | "
                f"fitness={self.fitness:.2f})")


class PBASearcher:
    """
    Population-Based Augmentation (PBA) searcher.
    
    Evolutionary algorithm for finding optimal augmentation policies:
    1. Initialize population of N policies
    2. For each generation:
        - Evaluate all policies (train model with each policy)
        - Select top-K elites
        - Replace worst-K with copies of elites
        - Mutate non-elite policies
    """
    
    # Forbidden combinations
    FORBIDDEN_PAIRS = [
        ('TIME_WARPING', 'WINDOW_WARPING'),
        ('WINDOW_WARPING', 'TIME_WARPING')
    ]
    
    def __init__(
        self,
        augmentation_classes: List[Type],
        population_size: int = 16,
        n_elites: int = 3,
        mutation_prob_method: float = 0.2,
        mutation_prob_param: float = 0.2,
        param_change_amount: List[int] = [0, 1, 2, 3],
        seed: Optional[int] = None
    ):
        """
        Args:
            augmentation_classes: List of augmentation classes (not instances!)
            population_size: Number of policies in population
            n_elites: Number of top policies to preserve
            mutation_prob_method: Probability of changing augmentation method
            mutation_prob_param: Probability of completely random parameter reset
            param_change_amount: Possible amounts to change parameters by
            seed: Random seed for reproducibility
        """
        self.augmentation_classes = augmentation_classes
        self.population_size = population_size
        self.n_elites = n_elites
        self.mutation_prob_method = mutation_prob_method
        self.mutation_prob_param = mutation_prob_param
        self.param_change_amount = param_change_amount
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize population
        self.population: List[AugmentationPolicy] = []
        self._initialize_population()
        
        # Search history
        self.history: List[Dict[str, Any]] = []
        
    def _initialize_population(self) -> None:
        """Initialize population with random policies"""
        for _ in range(self.population_size):
            policy = self._create_random_policy()
            self.population.append(policy)
    
    def _create_random_policy(self) -> AugmentationPolicy:
        """Create a single random policy"""
        while True:
            # Sample two augmentation classes
            aug_classes = random.sample(self.augmentation_classes, 2)
            
            # Check if combination is forbidden
            if not self._is_forbidden_combination(aug_classes[0], aug_classes[1]):
                break
        
        # Random parameters
        prob1 = random.randint(0, 10)
        mag1 = random.randint(0, 9)
        prob2 = random.randint(0, 10)
        mag2 = random.randint(0, 9) # random.randint(0, 9)는 0,1,2...9 중 하나를 반환한다
        
        op1 = (aug_classes[0], prob1, mag1)
        op2 = (aug_classes[1], prob2, mag2)
        
        return AugmentationPolicy(op1, op2)
    
    def _is_forbidden_combination(self, class1: Type, class2: Type) -> bool:
        """Check if two augmentation classes form a forbidden combination"""
        name1 = class1.__name__
        name2 = class2.__name__
        
        return (name1, name2) in self.FORBIDDEN_PAIRS or name1 == name2
    
    def mutate_policy(self, policy: AugmentationPolicy) -> AugmentationPolicy:
        """
        Mutate a policy by changing methods and/or parameters.
        
        Strategy:
        - Each parameter has 20% chance of complete random reset
        - Otherwise, change by ±[0,1,2,3] with equal probability
        - Each method has 20% chance of being replaced
        """
        # Extract current values
        class1, prob1, mag1 = policy.op1
        class2, prob2, mag2 = policy.op2
        
        # Mutate parameters
        new_params = []
        for i, param in enumerate([prob1, mag1, prob2, mag2]):
            if random.random() < self.mutation_prob_param:
                # Complete random reset
                if i % 2 == 0:  # prob (0-10)
                    new_params.append(random.randint(0, 10))
                else:  # mag (0-9)
                    new_params.append(random.randint(0, 9))
            else:
                # Small change
                amt = int(np.random.choice(
                    self.param_change_amount, 
                    p=[0.25] * len(self.param_change_amount)
                ))
                
                if random.random() < 0.5:
                    new_val = max(0, param - amt)
                else:
                    if i % 2 == 0:  # prob (0-10)
                        new_val = min(10, param + amt)
                    else:  # mag (0-9)
                        new_val = min(9, param + amt)
                
                new_params.append(new_val)
        
        # Mutate methods
        new_class1 = class1
        new_class2 = class2
        
        if random.random() < self.mutation_prob_method:
            new_class1 = random.choice(self.augmentation_classes)
        
        if random.random() < self.mutation_prob_method:
            new_class2 = random.choice(self.augmentation_classes)
        
        # Ensure no forbidden combination
        while self._is_forbidden_combination(new_class1, new_class2):
            if random.random() < 0.5:
                new_class1 = random.choice(self.augmentation_classes)
            else:
                new_class2 = random.choice(self.augmentation_classes)
        
        # Create new policy
        op1 = (new_class1, new_params[0], new_params[1])
        op2 = (new_class2, new_params[2], new_params[3])
        
        return AugmentationPolicy(op1, op2)
    
    def evolve(self, fitness_scores: List[float]) -> None:
        """
        Evolve population based on fitness scores.
        
        Args:
            fitness_scores: List of validation accuracies for each policy
        """
        assert len(fitness_scores) == self.population_size, \
            f"Expected {self.population_size} fitness scores, got {len(fitness_scores)}"
        
        # Update fitness
        for policy, fitness in zip(self.population, fitness_scores):
            policy.fitness = fitness
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        
        # Log generation statistics
        generation_info = {
            'best_fitness': float(fitness_scores[sorted_indices[0]]),
            'worst_fitness': float(fitness_scores[sorted_indices[-1]]),
            'mean_fitness': float(np.mean(fitness_scores)),
            'std_fitness': float(np.std(fitness_scores)),
            'best_policy': sorted_population[0].to_dict()
        }
        self.history.append(generation_info)
        
        # Select elites (top n_elites)
        elites = sorted_population[:self.n_elites]
        
        # Create new population
        new_population = []
        
        # 1. Keep elites (indices 0 ~ n_elites-1)
        for elite in elites:
            new_population.append(elite)
        
        # 2. Mutate middle policies (indices n_elites ~ population_size - n_elites - 1)
        for i in range(self.n_elites, self.population_size - self.n_elites):
            mutated = self.mutate_policy(sorted_population[i])
            new_population.append(mutated)
        
        # 3. Replace worst with MUTATED elite copies
        for i in range(self.n_elites):
            elite = elites[i]
            elite_copy = AugmentationPolicy(elite.op1, elite.op2)
            # Elite 복사본을 mutate!
            mutated_elite_copy = self.mutate_policy(elite_copy)
            new_population.append(mutated_elite_copy)
        
        self.population = new_population

    
    def get_population(self) -> List[AugmentationPolicy]:
        """Get current population"""
        return self.population
    
    def get_best_policy(self) -> AugmentationPolicy:
        """Get the best policy from current population"""
        return max(self.population, key=lambda p: p.fitness)
    
    def save_history(self, save_path: Path) -> None:
        """Save search history to JSON"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        history_dict = {
            'search_config': {
                'population_size': self.population_size,
                'n_elites': self.n_elites,
                'mutation_prob_method': self.mutation_prob_method,
                'mutation_prob_param': self.mutation_prob_param,
                'seed': self.seed
            },
            'history': self.history,
            'final_population': [p.to_dict() for p in self.population]
        }
        
        with open(save_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
    
    def __repr__(self) -> str:
        return (f"PBASearcher(population_size={self.population_size}, "
                f"n_elites={self.n_elites}, "
                f"generations_run={len(self.history)})")