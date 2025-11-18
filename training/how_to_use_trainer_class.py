# 1. Augmentation 없이 학습
trainer = Trainer(model, optimizer, criterion, augmentation=None)

# 2. 단일 augmentation
from augment.torch_augmentations import JITTERING
aug = JITTERING(prob=0.5, magnitude=5)
trainer = Trainer(model, optimizer, criterion, augmentation=aug)

# 3. 여러 augmentation (PBA policy)
policy = searcher.population[0]
aug_pipeline = policy.get_transforms()  # nn.Sequential
trainer = Trainer(model, optimizer, criterion, augmentation=aug_pipeline)

# 4. 중간에 바꾸기 (PBA search용)
trainer.set_augmentation(new_policy.get_transforms())