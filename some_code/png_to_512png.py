from PIL import Image
from glob import glob

results = {}
results[0] = 100.0 * 0.7549
results[1] = 100.0 * 0.7129
results[2] = 100.0 * 0.7129
results[3] = 100.0 * 0.6931
results[4] = 100.0 * 0.7426


print(f'K-FOLD CROSS VALIDATION RESULTS FOR {5} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value:.3f} %')
    sum += value
print(f'Average: {sum/len(results.items())} %')