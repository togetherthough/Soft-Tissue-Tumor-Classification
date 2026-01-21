import pandas as pd

df = pd.read_csv('sheet.csv')

output = []
output.append('=' * 60)
output.append('DATASET STRUCTURE ANALYSIS')
output.append('=' * 60)

print(f'\nTotal studies: {len(df)}')

print('\n--- Tumor Types (Dataset column) ---')
for dataset, count in df['Dataset'].value_counts().items():
    print(f'{dataset:15s}: {count:4d} studies')

print('\n--- Binary Labels (Diagnosis_binary column) ---')
for label, count in df['Diagnosis_binary'].value_counts().items():
    print(f'Label {label}: {count:4d} studies')

print('\n--- Cross-tabulation: Dataset vs Binary Label ---')
cross = pd.crosstab(df['Dataset'], df['Diagnosis_binary'], margins=True)
print(cross)

print('\n--- What does Diagnosis_binary mean? ---')
for dataset in df['Dataset'].unique():
    subset = df[df['Dataset'] == dataset]
    print(f'\n{dataset}:')
    diag_counts = subset.groupby(['Diagnosis_binary', 'Diagnosis']).size()
    for (binary, diag), count in diag_counts.items():
        print(f'  Binary={binary} → {diag} ({count} cases)')

print('\n' + '=' * 60)
print('KEY FINDINGS')
print('=' * 60)
print('\n❌ This is NOT "tumor vs. no-tumor" detection')
print('   → ALL cases are tumors (no healthy controls)')
print('\n❌ Binary label meaning is DATASET-SPECIFIC:')
print('   → CRLM: 0=rHGP (responding), 1=dHGP (not responding)')
print('   → Desmoid: 0=non-DTF (sarcomas), 1=DTF (desmoid)')
print('   → GIST: 0=non-GIST (mimics), 1=GIST')
print('   → Lipo: 0=Lipoma (benign), 1=WDLPS (malignant)')
print('\n✅ CONCLUSION:')
print('   The suggested binary MIL approach does NOT apply here.')
print('   Your data has multiple tumor types with different meanings.')
print('   Keep the hierarchical design OR use multi-task learning.')
