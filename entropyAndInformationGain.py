import pandas as pd
from math import log2

# Load the dataset 
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny',
                'Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool',
                    'Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High',
                 'Normal','Normal','Normal','High','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak',
             'Weak','Weak','Strong','Strong','Weak','Strong'],
    'Play Tennis': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes',
                    'Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

#  FUNCTIONS 
def entropy(labels):
    """Calculate entropy of a label series"""
    value_counts = labels.value_counts(normalize=True)
    return -sum(p * log2(p) for p in value_counts if p > 0)

def information_gain(df, feature, target='Play Tennis'):
    """Calculate Information Gain for a feature"""
    total_entropy = entropy(df[target])
    weighted_entropy = 0.0
    
    for value in df[feature].unique():
        subset = df[df[feature] == value]
        prob = len(subset) / len(df)
        weighted_entropy += prob * entropy(subset[target])
    
    return total_entropy - weighted_entropy

#  CALCULATIONS 
total_ent = entropy(df['Play Tennis'])
print(f"Total Entropy: {total_ent:.4f}\n")

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

for feat in features:
    ig = information_gain(df, feat)
    print(f"Information Gain ({feat:12}): {ig:.4f}")

# Find best feature
best_feature = max(features, key=lambda f: information_gain(df, f))
print(f"\nBest feature to split on: **{best_feature}**")