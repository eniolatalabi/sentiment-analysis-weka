import pandas as pd
import re

print("="*60)
print("SENTIMENT ANALYSIS - DATA PREPARATION")
print("="*60)

print("\n1. Loading dataset...")
df = pd.read_csv('data/Reviews.csv')
print(f"   Total reviews: {len(df):,}")

print("\n2. Taking first 1500 reviews...")
df = df.head(1500)

print("\n3. Cleaning text...")
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(text.split())
    return text.lower()

df['CleanText'] = df['Text'].apply(clean_text)
df = df[df['CleanText'].str.len() > 10]

print("\n4. Creating sentiment labels...")
df = df[df['Score'] != 3]
df['Sentiment'] = df['Score'].apply(lambda x: 'positive' if x >= 4 else 'negative')

df_final = df[['CleanText', 'Sentiment']].copy()
df_final.columns = ['text', 'sentiment']

print("\n5. Balancing dataset (500 positive, 500 negative)...")
positive = df_final[df_final['sentiment'] == 'positive'].head(500)
negative = df_final[df_final['sentiment'] == 'negative'].head(500)
df_final = pd.concat([positive, negative]).sample(frac=1).reset_index(drop=True)

print(f"\n6. Final dataset:")
print(f"   Total: {len(df_final)}")
print(f"   Positive: {len(df_final[df_final['sentiment']=='positive'])}")
print(f"   Negative: {len(df_final[df_final['sentiment']=='negative'])}")

print("\n7. Saving files...")
df_final.to_csv('data/amazon_sentiment_clean.csv', index=False)
print("   ✓ Saved: data/amazon_sentiment_clean.csv")

sample = df_final.head(10)
sample.to_csv('results/sample_reviews.csv', index=False)

print("   ✓ Saved: results/sample_reviews.csv")

print("\n" + "="*60)
print("COMPLETE! Ready for Weka.")
print("="*60)