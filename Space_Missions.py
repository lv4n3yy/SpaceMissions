import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import os

# PyCharm-friendly settings
plt.style.use('default')
sns.set_palette("husl")
pd.options.display.float_format = '{:,.2f}'.format

print("🚀 Loading Space Mission Launches Dataset...")
df = pd.read_csv('mission_launches.csv')

print(f"Dataset shape: {df.shape}")
print("\nDataset columns:", df.columns.tolist())

# ========== DATA CLEANING (Fixed) ==========
print("\n🧹 Cleaning data...")

# Drop unnamed index columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

print("Columns after dropping unnamed:", df.columns.tolist())
print(f"Missing values before cleaning: {df.isna().sum().sum()}")

# Convert Price to numeric (handle strings)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Drop rows with missing Price (keep only priced missions for analysis)
df_clean = df.dropna(subset=['Price']).copy()

print(f"Cleaned shape (with prices): {df_clean.shape}")

# ========== DATA EXPLORATION ==========
print("\n📊 Basic Statistics:")
print(df_clean[['Price']].describe())

print("\n📈 Top Organizations by Launch Count (priced missions):")
org_counts = df_clean['Organisation'].value_counts().head(10)
print(org_counts)

print("\n📍 Top Locations by Launch Count:")
loc_counts = df_clean['Location'].value_counts().head(10)
print(loc_counts)

# Convert Date column to datetime
df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
df_clean['Year'] = df_clean['Date'].dt.year

# ========== VISUALIZATIONS (Fixed column references) ==========
# Plot 1: Launches by Organization
plt.figure(figsize=(12, 6))
org_counts.plot(kind='bar')
plt.title('Top 10 Organizations by Priced Launch Count')
plt.xlabel('Organization')
plt.ylabel('Number of Launches')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('launches_by_org.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Success Rate by Organization (FIXED)
success_rate = df_clean.groupby('Organisation')['Mission_Status'].apply(
    lambda x: (x.str.contains('Success', case=False, na=False)).mean()
).sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
success_rate.plot(kind='bar', color='green')
plt.title('Success Rate by Top Organizations')
plt.xlabel('Organization')
plt.ylabel('Success Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('success_rate_by_org.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Launches Over Time
plt.figure(figsize=(12, 6))
yearly_launches = df_clean.groupby('Year').size()
yearly_launches.plot(kind='line', marker='o', linewidth=2, markersize=8)
plt.title('Priced Launches Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Launches')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('launches_over_time.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 4: Price Distribution
plt.figure(figsize=(10, 6))
plt.hist(df_clean['Price'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
plt.title('Distribution of Launch Prices')
plt.xlabel('Price (millions USD)')
plt.ylabel('Frequency')
plt.axvline(df_clean['Price'].mean(), color='red', linestyle='--', label=f'Mean: ${df_clean["Price"].mean():.1f}M')
plt.legend()
plt.tight_layout()
plt.savefig('price_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 5: Average Price by Organization
avg_price_org = df_clean.groupby('Organisation')['Price'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
avg_price_org.head(10).plot(kind='bar', color='orange')
plt.title('Average Launch Price by Organization (Top 10)')
plt.xlabel('Organization')
plt.ylabel('Average Price (millions USD)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('avg_price_by_org.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE!")
print("="*60)
print("Generated graphs:")
print("- launches_by_org.png")
print("- success_rate_by_org.png")
print("- launches_over_time.png")
print("- price_distribution.png")
print("- avg_price_by_org.png")
print("\n💰 Dataset Insights:")
print(f"Average launch price: ${df_clean['Price'].mean():.1f}M")
print(f"Most expensive org avg: ${df_clean.groupby('Organisation')['Price'].mean().max():.1f}M")
print(f"SpaceX avg price: ${df_clean[df_clean['Organisation']=='SpaceX']['Price'].mean():.1f}M")
