"""
Exploratory Data Analysis Module.

Generates 7 professional visualizations:
1. Sentiment distribution
2. Review length distribution
3. Word count distribution
4. Most common positive words
5. Most common negative words
6. Positive word cloud
7. Negative word cloud

All plots are saved to the outputs/plots/ directory.
"""

import os
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from src.utils import print_header, print_info, print_success, print_subheader

# ─── Style Configuration ─────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#FAFAFA',
    'axes.facecolor': '#FAFAFA',
    'axes.edgecolor': '#CCCCCC',
    'axes.labelsize': 12,
    'axes.titlesize': 15,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'sans-serif',
    'figure.dpi': 150,
})

# Color palette
POSITIVE_COLOR = '#2ECC71'
NEGATIVE_COLOR = '#E74C3C'
ACCENT_1 = '#3498DB'
ACCENT_2 = '#9B59B6'
PALETTE = [NEGATIVE_COLOR, POSITIVE_COLOR]


def run_eda(df, output_dir):
    """Run complete EDA and save all visualizations.

    Args:
        df: DataFrame with 'cleaned_text', 'sentiment', 'cleaned_length',
            and 'word_count' columns.
        output_dir: Path to the outputs directory.
    """
    print_header("STEP 3: EXPLORATORY DATA ANALYSIS")

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # 1. SENTIMENT DISTRIBUTION
    # ═══════════════════════════════════════════════════════════
    print_info("1/7  Sentiment distribution...")

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df['sentiment'].value_counts().sort_index()
    labels = ['Negative (0)', 'Positive (1)']
    bars = ax.bar(labels, counts.values, color=PALETTE,
                  edgecolor='white', linewidth=1.5, width=0.55)

    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 200,
                f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=13)

    ax.set_title('Sentiment Distribution in IMDB Dataset',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel('Number of Reviews', fontsize=12)
    ax.set_xlabel('Sentiment Class', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '01_sentiment_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Saved: 01_sentiment_distribution.png")

    # ═══════════════════════════════════════════════════════════
    # 2. REVIEW LENGTH DISTRIBUTION
    # ═══════════════════════════════════════════════════════════
    print_info("2/7  Review length distribution...")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[df['sentiment'] == 1]['cleaned_length'], bins=60, alpha=0.65,
            label='Positive', color=POSITIVE_COLOR, edgecolor='white')
    ax.hist(df[df['sentiment'] == 0]['cleaned_length'], bins=60, alpha=0.65,
            label='Negative', color=NEGATIVE_COLOR, edgecolor='white')
    ax.set_title('Review Length Distribution by Sentiment',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Review Length (characters)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '02_review_length_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Saved: 02_review_length_distribution.png")

    # ═══════════════════════════════════════════════════════════
    # 3. WORD COUNT DISTRIBUTION
    # ═══════════════════════════════════════════════════════════
    print_info("3/7  Word count distribution...")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[df['sentiment'] == 1]['word_count'], bins=60, alpha=0.65,
            label='Positive', color=POSITIVE_COLOR, edgecolor='white')
    ax.hist(df[df['sentiment'] == 0]['word_count'], bins=60, alpha=0.65,
            label='Negative', color=NEGATIVE_COLOR, edgecolor='white')
    ax.set_title('Word Count Distribution by Sentiment',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Word Count', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '03_word_count_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Saved: 03_word_count_distribution.png")

    # ═══════════════════════════════════════════════════════════
    # 4. MOST COMMON POSITIVE WORDS
    # ═══════════════════════════════════════════════════════════
    print_info("4/7  Most common positive words...")

    pos_text = ' '.join(df[df['sentiment'] == 1]['cleaned_text'].dropna())
    neg_text = ' '.join(df[df['sentiment'] == 0]['cleaned_text'].dropna())

    pos_common = Counter(pos_text.split()).most_common(20)
    neg_common = Counter(neg_text.split()).most_common(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    words, counts = zip(*pos_common)
    y_pos = range(len(words))
    ax.barh(y_pos, counts, color=POSITIVE_COLOR, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=11)
    ax.set_title('Top 20 Most Common Words in Positive Reviews',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '04_common_positive_words.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Saved: 04_common_positive_words.png")

    # ═══════════════════════════════════════════════════════════
    # 5. MOST COMMON NEGATIVE WORDS
    # ═══════════════════════════════════════════════════════════
    print_info("5/7  Most common negative words...")

    fig, ax = plt.subplots(figsize=(10, 7))
    words, counts = zip(*neg_common)
    y_pos = range(len(words))
    ax.barh(y_pos, counts, color=NEGATIVE_COLOR, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=11)
    ax.set_title('Top 20 Most Common Words in Negative Reviews',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '05_common_negative_words.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Saved: 05_common_negative_words.png")

    # ═══════════════════════════════════════════════════════════
    # 6. POSITIVE WORD CLOUD
    # ═══════════════════════════════════════════════════════════
    print_info("6/7  Positive word cloud...")

    fig, ax = plt.subplots(figsize=(12, 6))
    pos_wc = WordCloud(
        width=1200, height=600,
        background_color='white',
        colormap='Greens',
        max_words=150,
        contour_width=1,
        contour_color=POSITIVE_COLOR,
        random_state=42,
    ).generate(pos_text)
    ax.imshow(pos_wc, interpolation='bilinear')
    ax.set_title('Positive Reviews — Word Cloud',
                 fontsize=16, fontweight='bold', pad=15)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '06_positive_wordcloud.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Saved: 06_positive_wordcloud.png")

    # ═══════════════════════════════════════════════════════════
    # 7. NEGATIVE WORD CLOUD
    # ═══════════════════════════════════════════════════════════
    print_info("7/7  Negative word cloud...")

    fig, ax = plt.subplots(figsize=(12, 6))
    neg_wc = WordCloud(
        width=1200, height=600,
        background_color='white',
        colormap='Reds',
        max_words=150,
        contour_width=1,
        contour_color=NEGATIVE_COLOR,
        random_state=42,
    ).generate(neg_text)
    ax.imshow(neg_wc, interpolation='bilinear')
    ax.set_title('Negative Reviews — Word Cloud',
                 fontsize=16, fontweight='bold', pad=15)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '07_negative_wordcloud.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print_success("Saved: 07_negative_wordcloud.png")

    # ─── Summary ─────────────────────────────────────────
    print_success(f"\nAll 7 EDA visualizations saved to: {plots_dir}")
