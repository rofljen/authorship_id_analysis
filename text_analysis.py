# Imports
import os
import random
import shutil
import re
from collections import defaultdict
import nltk
import pandas as pd     
import numpy as np
import matplotlib.pyplot as plt
from nltk import word_tokenize, pos_tag, FreqDist
import sys
import importlib.util
from nltk.corpus import stopwords
from string import punctuation
import contractions
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Load external Zipf module
file_path = "C:/Users/Jennifer/Downloads/zipf.py"
module = "zipf"
spec = importlib.util.spec_from_file_location(module, file_path)
zipf = importlib.util.module_from_spec(spec)
sys.modules[module] = zipf
spec.loader.exec_module(zipf)

# Download required NLTK data
nltk.data.path.append('C:/Users/Jennifer/AppData/Roaming/nltk_data')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('stopwords')

# === Parameters ===
flat_chunk_dir = "C:/Users/Jennifer/Downloads/essay_chunks"
output_dir = "C:/Users/Jennifer/Documents/balanced_chunk_dataset"
target_range = (100, 200)
chunks_per_author = 6
authors_to_include = 4

# === Load valid text chunks ===
def get_valid_chunks(base_dir, word_range):
    author_chunks = defaultdict(list)
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                try:
                    with open(path, encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(path, encoding="latin-1") as f:
                        content = f.read()
                wc = len(content.split())
                if word_range[0] <= wc <= word_range[1]:
                    rel_path = os.path.relpath(path, base_dir)
                    author = rel_path.split(os.sep)[0]
                    author_chunks[author].append((path, wc))
    return author_chunks

chunks = get_valid_chunks(flat_chunk_dir, target_range)

# === Sample N chunks per author ===
def sample_chunks(author_chunks, output_dir, per_author, num_authors):
    eligible = [a for a, chunks in author_chunks.items() if len(chunks) >= per_author]
    selected = random.sample(eligible, min(num_authors, len(eligible)))

    os.makedirs(output_dir, exist_ok=True)
    texts = []
    metadata = []

    for author in selected:
        samples = random.sample(author_chunks[author], per_author)
        for i, (src_path, wc) in enumerate(samples):
            filename = f"{author}_{os.path.basename(src_path)}"
            dst_path = os.path.join(output_dir, filename)
            shutil.copy(src_path, dst_path)
            with open(src_path, encoding="utf-8") as f:
                text = f.read()
            texts.append(text)
            metadata.append({"author": author, "chunk_id": i + 1})
            print(f"{author}: ✅ {os.path.basename(src_path)} ({wc} words)")
    return texts, metadata

texts, metadata = sample_chunks(chunks, output_dir, chunks_per_author, authors_to_include)

# === POS Analysis ===
def pos_analysis(texts, metadata):
    records = []
    for i, text in enumerate(texts):
        tokens = word_tokenize(text)
        tokens = [re.sub(r"[^\w\s]", "", t) for t in tokens if re.sub(r"[^\w\s]", "", t)]
        tags = pos_tag(tokens, tagset="universal")
        for word, tag in tags:
            records.append({
                "author": metadata[i]["author"],
                "chunk_id": metadata[i]["chunk_id"],
                "word": word,
                "pos": tag
            })
    return pd.DataFrame(records)

df = pos_analysis(texts, metadata)

# === Frequency + Lexical Diversity ===
stop_words = set(stopwords.words('english'))

def clean_tokens(tokens):
    tokens = [contractions.fix(w) for w in tokens]
    return [w for w in tokens if w.lower() not in stop_words and w not in punctuation]

def freq_analysis_from_texts(texts, metadata):
    freq_dists = {}
    unique_counts = {}
    total_counts = {}
    lexical_diversity = {}

    for i, text in enumerate(texts):
        tokens = word_tokenize(text.lower())
        name = f"{metadata[i]['author']}{metadata[i]['chunk_id']}"
        dist = FreqDist(tokens)

        freq_dists[name] = dist
        unique_counts[name] = len(set(tokens))
        total_counts[name] = len(tokens)
        lexical_diversity[name] = (len(set(tokens)) / len(tokens)) * 100

    return freq_dists, unique_counts, total_counts, lexical_diversity

def ave_lexical_diversity(lexical_diversity):
    grouped = defaultdict(list)
    for name, value in lexical_diversity.items():
        author = ''.join(filter(str.isalpha, name))
        grouped[author].append(value)

    return {author.upper(): np.mean(values) for author, values in grouped.items()}

# === Clean token lists ===
cleaned_texts = []
for text in texts:
    tokens = word_tokenize(text)
    cleaned = clean_tokens(tokens)
    cleaned_texts.append(cleaned)

# === Plotting function ===
def show_stats(df, unique_counts, clean_ave_div_author, show_labels=True, show_trend=True):
    def add_labels(ax, show_labels):
        if show_labels:
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{height:.1f}', (p.get_x() + p.get_width() / 2, height),
                            ha='center', va='bottom', fontsize=8)

    # POS counts
    pos_counts = df.groupby(['author', 'chunk_id'])['pos'].value_counts().rename('count').reset_index()
    avg_pos = pos_counts.groupby(['pos', 'author'])['count'].mean().reset_index(name='average_count')
    print("\nAverage POS counts per author:\n", avg_pos)

    pivot = pd.pivot_table(avg_pos, values="average_count", index="author", columns="pos", aggfunc=np.mean)
    ax = pivot.plot(kind="bar", figsize=(8, 6))
    add_labels(ax, show_labels)
    ax.set_xlabel("Author")
    ax.set_ylabel("Average POS Count")
    ax.set_title("Average POS Counts per Author")
    plt.tight_layout()
    plt.show()
    ax.get_figure().savefig("pos_average_barplot.png")

    # Unique words
    grouped = defaultdict(list)
    for name, count in unique_counts.items():
        author = re.match(r"[a-zA-Z_]+", name).group().rstrip('_').lower()
        grouped[author].append(count)

    ave_unique_words = {author.upper(): np.mean(counts) for author, counts in grouped.items()}
    unique_df = pd.DataFrame(list(ave_unique_words.items()), columns=["author", "average_unique_words"])

    print("\nAverage unique words per author:\n", unique_df)

    ax = unique_df.set_index("author").plot(kind="bar", legend=False, figsize=(8, 6))
    add_labels(ax, show_labels)
    ax.set_xlabel("Author")
    ax.set_ylabel("Average Unique Word Count")
    ax.set_title("Average Unique Words per Author")
    plt.tight_layout()
    plt.show()
    ax.get_figure().savefig("unique_words_barplot.png")

    # Lexical diversity
    ave_div_df = pd.DataFrame(list(clean_ave_div_author.items()), columns=["author", "average_lexical_diversity"])
    print("\nAverage lexical diversity per author:\n", ave_div_df)

    ax = ave_div_df.set_index("author").plot(kind="bar", legend=False, figsize=(8, 6))
    add_labels(ax, show_labels)
    ax.set_xlabel("Author")
    ax.set_ylabel("Average Lexical Diversity")
    ax.set_title("Average Lexical Diversity per Author")
    plt.tight_layout()
    plt.show()
    ax.get_figure().savefig("lexical_diversity_barplot.png")

# === Run Analysis ===
freq_dists, unique_counts, total_counts, lexical_diversity = freq_analysis_from_texts(texts, metadata)
clean_ave_div_author = ave_lexical_diversity(lexical_diversity)
show_stats(df, unique_counts, clean_ave_div_author)
