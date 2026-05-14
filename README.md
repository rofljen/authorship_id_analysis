# Authorship Identification Analysis

This project explores authorship identification by comparing stylistic features across essays by four authors: David Foster Wallace, Joan Didion, Zadie Smith, and John Jeremiah Sullivan. The analysis uses Python and spaCy to examine grammatical and lexical patterns that may help distinguish one author’s style from another.

## Goals

- Compare stylistic features across four essayists.
- Analyze POS distribution, lexical diversity, unique word count, average word length, and hapax legomena.
- Generate visualizations that summarize differences across authors.

## Data

The corpus was built from publicly available online essays. Texts were downloaded locally, cleaned, and organized by author before analysis.

## Methodology

Each essay was split into three-paragraph chunks. Chunks between 100 and 200 words were retained so the samples would be comparable in length while still preserving enough context for stylistic analysis.

The text was then processed with spaCy for tokenization and POS tagging. From those processed chunks, I calculated:

- Average POS usage
- Average unique word count
- Lexical diversity
- Average word length
- Hapax legomena

POS counts were normalized by total token counts to make the author comparisons fair across chunks of slightly different lengths.

## Results

The analysis suggests that the four authors differ in measurable stylistic ways. David Foster Wallace showed the highest average unique word count, while John Jeremiah Sullivan showed the highest lexical diversity. POS patterns also varied across authors, especially in noun and pronoun usage.

## Repository Contents

- `analysis.py` — performs the core stylistic analysis.
- `essay_scraper.py` — processes the local text files.
- `charts/` — contains the generated figures.
- `data/` — contains the essay text files used for analysis.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the preprocessing or analysis script:
   ```bash
   python essay_scraper.py
   python analysis.py
   ```

## Limitations

This is an exploratory project with a relatively small corpus. It is not intended as a full authorship attribution system, but rather as a feature-based stylistic comparison.

## Tools Used

- Python
- spaCy
- pandas
- NumPy
- matplotlib or seaborn for charts
