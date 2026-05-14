## this is an example of the scraping text used, as well as the cleanup code in order to split, standardize and organiz the essays into comparable chunks.

import os

def chunk_all_texts(input_folder, output_folder, author, chunk_size=150):
    os.makedirs(output_folder, exist_ok=True)
    file_count = 0

    for fname in os.listdir(input_folder):
        if fname.endswith(".txt"):
            path = os.path.join(input_folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            words = text.split()
            num_chunks = len(words) // chunk_size

            for i in range(num_chunks):
                chunk_words = words[i * chunk_size : (i + 1) * chunk_size]
                chunk_text = " ".join(chunk_words)

                chunk_filename = f"{author}_{os.path.splitext(fname)[0]}_chunk{i}.txt"
                chunk_path = os.path.join(output_folder, chunk_filename)

                with open(chunk_path, "w", encoding="utf-8") as out:
                    out.write(chunk_text)

                file_count += 1

chunk_all_texts(
    input_folder="essay_chunks/david_foster_wallace", 
    output_folder="essay_chunks/david_foster_wallace/chunks",
    author="david_foster_wallace",
    chunk_size=150
)
