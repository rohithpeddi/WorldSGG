import os
from tqdm import tqdm


def main():
    rule_embeddings_root_dir = os.path.join(os.path.dirname(__file__), '../../lattedata/cleaned/processed_sentences')

    # Loop through all files in the directory and write everything in a single file
    # With the following format:
    # --- Video ID ---
    # CONTENT OF THE FILE

    output_file_path = os.path.join(os.path.dirname(__file__), '../../lattedata/all_sentences.txt')
    with open(output_file_path, 'w') as output_file:
        for filename in tqdm(os.listdir(rule_embeddings_root_dir)):
            if filename.endswith('.txt'):
                video_id = filename[:-4].split('_')[0]
                output_file.write(f'--- {video_id} ---\n')
                with open(os.path.join(rule_embeddings_root_dir, filename), 'r') as file:
                    content = file.read()
                    output_file.write(content + '\n\n')


if __name__ == "__main__":
    main()