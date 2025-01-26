# modules/connlu_converter.py

import os
import stanza
from stanza.utils.conll import CoNLL
import logging

logger = logging.getLogger(__name__)


def convert_to_connlu(df, output_dir, column_name):
    """
    Convert normalized text to CoNLL-U format
    """
    try:
        # Initialize Stanza pipeline
        nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse")

        file_index = 1
        sentence_count = 0
        total_sentences = 0
        output_file = os.path.join(output_dir, f"output_{file_index}.conllu")

        current_file = None
        try:
            current_file = open(output_file, "w", encoding="utf-8")

            for idx, row in df.iterrows():
                sentence = " ".join(row[f"{column_name}_normalized"])
                doc = nlp(sentence)
                CoNLL.write_doc2conll(doc, current_file)
                current_file.write("\n")

                sentence_count += 1
                total_sentences += 1

                # Start new file after 10 sentences
                if sentence_count >= 10:
                    current_file.close()
                    file_index += 1
                    output_file = os.path.join(
                        output_dir, f"output_{file_index}.conllu"
                    )
                    current_file = open(output_file, "w", encoding="utf-8")
                    sentence_count = 0

        finally:
            if current_file and not current_file.closed:
                current_file.close()

        created_files = len(
            [
                name
                for name in os.listdir(output_dir)
                if name.startswith("output") and name.endswith(".conllu")
            ]
        )

        logger.info(f"Total sentences processed: {total_sentences}")
        logger.info(f"Total files created: {created_files}")

    except Exception as e:
        logger.error(f"Error in convert_to_connlu: {str(e)}")
        raise
