# modules/data_loader.py

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_initial_data(documents_path, annotations_file):
    """
    Load and prepare initial data from documents and annotations
    """
    try:
        # Read annotations file
        annotations = pd.read_csv(
            annotations_file,
            sep="\t",
            header=None,
            names=["filename", "narrative", "subnarrative"],
        )

        # Remove prefix from narratives and subnarratives
        annotations["narrative"] = annotations["narrative"].str.replace(
            r"(CC: |URW: )", "", regex=True
        )
        annotations["subnarrative"] = annotations["subnarrative"].str.replace(
            r"(CC: |URW: )", "", regex=True
        )

        # Split into lists
        annotations["narrative"] = annotations["narrative"].str.split(";")
        annotations["subnarrative"] = annotations["subnarrative"].str.split(";")

        # Initialize data list
        data = []

        # Process each annotation
        for _, row in annotations.iterrows():
            filename = row["filename"]
            file_path = os.path.join(documents_path, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                # Determine topic from filename
                topic = "UA" if "UA" in filename else "CC"

                # Create narrative-subnarrative pairs
                pairs = []
                for narrative, subnarrative in zip(
                    row["narrative"], row["subnarrative"]
                ):
                    pairs.append(
                        {
                            "narrative": narrative.strip(),
                            "subnarrative": subnarrative.strip(),
                        }
                    )

                data.append(
                    {
                        "filename": filename,
                        "content": content,
                        "topic": topic,
                        "narrative_subnarrative_pairs": pairs,
                    }
                )

            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {filename}: {str(e)}")

        # Create DataFrame
        df = pd.DataFrame(data)

        # Remove problematic rows (mixed topics)
        df = df.drop([65, 143])

        return df

    except Exception as e:
        logger.error(f"Error in load_initial_data: {str(e)}")
        raise
