"""
NOTE:
The teacher forgot to multiply the percentages by 100
"""
from collections import defaultdict


def main(conll):

    sequences = []
    tags = defaultdict(int)

    with open("sample.tsv", "w") as fp:

        for line in conll:

            # skip comments
            if line.startswith("#"):
                continue

            # record end
            if line.strip() == "":
                fp.write("*\n")
                sequences.append(last_num)
                continue

            # normal line
            row = line.strip().split()
            string = "\t".join(row[2:5]) + "\n"
            fp.write(string)

            tags[row[4]] += 1
            last_num = int(row[2]) + 1

        # compute the mean
        max_length, min_length = max(sequences), min(sequences)
        mean_length = sum(sequences) / len(sequences)

        N = sum(tags.values())
        tags = sorted((t, v/N) for t, v in tags.items())
        #tags = sorted((t.v/N*100) for t, v in tags.items())

        with open("sample.info", "w") as fp:
            fp.write(f"Max sequence length: {max_length}\n")
            fp.write(f"Min sequence length: {min_length}\n")
            fp.write(f"Mean sequence length: {mean_length}\n")
            fp.write(f"Number of sequences: {len(sequences)}\n")

            fp.write("\nTags:\n")
            for t, v in tags:
                fp.write(f"{t}\t{v:.2f}%\n")


if __name__ == "__main__":

    with open("sample.conll") as fp:
        main(fp)
