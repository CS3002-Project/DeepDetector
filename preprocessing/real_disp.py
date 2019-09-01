def preprocess_real_disp(input_path, output_path, start=None, end=None):
    start_sec = None
    start = 0 if start is None else start
    end = 117 if end is None else end
    output_file = open(output_path, "w")
    with open(input_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            tokens = line.split("\t")
            if i == 0:
                start_sec = float(tokens[0]) + float(tokens[1]) / 10.0 ** 6
                time_stamp = 0.
            else:
                time_stamp = float(tokens[0]) + float(tokens[1]) / 10.0 ** 6 - start_sec
            label = tokens[-1].strip()
            readings = tokens[2: 119][start: end]
            row = "\t".join([str(time_stamp), " ".join(readings), label])
            output_file.writelines(row + "\n")
    output_file.close()
