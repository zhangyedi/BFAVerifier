import argparse
import sys
import os

parser = argparse.ArgumentParser("Parse parameters that need to be parsed")
parser.add_argument(
    "--res_file",
    type=str,
    default="example.res",
    help="The file that contains the res log to be parsed",
)
parser.add_argument(
    "--parameters_file",
    type=str,
    default=None,
    help="The output file. default to stdout",
)

args = parser.parse_args()

para = []

with open(args.res_file, "r") as f:
    lines = f.readlines()
    for line in lines:
        # (Overall) Fail to prove 4 7 0 with all masks. Summary: 0 1 0
        # (Overall) Fail to prove 4 7 (bias) with all masks. Summary: 0 1 0
        if line.startswith("(Overall) Fail to prove"):
            line = line.strip()
            dp = line.split("Fail to prove ")[1].strip().split(" ")
            i = int(dp[0])
            j = int(dp[1])
            k = None if dp[2] == "(bias)" else int(dp[2])
            para.append((i, j, k))
    if len(para) == 0:
        para.append((1, 1, None))
    # make para unique
    para = list(set(para))
    # lambda x:(x[0],x[1],str(x[2])) < (x[0],x[1],str(x[2]))
    list.sort(para, key=lambda x: (x[0], x[1], str(x[2])))


if args.parameters_file is not None:
    with open(args.parameters_file, "w") as f:
        for p in para:
            f.write("{},{},{}\n".format(p[0] - 1, p[1], p[2] if p[2] else -1))
else:
    for p in para:
        print("{},{},{}".format(p[0] - 1, p[1], p[2] if p[2] else -1))
