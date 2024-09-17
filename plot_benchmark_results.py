#!/usr/bin/env python3

import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from pathlib import Path

matplotlib.style.use("fivethirtyeight")
matplotlib.style.use("seaborn-v0_8-talk")
matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.facecolor"] = "white"

KERNEL_NAMES = {
    0: "hipBLAS",
    1: "Naive",
    2: "GMEM Coalescing",
    3: "SMEM Caching",
    4: "1D Blocktiling",
    5: "2D Blocktiling",
    6: "Vectorized Mem Access",
    7: "Avoid Bank Conflicts (Linearize)",
    8: "Avoid Bank Conflicts (Offset)",
    9: "Autotuning",
    10: "Warptiling"
}


def parse_file(file):
    """
    The data we want to parse has this format:

    Average elapsed time: (56.61) ms, performance: (24277.4) GFLOPS. size: (4096).
    """
    with open(file, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    data = {"size": [], "gflops": []}
    pattern = r"Average elapsed time: \((.*?)\) ms, performance: \((.*?)\) GFLOPS. size: \((.*?)\)."
    for line in lines:
        if r := re.match(pattern, line):
            data["size"].append(int(r.group(3)))
            data["gflops"].append(float(r.group(2)))
    return data


def plot(df: pd.DataFrame):
    save_dir = Path.cwd()
    df["kernel_name"] = df["kernel"].apply(lambda k: f"{k}: {KERNEL_NAMES[k]}")
    df = df.sort_values(by="kernel")

    plt.figure(figsize=(18, 10))
    colors = sn.color_palette("husl", len(df["kernel_name"].unique()))

    sn.lineplot(data=df, x="size", y="gflops", hue="kernel_name", palette=colors)
    sn.scatterplot(data=df, x="size", y="gflops", hue="kernel_name", palette=colors, legend=False)

    for kernel in df["kernel"].unique():
        kernel_data = df[df["kernel"] == kernel]
        final_size = kernel_data["size"].max()
        final_point = kernel_data[kernel_data["size"] == final_size]
        final_gflops = final_point["gflops"].values[0]
        kernel_label = f"{kernel}: {KERNEL_NAMES[kernel]}"
        plt.text(final_size, final_gflops, kernel_label, fontsize=14, verticalalignment='center')

    plt.xticks(df["size"].unique())
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")

    # Set the x-axis to log scale with base 2
    plt.xscale('log', base=2)
    plt.legend(title='Kernel (number: name)', loc='upper left')
    plt.title("Performance of different kernels")
    plt.xlabel("Matrix size (square, one side)")
    plt.ylabel("GFLOPs/s")

    plt.savefig(save_dir / "benchmark_results.png")


if __name__ == "__main__":
    results_dir = Path("benchmark_results")
    assert results_dir.is_dir()

    data = []
    for filename in results_dir.glob("*.txt"):
        # filenames have the format: <kernel_nr>_output.txt
        if not filename.stem.split("_")[0].isdigit() and "_output" not in filename.stem:
            continue
        results_dict = parse_file(filename)
        kernel_nr = int(filename.stem.split("_")[0])
        for size, gflops in zip(results_dict["size"], results_dict["gflops"]):
            data.append({"kernel": kernel_nr, "size": size, "gflops": gflops})
    df = pd.DataFrame(data)

    plot(df)

    df = df[df["size"] == 4096].sort_values(by="gflops", ascending=True)[["kernel", "gflops"]]
    df["kernel"] = df["kernel"].map({k: f"{k}: {v}" for k, v in KERNEL_NAMES.items()})
    df["relperf"] = df["gflops"] / df[df["kernel"] == "0: hipBLAS"]["gflops"].iloc[0]
    df["relperf"] = df["relperf"].apply(lambda x: f"{x*100:.1f}%")
    df.columns = ["Kernel", "GFLOPs/s", "Performance relative to hipBLAS"]

    # update the README.md with the new results
    with open("README.md", "r") as f:
        readme = f.read()
    # delete old results
    readme = re.sub(
        r"<!-- benchmark_results -->.*<!-- benchmark_results -->",
        "<!-- benchmark_results -->\n{}\n<!-- benchmark_results -->".format(
            df.to_markdown(index=False)
        ),
        readme,
        flags=re.DOTALL,
    )
    # input new results
    with open("README.md", "w") as f:
        f.write(readme)
