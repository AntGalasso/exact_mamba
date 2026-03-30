import os
import matplotlib.pyplot as plt


def plot_exp1_residual(res1, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    curves = res1.get("adam_curves", {})
    plt.figure()

    for lr, vals in curves.items():
        plt.plot(vals, label=f"lr={lr}")

    plt.yscale("log")
    plt.title("Exp1 - Adam Residual Curves")
    plt.xlabel("Steps")
    plt.ylabel("Residual")
    plt.legend()

    plt.savefig(os.path.join(output_dir, "exp1_residual.png"))
    plt.close()


def plot_exp2_anchor_sweep(res2, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ks = [r["k"] for r in res2]
    errs = [r.get("res", r.get("residual")) for r in res2]

    plt.figure()
    plt.plot(ks, errs, marker='o')
    plt.yscale("log")
    plt.title("Exp2 - Anchor Sweep")
    plt.xlabel("k")
    plt.ylabel("Error")

    plt.savefig(os.path.join(output_dir, "exp2_anchor_sweep.png"))
    plt.close()


def plot_exp3_singular_values(res3, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    s = res3.get("singular_values", [])

    plt.figure()
    plt.plot(s)
    plt.yscale("log")
    plt.title("Exp3 - Singular Values")
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.savefig(os.path.join(output_dir, "exp3_singular_values.png"))
    plt.close()


def plot_exp4_perplexity(res4, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_ppl = res4.get("train_perplexity", [])
    val_ppl = res4.get("val_perplexity", [])

    plt.figure()
    plt.plot(train_ppl, label="train")
    plt.plot(val_ppl, label="val")

    plt.yscale("log")
    plt.title("Exp4 - Perplexity")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.legend()

    plt.savefig(os.path.join(output_dir, "exp4_perplexity.png"))
    plt.close()


def plot_exp5_monitoring(res5, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metrics = res5.get("metrics", {})

    plt.figure()
    for name, vals in metrics.items():
        plt.plot(vals, label=name)

    plt.title("Exp5 - Algebraic Monitoring")
    plt.xlabel("Steps")
    plt.legend()

    plt.savefig(os.path.join(output_dir, "exp5_monitoring.png"))
    plt.close()


def generate_latex_tables(all_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    tex_path = os.path.join(output_dir, "reproducibility_table.tex")

    with open(tex_path, "w") as f:
        f.write("% Auto-generated LaTeX table\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\hline\n")
        f.write("Experiment & Metric & Value \\\\\n")
        f.write("\\hline\n")

        exp1 = all_results.get("exp1", {})
        f.write(f"Exp1 & Exact Residual & {exp1.get('exact_residual', 'NA')} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")