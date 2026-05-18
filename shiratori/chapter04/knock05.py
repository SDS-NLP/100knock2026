from graphviz import Digraph
import subprocess


def visualize_tree(text):
    result = subprocess.run(["cabocha", "-f1"], input=text, text=True, capture_output=True)

    chunks = []  # 文節
    dsts = []  # 係り先

    for line in result.stdout.splitlines():
        if line == "EOS":
            break

        elif line.startswith("*"):  # 係り先
            dst = int(line.split()[2][:-1])
            dsts.append(dst)
            chunks.append("")

        else:
            if chunks:
                chunks[-1] += line.split("\t")[0]

    print(chunks)
    print(dsts)

    dot = Digraph()

    # dot.node('0', 'メロスは')
    # dot.node('1', '激怒した')

    # dot.edge('0', '1')

    for i, chunk in enumerate(chunks):
        dot.node(str(i), chunk)

    for i, dst in enumerate(dsts):
        if dst != -1:
            dot.edge(str(i), str(dst))

    dot.render("melos_tree", format="png", cleanup=True)
    print("melos_tree.png")


if __name__ == "__main__":
    visualize_tree("メロスは激怒した。")
