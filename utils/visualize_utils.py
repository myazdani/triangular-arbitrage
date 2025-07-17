#!/usr/bin/env python3
import re
import sys
import argparse
import json

# Visualization libs
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey


def parse_raw_steps(text):
    """
    Extracts each quoted step string from the input text, dropping any trailing separators.
    """
    # Grab everything inside quotes
    parts = re.findall(r'"\s*(.*?)\s*"', text, re.DOTALL)
    # Only take the first line of each (to drop separator lines)
    return [p.splitlines()[0] for p in parts]


def parse_step_line(step):
    """
    Parses a single step like "sell IOTA for USDT at 0.22900" or
    "buy DGB with USDT at 113.12217" into (src_currency, dst_currency, rate).
    """
    # Clean the step string by removing separator lines and extra whitespace
    cleaned_step = step.strip()
    # Remove any separator lines (dashes)
    cleaned_step = re.sub(r'-+', '', cleaned_step).strip()
    # Take only the first line if there are multiple lines
    cleaned_step = cleaned_step.split('\n')[0].strip()
    
    m = re.match(r'^(?P<action>sell|buy)\s+(?P<fst>\w+)\s+(?:for|with)\s+(?P<snd>\w+)\s+at\s+(?P<rate>[\d\.]+)$',
                 cleaned_step, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse step: '{cleaned_step}' (original: '{step}')")
    action = m.group('action').lower()
    first, second = m.group('fst'), m.group('snd')
    rate = float(m.group('rate'))

    # Normalize: always create an edge src->dst with given rate
    if action == 'sell':
        src, dst = first, second
    else:  # buy X with Y means use Y to get X
        src, dst = second, first
    return src, dst, rate


def build_edges(step_lines):
    """
    Converts a list of raw step strings into structured edges.
    """
    return [parse_step_line(s) for s in step_lines]


def draw_graph(edges, title=None):
    G = nx.DiGraph()
    for src, dst, rate in edges:
        G.add_edge(src, dst, label=f"{rate}")
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1500, font_size=10)
    lbls = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=lbls, font_color='red')
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def draw_sankey(edges, title=None):
    """
    Builds a Sankey flow where each step's flow width is proportional to the amount.
    """
    if not edges:
        print("No edges to visualize")
        return
    
    # Calculate the flow through each step
    current_amount = 1.0
    flow_amounts = [current_amount]
    
    for src, dst, rate in edges:
        current_amount *= rate
        flow_amounts.append(current_amount)
    
    # Create flows array: alternating positive and negative values
    flows = []
    for i, amount in enumerate(flow_amounts):
        if i == 0:
            flows.append(amount)  # Initial positive flow
        else:
            flows.append(-flow_amounts[i-1])  # Negative outflow from previous
            flows.append(amount)              # Positive inflow to current
    
    # Create labels for each flow
    labels = ["Start"]
    for src, dst, rate in edges:
        labels.append(f"{src} → {dst}")
    
    # Create orientations (all horizontal for simplicity)
    orientations = [0] * len(labels)
    
    # Ensure all arrays have the same length
    n_flows = len(flows)
    n_labels = len(labels)
    n_orientations = len(orientations)
    
    # Pad shorter arrays to match the longest one
    max_len = max(n_flows, n_labels, n_orientations)
    flows.extend([0] * (max_len - n_flows))
    labels.extend([''] * (max_len - n_labels))
    orientations.extend([0] * (max_len - n_orientations))
    
    sank = Sankey(unit=None)
    sank.add(flows=flows, labels=labels, orientations=orientations)
    if title:
        plt.gcf().suptitle(title)
    sank.finish()
    plt.show()


def draw_bar(edges, title=None):
    """
    Plots the overall cycle multiplier as a single bar vs. break-even line at 1×.
    """
    mult = 1.0
    for _, _, rate in edges:
        mult *= rate

    plt.bar(['Cycle Return'], [mult])
    plt.axhline(1, linestyle='--', label='Break-even')
    plt.ylabel('Multiplier')
    if title:
        plt.title(title)
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize a 4-step arbitrage cycle")
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin,
                        help='Text file containing your 4 quoted step lines')
    parser.add_argument('--graph', action='store_true', help='Show directed graph')
    parser.add_argument('--sankey', action='store_true', help='Show Sankey diagram')
    parser.add_argument('--bar', action='store_true', help='Show bar chart')
    parser.add_argument('--all', action='store_true', help='Show all visualizations')
    args = parser.parse_args()

    raw = args.infile.read()
    steps = parse_raw_steps(raw)
    edges = build_edges(steps)

    title = 'Arbitrage Cycle'
    if args.all or args.graph:
        draw_graph(edges, title)
    if args.all or args.sankey:
        draw_sankey(edges, title)
    if args.all or args.bar:
        draw_bar(edges, title)

    # If no viz flags, just output parsed edges
    if not (args.graph or args.sankey or args.bar or args.all):
        print(json.dumps(edges, indent=2))


if __name__ == '__main__':
    main()
